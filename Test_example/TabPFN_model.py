from __future__ import annotations
import logging
import random
import warnings
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Union, Sequence
import numpy as np
import pandas as pd
import torch
from finetuning_scripts.constant_utils import (
    SupportedDevice,
    TaskType,
)

from tabpfn.base import load_model_criterion_config
from torch import autocast
from torch.nn import DataParallel

if TYPE_CHECKING:
    from tabpfn.model.transformer import PerFeatureTransformer

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=".*input value tensor is non-contiguous.*",
)


def TabPFN_model_main(
    *,
    path_to_base_model: Path | Literal["auto"] = "auto",
    save_path_to_fine_tuned_model: Path,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    n_classes: int | None = None,
    categorical_features_index: list[int] | None,
    task_type: TaskType,
    device: SupportedDevice,
    use_multiple_gpus: bool = False,
    multiple_device_ids: Sequence[Union[int, torch.device]] | None  = None,
    random_seed: int = 42,
) -> torch.Tensor:

    # Control randomness
    rng = np.random.RandomState(random_seed)
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch_rng = torch.Generator()
    torch_rng.manual_seed(random_seed)

    # Meta
    is_classification = task_type != TaskType.REGRESSION
    use_autocast = False
    if device == SupportedDevice.GPU:
        use_autocast = True
    use_grad_scaler = use_autocast

    model, criterion, checkpoint_config = load_model_criterion_config(
        model_path=save_path_to_fine_tuned_model,
        check_bar_distribution_criterion=False,
        cache_trainset_representation=False,
        which="classifier" if is_classification else "regressor",
        version="v2",
        download=True,
    )

    model.criterion = criterion
    checkpoint_config = checkpoint_config.__dict__
    is_data_parallel = False
    if device == 'cuda' and use_multiple_gpus and torch.cuda.device_count() > 1:
        model = DataParallel(model, device_ids=multiple_device_ids)
        is_data_parallel = True
    model.to(device)

    model_forward_fn = partial(
        _model_forward,
        n_classes=n_classes,
        categorical_features_index=categorical_features_index,
        use_autocast=use_autocast,
        device=device,
        is_data_parallel=is_data_parallel,
    )

    if n_classes is not None:
        X_train_test = torch.cat((X_train, X_test), dim=0)
        pred_logits = model_forward_fn(
            model=model,
            X_train_test=X_train_test.reshape(X_train_test.shape[0], 1, X_train_test.shape[1]).to(device),#float()
            y_train=y_train.reshape(y_train.shape[0], 1, 1).to(device),
            X_test=X_test.reshape(X_test.shape[0], 1, X_test.shape[1]).to(device),
            forward_for_validation=True,
        )
    else:

        X_train_test = torch.cat((X_train, X_test), dim=0)
        pred_logits, pred_lowers, pred_uppers, pred_variances = model_forward_fn(
            model=model,
            X_train_test=X_train_test.reshape(X_train_test.shape[0], 1, X_train_test.shape[1]).to(device),#float()
            y_train=y_train.reshape(y_train.shape[0], 1, 1).to(device),
            forward_for_validation=True,
        )

    if n_classes is not None:
        return pred_logits
    else:
        return pred_logits, pred_lowers, pred_uppers, pred_variances


def _model_forward(
    *,
    model: PerFeatureTransformer,
    X_train_test: torch.Tensor,  # (n_samples, batch_size, n_features)
    y_train: torch.Tensor,  # (n_samples, batch_size, 1)
    #X_test: torch.Tensor,  # (n_samples, batch_size, n_features)
    n_classes: int | None,
    # softmax_temperature: torch.Tensor | None = None,
    softmax_temperature: torch.Tensor = 0.9,
    categorical_features_index: list[torch.Tensor] | None,
    use_autocast: bool = True,
    forward_for_validation: bool = False,
    device: SupportedDevice,
    outer_loop_autocast: bool = False,
    is_data_parallel: bool,
) -> torch.Tensor:
    """Wrapper function to perform a forward pass with a TabPFN model.

    Arguments:
    ----------
    model: PerFeatureTransformer
        The model to use for the forward pass.
    X_train_test: torch.Tensor
        The training + test features.
    y_train: torch.Tensor
        The training target.
    n_classes: int | None
        The number of classes for classification tasks, otherwise None.
    softmax_temperature: torch.Tensor | None
        The softmax temperature for the model, used to scale the logits.
        If None, no scaling is applied.
    categorical_features_index: list[int] | None
        The indices of the categorical features.
    use_autocast: bool
        Whether to use FP16 precision for the forward pass.
        This is required for flash attention!
    forward_for_validation: boo
        If True, this indicates that this is a forward pass for a validation score.
        This means that a regression model will return predictions instead of logits for the bar distribution.
    device: SupportedDevice
        The device to use for autocasting in the forward pass.

    Returns:
    --------
    pred_logits: torch.Tensor
        The predicted logits of the model. Logits are softmax scaled and selected down to:
            - classification: (n_samples, batch_size, n_classes)
            - regression: (n_samples, batch_size)
    """
    is_classification = n_classes is not None
    if not is_classification:
        # TabPFN model assumes z-normalized inputs.
        mean = y_train.mean(dim=0)
        std = y_train.std(dim=0)
        y_train = (y_train - mean) / std

    forward_kwargs = dict(
        x=X_train_test,
        y=y_train,
    )

    if outer_loop_autocast:
        pred_logits = model(**forward_kwargs)
    else:
        with autocast(device_type=device, enabled=use_autocast):
            pred_logits = model(**forward_kwargs)

    if is_classification:
        pred_logits = pred_logits[:, :, :n_classes].float()

        if softmax_temperature is not None:
            pred_logits = pred_logits / softmax_temperature

        pred_logits = torch.nn.functional.softmax(pred_logits[:, 0, :], dim=-1)

    else:
        pred_logits = pred_logits.float()

        if softmax_temperature is not None:
            pred_logits = pred_logits / softmax_temperature

        if forward_for_validation:
            new_pred_logits = []
            new_pred_lowers = []
            new_pred_uppers = []
            new_pred_variances = []
            for batch_i in range(pred_logits.shape[1]):
                bar_dist = deepcopy(model.module.criterion if is_data_parallel else model.criterion)
                bar_dist.borders = (
                    bar_dist.borders * std[batch_i] + mean[batch_i]
                ).float()
                new_pred_logits.append(bar_dist.mean(pred_logits[:, batch_i, :]))
                # #####################################Variance
                Variance = bar_dist.variance(pred_logits[:, batch_i, :])
                new_pred_variances.append(Variance)
                # #####################################Quantile
                Quantile_left1 = bar_dist.icdf(pred_logits[:, batch_i, :], left_prob=0.1)
                Quantile_left2 = bar_dist.icdf(pred_logits[:, batch_i, :], left_prob=0.9)
                new_pred_lowers.append(Quantile_left1)
                new_pred_uppers.append(Quantile_left2)

            pred_logits = torch.stack(new_pred_logits, dim=-1)
            pred_lowers = torch.stack(new_pred_lowers, dim=-1)
            pred_uppers = torch.stack(new_pred_uppers, dim=-1)
            pred_variances = torch.stack(new_pred_variances, dim=-1)

    if is_classification:
        return pred_logits
    else:
        return pred_logits, pred_lowers, pred_uppers, pred_variances
