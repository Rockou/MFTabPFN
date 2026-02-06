import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from tabpfn import TabPFNRegressor
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)
from RCNN_MFTabPFN_S import RCNN_S
from MLP_MFTabPFN_S import MLP_S
from pathlib import Path
from TabPFN_model import TabPFN_model_main
from autogluon.tabular import TabularPredictor
import pickle
from sklearn.model_selection import RepeatedKFold, train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
import time
from tabarena.models.utils import get_configs_generator_from_name
from tabpfn_extensions.hpo import TunedTabPFNRegressor
from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import AutoTabPFNRegressor
import torch
import gc
if not torch.cuda.is_available():
    raise SystemError('GPU device not found. For fast training, please enable GPU.')

SAVE_DIR = Path(__file__).resolve().parents[1] / "Datasets" / "Single-fidelity" / "CTR23" / "ctr23_used"
TASK_INFO_PATH = SAVE_DIR / 'task_info_used.csv'
task_info_used = pd.read_csv(TASK_INFO_PATH)
csv_files = list(SAVE_DIR.glob("*.csv"))
csv_files = [f for f in csv_files if f.name != 'task_info_used.csv']
datasets = {}
for file_path in csv_files:
    try:
        filename = file_path.stem
        data_id_str = filename.split('_', 1)[0]
        data_id = int(data_id_str)
        df = pd.read_csv(file_path)
        match = task_info_used[task_info_used['data_id'] == data_id]
        target_col = match.iloc[0]['target']
        dataset_name = match.iloc[0]['dataset_name']
        samples = match.iloc[0]['samples']
        X = df.drop(columns=[target_col])
        y = df[target_col]
        datasets[data_id] = {
            "X": X,
            "y": y,
            "name": dataset_name,
            "samples": samples,
            "features": X.shape[1],
            "target": target_col,
        }
    except Exception as e:
        print(f"Failure: {e}")
print("datasets：")
for data_id, info in datasets.items():
    print(f"  [{data_id}] {info['name']} | {info['features']} features | {info['samples']} samples | target: {info['target']}")

X_list = []
y_list = []
name_list = []
for data_id in datasets.keys():
    X_list.append(datasets[data_id]['X'])
    y_list.append(datasets[data_id]['y'])
    name_list.append(datasets[data_id]['name'])


# Create a stratified 3-repeated 3-fold split
n_repeats, n_splits = 1, 3
sklearn_splits = RepeatedKFold(n_repeats=n_repeats, n_splits=n_splits, random_state=42)
def SPLITS(XX):
    splits = {}
    for split_i, (train_idx, test_idx) in enumerate(sklearn_splits.split(X=XX)):
        repeat_i = split_i // n_splits
        fold_i = split_i % n_splits
        if repeat_i not in splits:
            splits[repeat_i] = {}
        splits[repeat_i][fold_i] = (train_idx.tolist(), test_idx.tolist())
    return splits

def PREPROCESSOR(df):
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler())
        ]), num_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(sparse_output=False, handle_unknown="ignore"))
        ]), cat_cols)
    ], remainder="drop")
    return preprocessor


n_simulations = n_repeats * n_splits
Results_ctr23_list = [[None for _ in range(n_simulations)] for _ in range(len(y_list))]
Results_ctr23_prediction = [[None for _ in range(n_simulations)] for _ in range(len(y_list))]
random_seeds_train = np.array([111, 222, 333])

n_hyperparameter_configs1 = 100
num_gpus1 = 1
time_limit1 = 3600

for j in range(0, len(y_list)):
    XX_yuan = X_list[j]
    YY_yuan = y_list[j]
    splits = SPLITS(XX_yuan)
    XX_train_yuan_list = []
    XX_total_yuan_list = []
    YY_train_yuan_list = []
    YY_total_yuan_list = []
    for k1 in range(n_repeats):
        for k2 in range(n_splits):
            train_idx = splits[k1][k2][0]
            test_idx = splits[k1][k2][1]
            XX_train_yuan = XX_yuan.iloc[train_idx]
            XX_total_yuan = XX_yuan.iloc[test_idx]
            YY_train_yuan =YY_yuan.iloc[train_idx]
            YY_total_yuan = YY_yuan.iloc[test_idx]
            XX_train_yuan_list.append(XX_train_yuan)
            XX_total_yuan_list.append(XX_total_yuan)
            YY_train_yuan_list.append(YY_train_yuan)
            YY_total_yuan_list.append(YY_total_yuan)
    for k in range(n_simulations):
        print(j*10+k)
        XX_train_yuan = XX_train_yuan_list[k]
        XX_total_yuan = XX_total_yuan_list[k]
        YY_train_yuan = YY_train_yuan_list[k]
        YY_total_yuan = YY_total_yuan_list[k]
        preprocessor = PREPROCESSOR(XX_train_yuan)
        XX_train_yuan = preprocessor.fit_transform(XX_train_yuan)
        XX_total_yuan = preprocessor.transform(XX_total_yuan)
        YY_train_yuan = YY_train_yuan.to_numpy()
        YY_total_yuan = YY_total_yuan.to_numpy()
        nx = XX_train_yuan.shape[1]

        xx_train = XX_train_yuan
        xx_total = XX_total_yuan
        scaler_Y = StandardScaler()
        yy_train = scaler_Y.fit_transform(YY_train_yuan.reshape(-1, 1))
        Mu_scaler = scaler_Y.mean_
        Sigma_scaler = np.sqrt(scaler_Y.var_)

        INPUT = torch.tensor(xx_train, dtype=torch.float)
        OUTPUT = torch.tensor(yy_train, dtype=torch.float)

        input_TabPFN = np.min([np.max([20, nx]), 500])
        n_layer = 3
        hidden_channels = np.max([128, 2 * nx])
        activate_function = 'tanh'  # 'tanh'; 'sigmoid'; 'relu'
        lr = 0.001
        weight_decay = 1e-3
        epochs = 100
        bili = 1.0
        task_type = "regressor"
        batch = 256

        random_seeds = random_seeds_train[k]
        np.random.seed(random_seeds)
        # ────────────────────────────────────────────────────────────────
        # 1. TabPFN (Default)
        # ────────────────────────────────────────────────────────────────
        reg_default = TabPFNRegressor(device="cuda", random_state=random_seeds)
        start_time = time.perf_counter()
        reg_default.fit(xx_train, yy_train.ravel())
        train_time_default = time.perf_counter() - start_time

        start_time = time.perf_counter()
        YY_test_prediction_initial = reg_default.predict(xx_total)
        pred_time_default = time.perf_counter() - start_time

        # Get full output (mean + variance)
        YY_test_prediction_initial = reg_default.predict(xx_total, output_type="full")
        Y_test_prediction_initial_default = YY_test_prediction_initial["mean"] * Sigma_scaler + Mu_scaler
        Variance_initial = YY_test_prediction_initial["variance"].reshape(-1, 1)
        Prediction_sigma_initial_default = np.sqrt(Variance_initial.ravel() * Sigma_scaler ** 2)

        rmse_default = root_mean_squared_error(YY_total_yuan, Y_test_prediction_initial_default)
        mae_default = mean_absolute_error(YY_total_yuan, Y_test_prediction_initial_default)
        r2_default = r2_score(YY_total_yuan, Y_test_prediction_initial_default)
        rmse_normalized_default = 1 - rmse_default / (np.max(YY_total_yuan) - np.min(YY_total_yuan))
        mae_normalized_default = 1 - mae_default / (np.max(YY_total_yuan) - np.min(YY_total_yuan))

        results_list = []
        results_list.append({
            'Model': 'TabPFN (Default)',
            'RMSE': rmse_default, 'R2': r2_default, 'MAE': mae_default,
            'RMSE_N': rmse_normalized_default, 'MAE_N': mae_normalized_default,
            'Time_Train': train_time_default, 'Time_Pred': pred_time_default
        })

        results_prediction = []
        results_prediction.append({'Model': 'Actual', 'Performance': YY_total_yuan})
        results_prediction.append({
            'Model': 'TabPFN (Default)',
            'Performance': Y_test_prediction_initial_default,
            'Prediction_sigma_initial_default': Prediction_sigma_initial_default
        })

        # ────────────────────────────────────────────────────────────────
        # 2. Tuned TabPFN
        # ────────────────────────────────────────────────────────────────
        start_time = time.perf_counter()
        reg_tuned = TunedTabPFNRegressor(
            device="cuda",
            n_trials=100,
            random_state=random_seeds,
        )
        reg_tuned.fit(xx_train, yy_train.ravel())
        train_time_tuned = time.perf_counter() - start_time

        start_time = time.perf_counter()
        predictions_tuned = reg_tuned.predict(xx_total)
        pred_time_tuned = time.perf_counter() - start_time

        Y_test_prediction_initial_tuned = predictions_tuned * Sigma_scaler + Mu_scaler
        rmse_tuned = root_mean_squared_error(YY_total_yuan, Y_test_prediction_initial_tuned)
        mae_tuned = mean_absolute_error(YY_total_yuan, Y_test_prediction_initial_tuned)
        r2_tuned = r2_score(YY_total_yuan, Y_test_prediction_initial_tuned)
        rmse_normalized_tuned = 1 - rmse_tuned / (np.max(YY_total_yuan) - np.min(YY_total_yuan))
        mae_normalized_tuned = 1 - mae_tuned / (np.max(YY_total_yuan) - np.min(YY_total_yuan))

        results_list.append({
            'Model': 'TabPFN (Tuned)',
            'RMSE': rmse_tuned, 'R2': r2_tuned, 'MAE': mae_tuned,
            'RMSE_N': rmse_normalized_tuned, 'MAE_N': mae_normalized_tuned,
            'Time_Train': train_time_tuned, 'Time_Pred': pred_time_tuned
        })
        results_prediction.append({'Model': 'TabPFN (Tuned)', 'Performance': Y_test_prediction_initial_tuned})

        # ────────────────────────────────────────────────────────────────
        # 3. Tuned + Post-hoc Ensembled TabPFN
        # ────────────────────────────────────────────────────────────────
        start_time = time.perf_counter()
        reg_ensembled = AutoTabPFNRegressor(
            device="cuda",
            max_time=3600,
            random_state=random_seeds,
        )
        reg_ensembled.fit(xx_train, yy_train.ravel())
        train_time_ensembled = time.perf_counter() - start_time

        start_time = time.perf_counter()
        predictions_ensembled = reg_ensembled.predict(xx_total)
        pred_time_ensembled = time.perf_counter() - start_time

        Y_test_prediction_initial_ensembled = predictions_ensembled * Sigma_scaler + Mu_scaler
        rmse_ensembled = root_mean_squared_error(YY_total_yuan, Y_test_prediction_initial_ensembled)
        mae_ensembled = mean_absolute_error(YY_total_yuan, Y_test_prediction_initial_ensembled)
        r2_ensembled = r2_score(YY_total_yuan, Y_test_prediction_initial_ensembled)
        rmse_normalized_ensembled = 1 - rmse_ensembled / (np.max(YY_total_yuan) - np.min(YY_total_yuan))
        mae_normalized_ensembled = 1 - mae_ensembled / (np.max(YY_total_yuan) - np.min(YY_total_yuan))

        results_list.append({
            'Model': 'TabPFN (Tuned + Ensembled)',
            'RMSE': rmse_ensembled, 'R2': r2_ensembled, 'MAE': mae_ensembled,
            'RMSE_N': rmse_normalized_ensembled, 'MAE_N': mae_normalized_ensembled,
            'Time_Train': train_time_ensembled, 'Time_Pred': pred_time_ensembled
        })
        results_prediction.append(
            {'Model': 'TabPFN (Tuned + Ensembled)', 'Performance': Y_test_prediction_initial_ensembled})
        # ────────────────────────────────────────────────────────────────
        # 4. MFTabPFN (Default base)
        # ────────────────────────────────────────────────────────────────
        print(f"Operating model: MFTabPFN")
        criterion = torch.nn.MSELoss()

        start_time = time.perf_counter()
        TabPFN_prediction_initial = reg_default.predict(xx_train).reshape(-1, 1)
        YY_train_prediction_initial = torch.tensor(TabPFN_prediction_initial, dtype=torch.float)
        loss_train = criterion(OUTPUT, YY_train_prediction_initial)

        if loss_train > 0.01:
            # Choose MLP or RCNN depending on dimensionality
            if nx < 100:
                model_RCNN = MLP_S(task_type, INPUT, OUTPUT, input_TabPFN, bili, n_layer, lr, weight_decay,
                                   epochs, hidden_channels, activate_function, batch, TabPFN_prediction_initial,
                                   random_seeds)
            else:
                model_RCNN = RCNN_S(task_type, INPUT, OUTPUT, input_TabPFN, bili, n_layer, lr, weight_decay,
                                    epochs, hidden_channels, activate_function, batch, TabPFN_prediction_initial,
                                    random_seeds)

            train_time = time.perf_counter() - start_time + train_time_default

            model_RCNN.eval()
            start_time = time.perf_counter()

            YY_train_middle = model_RCNN(torch.tensor(xx_train, dtype=torch.float))
            YY_test_middle = model_RCNN(torch.tensor(xx_total, dtype=torch.float))

            TabPFN_prediction_initial_full = reg_default.predict(xx_total)
            TabPFN_prediction_initial = TabPFN_prediction_initial_full.reshape(-1, 1)
            TabPFN_prediction_tensor_initial = torch.tensor(TabPFN_prediction_initial, dtype=torch.float)

            TabPFN_prediction_initial0 = reg_default.predict(INPUT.detach().numpy()).reshape(-1, 1)
            TabPFN_prediction_tensor_initial0 = torch.tensor(TabPFN_prediction_initial0, dtype=torch.float)

            save_path_to_fine_tuned_model = (Path(__file__).parent / f"tabpfn-v2-{task_type}.ckpt")

            TabPFN_prediction_tensor, Lower, Upper, Variance = TabPFN_model_main(
                path_to_base_model="auto",
                save_path_to_fine_tuned_model=save_path_to_fine_tuned_model,
                X_train=YY_train_middle,
                y_train=OUTPUT - TabPFN_prediction_tensor_initial0.to('cpu'),
                X_test=YY_test_middle,
                n_classes=None,
                categorical_features_index=None,
                task_type=task_type,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )

            Y_test_prediction = (TabPFN_prediction_tensor.detach().cpu().numpy() +
                                 TabPFN_prediction_tensor_initial.detach().cpu().numpy()) * Sigma_scaler + Mu_scaler

            pred_time = time.perf_counter() - start_time

            TabPFN_prediction_initial_full = reg_default.predict(xx_total, output_type="full")
            Variance_initial = TabPFN_prediction_initial_full["variance"].reshape(-1, 1)
            Prediction_sigma_initial = np.sqrt(Variance_initial.ravel() * Sigma_scaler ** 2)
            Prediction_sigma = np.sqrt(Variance.detach().cpu().numpy().ravel() * Sigma_scaler ** 2)
            Prediction_total_sigma = np.sqrt(Prediction_sigma_initial ** 2 + Prediction_sigma ** 2)

            rmse = root_mean_squared_error(YY_total_yuan, Y_test_prediction)
            mae = mean_absolute_error(YY_total_yuan, Y_test_prediction)
            r2 = r2_score(YY_total_yuan, Y_test_prediction)
            rmse_normalized = 1 - rmse / (np.max(YY_total_yuan) - np.min(YY_total_yuan))
            mae_normalized = 1 - mae / (np.max(YY_total_yuan) - np.min(YY_total_yuan))

            results_prediction.append({
                'Model': 'MFTabPFN (Default)',
                'Performance': Y_test_prediction,
                'Prediction_total_sigma': Prediction_total_sigma,
                'Prediction_sigma_initial': Prediction_sigma_initial,
                'Prediction_sigma': Prediction_sigma
            })

            del reg_default, model_RCNN
            torch.cuda.empty_cache()
            gc.collect()

        else:
            # Skip correction if base model is already very good
            train_time = time.perf_counter() - start_time + train_time_default
            pred_time = pred_time_default
            rmse = rmse_default
            mae = mae_default
            r2 = r2_default
            rmse_normalized = rmse_normalized_default
            mae_normalized = mae_normalized_default
            results_prediction.append({
                'Model': 'MFTabPFN (Default)',
                'Performance': Y_test_prediction_initial_default,
                'Prediction_sigma_initial_default': Prediction_sigma_initial_default
            })

        results_list.append({
            'Model': 'MFTabPFN (Default)',
            'RMSE': rmse, 'R2': r2, 'MAE': mae,
            'RMSE_N': rmse_normalized, 'MAE_N': mae_normalized,
            'Time_Train': train_time, 'Time_Pred': pred_time
        })

        # ────────────────────────────────────────────────────────────────
        # 5. Tuned MFTabPFN
        # ────────────────────────────────────────────────────────────────
        print(f"Operating model: Tuned MFTabPFN")
        criterion = torch.nn.MSELoss()
        start_time = time.perf_counter()
        TabPFN_prediction_initial = reg_tuned.predict(xx_train).reshape(-1, 1)
        YY_train_prediction_initial = torch.tensor(TabPFN_prediction_initial, dtype=torch.float)
        loss_train = criterion(OUTPUT, YY_train_prediction_initial)
        if loss_train > 0.01:
            if nx < 100:
                model_RCNN = MLP_S(task_type, INPUT, OUTPUT, input_TabPFN, bili, n_layer, lr, weight_decay, epochs, hidden_channels, activate_function, batch, TabPFN_prediction_initial, random_seeds)
            else:
                model_RCNN = RCNN_S(task_type, INPUT, OUTPUT, input_TabPFN, bili, n_layer, lr, weight_decay, epochs, hidden_channels, activate_function, batch, TabPFN_prediction_initial, random_seeds)
            train_time = time.perf_counter() - start_time + train_time_tuned
            model_RCNN.eval()
            start_time = time.perf_counter()
            YY_train_middle = model_RCNN(torch.tensor(xx_train, dtype=torch.float))
            YY_test_middle = model_RCNN(torch.tensor(xx_total, dtype=torch.float))
            TabPFN_prediction_initial_full = reg_tuned.predict(xx_total)
            TabPFN_prediction_initial = TabPFN_prediction_initial_full.reshape(-1, 1)
            TabPFN_prediction_tensor_initial = torch.tensor(TabPFN_prediction_initial, dtype=torch.float)
            TabPFN_prediction_initial0 = reg_tuned.predict(INPUT.detach().numpy()).reshape(-1, 1)
            TabPFN_prediction_tensor_initial0 = torch.tensor(TabPFN_prediction_initial0, dtype=torch.float)
            save_path_to_fine_tuned_model = (Path(__file__).parent / f"tabpfn-v2-{task_type}.ckpt")
            TabPFN_prediction_tensor, Lower, Upper, Variance = TabPFN_model_main(
                path_to_base_model="auto",
                save_path_to_fine_tuned_model=save_path_to_fine_tuned_model,
                X_train=YY_train_middle,
                y_train=OUTPUT - TabPFN_prediction_tensor_initial0.to('cpu'),
                X_test=YY_test_middle,
                n_classes=None,
                categorical_features_index=None,
                task_type=task_type,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
            Y_test_prediction = (TabPFN_prediction_tensor.detach().cpu().numpy() + TabPFN_prediction_tensor_initial.detach().cpu().numpy()) * Sigma_scaler + Mu_scaler
            pred_time = time.perf_counter() - start_time
            Prediction_sigma = np.sqrt(Variance.detach().cpu().numpy().ravel() * Sigma_scaler ** 2)
            rmse = root_mean_squared_error(YY_total_yuan, Y_test_prediction)
            mae = mean_absolute_error(YY_total_yuan, Y_test_prediction)
            r2 = r2_score(YY_total_yuan, Y_test_prediction)
            rmse_normalized = 1 - rmse / (np.max(YY_total_yuan) - np.min(YY_total_yuan))
            mae_normalized = 1 - mae / (np.max(YY_total_yuan) - np.min(YY_total_yuan))
            results_prediction.append({'Model': 'MFTabPFN (Tuned)', 'Performance': Y_test_prediction, 'Prediction_sigma': Prediction_sigma})
            del reg_tuned, model_RCNN
            torch.cuda.empty_cache()
            gc.collect()
        else:
            train_time = time.perf_counter() - start_time + train_time_tuned
            pred_time = pred_time_tuned
            rmse = rmse_tuned
            mae = mae_tuned
            r2 = r2_tuned
            rmse_normalized = rmse_normalized_tuned
            mae_normalized = mae_normalized_tuned
            results_prediction.append({'Model': 'MFTabPFN (Tuned)', 'Performance': Y_test_prediction_initial_tuned})
        results_list.append({'Model': 'MFTabPFN (Tuned)', 'RMSE': rmse, 'R2': r2, 'MAE': mae, 'RMSE_N': rmse_normalized, 'MAE_N': mae_normalized, 'Time_Train': train_time, 'Time_Pred': pred_time})
        # ────────────────────────────────────────────────────────────────
        # 6. Tuned + Post-hoc Ensembled MFTabPFN
        # ────────────────────────────────────────────────────────────────
        print(f"Operating model: Tuned + Ensembled MFTabPFN")
        criterion = torch.nn.MSELoss()
        start_time = time.perf_counter()
        TabPFN_prediction_initial = reg_ensembled.predict(xx_train).reshape(-1, 1)
        YY_train_prediction_initial = torch.tensor(TabPFN_prediction_initial, dtype=torch.float)
        loss_train = criterion(OUTPUT, YY_train_prediction_initial)
        if loss_train > 0.01:
            if nx < 100:
                model_RCNN = MLP_S(task_type, INPUT, OUTPUT, input_TabPFN, bili, n_layer, lr, weight_decay, epochs, hidden_channels, activate_function, batch, TabPFN_prediction_initial, random_seeds)
            else:
                model_RCNN = RCNN_S(task_type, INPUT, OUTPUT, input_TabPFN, bili, n_layer, lr, weight_decay, epochs, hidden_channels, activate_function, batch, TabPFN_prediction_initial, random_seeds)
            train_time = time.perf_counter() - start_time + train_time_ensembled
            model_RCNN.eval()
            start_time = time.perf_counter()
            YY_train_middle = model_RCNN(torch.tensor(xx_train, dtype=torch.float))
            YY_test_middle = model_RCNN(torch.tensor(xx_total, dtype=torch.float))
            TabPFN_prediction_initial_full = reg_ensembled.predict(xx_total)
            TabPFN_prediction_initial = TabPFN_prediction_initial_full.reshape(-1, 1)
            TabPFN_prediction_tensor_initial = torch.tensor(TabPFN_prediction_initial, dtype=torch.float)
            TabPFN_prediction_initial0 = reg_ensembled.predict(INPUT.detach().numpy()).reshape(-1, 1)
            TabPFN_prediction_tensor_initial0 = torch.tensor(TabPFN_prediction_initial0, dtype=torch.float)
            save_path_to_fine_tuned_model = (Path(__file__).parent / f"tabpfn-v2-{task_type}.ckpt")
            TabPFN_prediction_tensor, Lower, Upper, Variance = TabPFN_model_main(
                path_to_base_model="auto",
                save_path_to_fine_tuned_model=save_path_to_fine_tuned_model,
                X_train=YY_train_middle,
                y_train=OUTPUT - TabPFN_prediction_tensor_initial0.to('cpu'),
                X_test=YY_test_middle,
                n_classes=None,
                categorical_features_index=None,
                task_type=task_type,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
            Y_test_prediction = (TabPFN_prediction_tensor.detach().cpu().numpy() + TabPFN_prediction_tensor_initial.detach().cpu().numpy()) * Sigma_scaler + Mu_scaler
            pred_time = time.perf_counter() - start_time
            Prediction_sigma = np.sqrt(Variance.detach().cpu().numpy().ravel() * Sigma_scaler ** 2)
            rmse = root_mean_squared_error(YY_total_yuan, Y_test_prediction)
            mae = mean_absolute_error(YY_total_yuan, Y_test_prediction)
            r2 = r2_score(YY_total_yuan, Y_test_prediction)
            rmse_normalized = 1 - rmse / (np.max(YY_total_yuan) - np.min(YY_total_yuan))
            mae_normalized = 1 - mae / (np.max(YY_total_yuan) - np.min(YY_total_yuan))
            results_prediction.append({'Model': 'MFTabPFN (Tuned + Ensembled)', 'Performance': Y_test_prediction, 'Prediction_sigma': Prediction_sigma})
            del reg_ensembled, model_RCNN
            torch.cuda.empty_cache()
            gc.collect()
        else:
            train_time = time.perf_counter() - start_time + train_time_ensembled
            pred_time = pred_time_ensembled
            rmse = rmse_ensembled
            mae = mae_ensembled
            r2 = r2_ensembled
            rmse_normalized = rmse_normalized_ensembled
            mae_normalized = mae_normalized_ensembled
            results_prediction.append({'Model': 'MFTabPFN (Tuned + Ensembled)', 'Performance': Y_test_prediction_initial_ensembled})
        results_list.append({'Model': 'MFTabPFN (Tuned + Ensembled)', 'RMSE': rmse, 'R2': r2, 'MAE': mae, 'RMSE_N': rmse_normalized, 'MAE_N': mae_normalized, 'Time_Train': train_time, 'Time_Pred': pred_time})
        # ────────────────────────────────────────────────────────────────
        # 7. Various ML models via TabArena / AutoGluon style
        # ────────────────────────────────────────────────────────────────
        TRAIN_Autogluon = pd.DataFrame(xx_train, columns=[f'feature_{i}' for i in range(nx)])
        TRAIN_Autogluon['target'] = yy_train
        TEST_Autogluon = pd.DataFrame(xx_total, columns=[f'feature_{i}' for i in range(nx)])
        recommended_models = [
            "RealMLP",
            "TabM",
            "LightGBM",
            "CatBoost",
            "XGBoost",
            "ModernNCA",
            "TorchMLP",
            "TabDPT",
            "EBM",
            "FastaiMLP",
            "ExtraTrees",
        ]
        for model_to_run in recommended_models:
            print(f"Operating model: {model_to_run}")
            n_hyperparameter_configs = n_hyperparameter_configs1
            use_ensemble_model = True
            model_meta = get_configs_generator_from_name(model_name=model_to_run)
            model_cls = model_meta.model_cls
            hpo_configs = model_meta.generate_all_configs_lst(num_random_configs=n_hyperparameter_configs)
            num_bag_folds = 8
            if model_to_run == "TabDPT":
                use_ensemble_model = False
                hpo_configs = model_meta.generate_all_configs_lst(num_random_configs=1)[0]
            elif model_to_run == "KNN":
                hpo_configs = model_meta.generate_all_configs_lst(num_random_configs=59)
            model_hyperparameters = {model_cls: hpo_configs}
            task_type_auto = "regression"
            num_gpus = num_gpus1 if model_to_run in ["TabM", "ModernNCA", "TabDPT"] else 0
            model = TabularPredictor(
                label="target",
                eval_metric="rmse" if task_type_auto == "regression" else "accuracy",
                problem_type=task_type_auto,
            ).fit(
                TRAIN_Autogluon,
                fit_weighted_ensemble=use_ensemble_model,
                hyperparameters=model_hyperparameters,
                num_bag_folds=num_bag_folds,
                time_limit=time_limit1,
                num_gpus=num_gpus,
            )
            leaderboard = model.leaderboard(silent=False)
            print(f"{model_to_run} leaderboard:\n", leaderboard)
            default_candidates = leaderboard[
                leaderboard['model'].str.contains('c1') & ~leaderboard['model'].str.contains('Weighted')]
            default_model = default_candidates.iloc[0]['model'] if not default_candidates.empty else None
            tuned_candidates = leaderboard[
                leaderboard['model'].str.contains('c1|r') & ~leaderboard['model'].str.contains('Weighted')]
            best_tuned_model = tuned_candidates.loc[
                tuned_candidates['score_val'].idxmax(), 'model'] if not tuned_candidates.empty else None
            ensemble_candidates = leaderboard[leaderboard['model'].str.contains('Weighted')]
            weighted_model = ensemble_candidates.iloc[0]['model'] if not ensemble_candidates.empty else None
            if model_to_run == "TabDPT":
                best_tuned_model = None
                weighted_model = None
            if weighted_model:
                start_time = time.perf_counter()
                y_pred_ensemble = model.predict(data=TEST_Autogluon, model=weighted_model)
                ensemble_pred_time = time.perf_counter() - start_time
            else:
                y_pred_ensemble = None
                ensemble_pred_time = 0
            if best_tuned_model:
                start_time = time.perf_counter()
                y_pred_tuned = model.predict(data=TEST_Autogluon, model=best_tuned_model)
                tuned_pred_time = time.perf_counter() - start_time
            else:
                y_pred_tuned = None
                tuned_pred_time = 0
            if default_model:
                start_time = time.perf_counter()
                y_pred_default = model.predict(data=TEST_Autogluon, model=default_model)
                default_pred_time = time.perf_counter() - start_time
            else:
                y_pred_default = None
                default_pred_time = 0
            default_train_time = default_candidates['fit_time'].iloc[0] if not default_candidates.empty else 0
            tuned_train_time = tuned_candidates['fit_time'].sum() if not tuned_candidates.empty else 0
            ensemble_train_time = tuned_train_time + (
                ensemble_candidates['fit_time'].iloc[0] if not ensemble_candidates.empty else 0)
            if default_model:
                Y_test_prediction_default = y_pred_default.to_numpy() * Sigma_scaler + Mu_scaler
                rmse_default = root_mean_squared_error(YY_total_yuan, Y_test_prediction_default)
                mae_default = mean_absolute_error(YY_total_yuan, Y_test_prediction_default)
                r2_default = r2_score(YY_total_yuan, Y_test_prediction_default)
                rmse_normalized_default = 1 - rmse_default / (np.max(YY_total_yuan) - np.min(YY_total_yuan))
                mae_normalized_default = 1 - mae_default / (np.max(YY_total_yuan) - np.min(YY_total_yuan))
                results_list.append({
                    'Model': f'{model_to_run} (Default)',
                    'RMSE': rmse_default,
                    'R2': r2_default,
                    'MAE': mae_default,
                    'RMSE_N': rmse_normalized_default,
                    'MAE_N': mae_normalized_default,
                    'Time_Train': default_train_time,
                    'Time_Pred': default_pred_time
                })
                results_prediction.append({'Model': f'{model_to_run} (Default)', 'Performance': Y_test_prediction_default})
            if best_tuned_model:
                Y_test_prediction_tuned = y_pred_tuned.to_numpy() * Sigma_scaler + Mu_scaler
                rmse_tuned = root_mean_squared_error(YY_total_yuan, Y_test_prediction_tuned)
                mae_tuned = mean_absolute_error(YY_total_yuan, Y_test_prediction_tuned)
                r2_tuned = r2_score(YY_total_yuan, Y_test_prediction_tuned)
                rmse_normalized_tuned = 1 - rmse_tuned / (np.max(YY_total_yuan) - np.min(YY_total_yuan))
                mae_normalized_tuned = 1 - mae_tuned / (np.max(YY_total_yuan) - np.min(YY_total_yuan))
                results_list.append({
                    'Model': f'{model_to_run} (Tuned)',
                    'RMSE': rmse_tuned,
                    'R2': r2_tuned,
                    'MAE': mae_tuned,
                    'RMSE_N': rmse_normalized_tuned,
                    'MAE_N': mae_normalized_tuned,
                    'Time_Train': tuned_train_time,
                    'Time_Pred': tuned_pred_time
                })
                results_prediction.append({'Model': f'{model_to_run} (Tuned)', 'Performance': Y_test_prediction_tuned})
            if weighted_model:
                Y_test_prediction_ensemble = y_pred_ensemble.to_numpy() * Sigma_scaler + Mu_scaler
                rmse_ensemble = root_mean_squared_error(YY_total_yuan, Y_test_prediction_ensemble)
                mae_ensemble = mean_absolute_error(YY_total_yuan, Y_test_prediction_ensemble)
                r2_ensemble = r2_score(YY_total_yuan, Y_test_prediction_ensemble)
                rmse_normalized_ensemble = 1 - rmse_ensemble / (np.max(YY_total_yuan) - np.min(YY_total_yuan))
                mae_normalized_ensemble = 1 - mae_ensemble / (np.max(YY_total_yuan) - np.min(YY_total_yuan))
                results_list.append({
                    'Model': f'{model_to_run} (Tuned + Ensembled)',
                    'RMSE': rmse_ensemble,
                    'R2': r2_ensemble,
                    'MAE': mae_ensemble,
                    'RMSE_N': rmse_normalized_ensemble,
                    'MAE_N': mae_normalized_ensemble,
                    'Time_Train': ensemble_train_time,
                    'Time_Pred': ensemble_pred_time
                })
                results_prediction.append(
                    {'Model': f'{model_to_run} (Tuned + Ensembled)', 'Performance': Y_test_prediction_ensemble})
            del model
            del leaderboard
            gc.collect()
            torch.cuda.empty_cache()
        # ────────────────────────────────────────────────────────────────
        # 8. Full AutoGluon (best_quality preset)
        # ────────────────────────────────────────────────────────────────
        TRAIN_Autogluon = pd.DataFrame(xx_train, columns=[f'feature_{i}' for i in range(nx)])
        TRAIN_Autogluon['target'] = yy_train
        TEST_Autogluon = pd.DataFrame(xx_total, columns=[f'feature_{i}' for i in range(nx)])
        start_time = time.perf_counter()
        predictor = TabularPredictor(
            label="target",
            problem_type="regression",
        )
        predictor.fit(train_data=TRAIN_Autogluon, time_limit=time_limit1, presets='best_quality')
        train_time = time.perf_counter() - start_time
        start_time = time.perf_counter()
        Y_test_prediction_initial = predictor.predict(TEST_Autogluon).to_numpy() * Sigma_scaler + Mu_scaler
        pred_time = time.perf_counter() - start_time
        rmse = root_mean_squared_error(YY_total_yuan, Y_test_prediction_initial)
        mae = mean_absolute_error(YY_total_yuan, Y_test_prediction_initial)
        r2 = r2_score(YY_total_yuan, Y_test_prediction_initial)
        rmse_normalized = 1 - rmse / (np.max(YY_total_yuan) - np.min(YY_total_yuan))
        mae_normalized = 1 - mae / (np.max(YY_total_yuan) - np.min(YY_total_yuan))
        results_list.append(
            {'Model': 'AutoGluon', 'RMSE': rmse, 'R2': r2, 'MAE': mae, 'RMSE_N': rmse_normalized, 'MAE_N': mae_normalized,
             'Time_Train': train_time, 'Time_Pred': pred_time})
        results_prediction.append({'Model': 'AutoGluon', 'Performance': Y_test_prediction_initial})
        Results_ctr23_list[j][k] = pd.DataFrame(results_list)
        Results_ctr23_prediction[j][k] = pd.DataFrame(results_prediction)
        with open('Results_ctr23_list.pkl', 'wb') as f:
            pickle.dump(Results_ctr23_list, f)
        with open('Results_ctr23_prediction.pkl', 'wb') as f:
            pickle.dump(Results_ctr23_prediction, f)
        del predictor
        del xx_train, xx_total, yy_train, YY_total_yuan
        del INPUT, OUTPUT
        del preprocessor, scaler_Y
        gc.collect()
        torch.cuda.empty_cache()

