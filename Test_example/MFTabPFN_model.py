import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from tabpfn import TabPFNRegressor
import numpy as np
from sklearn.preprocessing import StandardScaler
from MLP_MFTabPFN_S import MLP_S
from RCNN_MFTabPFN_S import RCNN_S
from RCNN_MFTabPFN_M import RCNN_M
from MLP_MFTabPFN_M import MLP_M
from pathlib import Path
from TabPFN_model import TabPFN_model_main
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
import pandas as pd
import torch
if not torch.cuda.is_available():
    raise SystemError('GPU device not found. For fast training, please enable GPU.')

def PREPROCESSOR(df):
    """
    Create a preprocessing pipeline for numerical and categorical features:
    - Numerical: median imputation + robust scaling
    - Categorical: most frequent imputation + one-hot encoding
    """
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

def MFTabPFN_SY(XX_train_yuan, YY_train_yuan, XX_total_yuan, random_seeds, N_train, N_high=None):

    N_low = N_train
    N_high = N_high

    nx = XX_train_yuan.shape[1]
    xx_train1 = pd.DataFrame(XX_train_yuan)
    xx_total1 = pd.DataFrame(XX_total_yuan)
    preprocessor = PREPROCESSOR(xx_train1)
    xx_train1 = preprocessor.fit_transform(xx_train1)
    xx_total1 = preprocessor.transform(xx_total1)

    xx_train = xx_train1
    xx_total = xx_total1
    scaler_Y = StandardScaler()
    yy_train = scaler_Y.fit_transform(YY_train_yuan.reshape(-1, 1))
    Mu_scaler = scaler_Y.mean_
    Sigma_scaler = np.sqrt(scaler_Y.var_)

    INPUT = torch.tensor(xx_train, dtype=torch.float)
    OUTPUT = torch.tensor(yy_train, dtype=torch.float)

    input_TabPFN = np.min([np.max([20, nx]), 500])
    n_layer = 3
    if nx < 20:
        n_layer = 2
    hidden_channels = np.max([128, 2 * nx])
    activate_function = 'tanh'
    lr = 0.001
    weight_decay = 1e-3
    epochs = 100
    bili = 1.0
    task_type = "regressor"
    batch = 256
    ##########################################################################MFTabPFN
    if N_high == None:
        criterion = torch.nn.MSELoss()
        reg = TabPFNRegressor(device="cuda", random_state=random_seeds)
        reg.fit(xx_train, yy_train.ravel())
        TabPFN_prediction_initial = reg.predict(xx_train).reshape(-1, 1)
        YY_train_prediction_initial = torch.tensor(TabPFN_prediction_initial, dtype=torch.float)
        loss_train = criterion(OUTPUT, YY_train_prediction_initial)
        if loss_train > 0.01:
            if nx < 100:
                model_RCNN = MLP_S(task_type, INPUT, OUTPUT, input_TabPFN, bili, n_layer, lr, weight_decay,
                                   epochs, hidden_channels, activate_function, batch, TabPFN_prediction_initial,
                                   random_seeds)
            else:
                model_RCNN = RCNN_S(task_type, INPUT, OUTPUT, input_TabPFN, bili, n_layer, lr, weight_decay,
                                    epochs, hidden_channels, activate_function, batch, TabPFN_prediction_initial,
                                    random_seeds)
            model_RCNN.eval()
            YY_train_middle = model_RCNN(torch.tensor(xx_train, dtype=torch.float))
            YY_test_middle = model_RCNN(torch.tensor(xx_total, dtype=torch.float))
            TabPFN_prediction_initial_full = reg.predict(xx_total)
            TabPFN_prediction_initial = TabPFN_prediction_initial_full.reshape(-1, 1)
            TabPFN_prediction_tensor_initial = torch.tensor(TabPFN_prediction_initial, dtype=torch.float)
            TabPFN_prediction_initial0 = reg.predict(INPUT.detach().numpy()).reshape(-1, 1)
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
            Y_test_prediction_MFTabPFN = Y_test_prediction.ravel()
            TabPFN_prediction_initial_full = reg.predict(xx_total, output_type="full")
            Variance_initial = TabPFN_prediction_initial_full[f"variance"].reshape(-1, 1)
            Prediction_sigma_initial = np.sqrt(Variance_initial.ravel() * Sigma_scaler ** 2)
            Prediction_sigma = np.sqrt(Variance.detach().cpu().numpy().ravel() * Sigma_scaler ** 2)
            Prediction_total_sigma = np.sqrt(Prediction_sigma_initial ** 2 + Prediction_sigma ** 2)
            Y_test_prediction_TabPFN = TabPFN_prediction_initial.ravel() * Sigma_scaler + Mu_scaler
        else:
            TabPFN_prediction_initial_full = reg.predict(xx_total, output_type="full")
            Y_test_prediction = TabPFN_prediction_initial_full[f"mean"].reshape(-1, 1)
            Y_test_prediction_MFTabPFN = Y_test_prediction.ravel() * Sigma_scaler + Mu_scaler
            Y_test_prediction_TabPFN = Y_test_prediction.ravel() * Sigma_scaler + Mu_scaler
            Variance_initial = TabPFN_prediction_initial_full[f"variance"].reshape(-1, 1)
            Prediction_sigma_initial = np.sqrt(Variance_initial.ravel() * Sigma_scaler ** 2)
            Prediction_total_sigma = Prediction_sigma_initial
            model_RCNN = None

    else:
        criterion = torch.nn.MSELoss()
        reg = TabPFNRegressor(device="cuda", random_state=random_seeds)
        reg.fit(xx_train[:N_low, :], yy_train[:N_low].ravel())
        TabPFN_prediction_initial = reg.predict(xx_train[N_low:, :]).reshape(-1, 1)
        YY_train_prediction_initial = torch.tensor(TabPFN_prediction_initial, dtype=torch.float)
        loss_train = criterion(OUTPUT[N_low:], YY_train_prediction_initial)
        TabPFN_prediction_initial11111 = TabPFN_prediction_initial
        alpha_index = 0
        alpha = 1
        if loss_train > 0.01:
            if nx < 100:
                model_RCNN = MLP_M(task_type, INPUT, OUTPUT, input_TabPFN, bili, n_layer, lr, weight_decay, epochs,
                                   hidden_channels, activate_function, batch, N_low, N_high,
                                   TabPFN_prediction_initial11111,
                                   random_seeds, alpha, alpha_index)
            else:
                model_RCNN = RCNN_M(task_type, INPUT, OUTPUT, input_TabPFN, bili, n_layer, lr, weight_decay, epochs,
                                    hidden_channels, activate_function, batch, N_low, N_high,
                                    TabPFN_prediction_initial11111,
                                    random_seeds, alpha, alpha_index)
            model_RCNN.eval()
            YY_train_middle = model_RCNN(torch.tensor(xx_train[N_low:, :], dtype=torch.float))
            YY_test_middle = model_RCNN(torch.tensor(xx_total, dtype=torch.float))
            TabPFN_prediction_initial_full = reg.predict(xx_total, output_type="full")
            TabPFN_prediction_initial = TabPFN_prediction_initial_full[f"mean"].reshape(-1, 1)
            Variance_initial = TabPFN_prediction_initial_full[f"variance"].reshape(-1, 1)
            Prediction_sigma_initial = np.sqrt(Variance_initial.ravel() * Sigma_scaler ** 2)
            TabPFN_prediction_tensor_initial = torch.tensor(TabPFN_prediction_initial, dtype=torch.float)
            TabPFN_prediction_initial0 = reg.predict(xx_train[N_low:, :]).reshape(-1, 1)
            TabPFN_prediction_tensor_initial0 = torch.tensor(TabPFN_prediction_initial0, dtype=torch.float)
            save_path_to_fine_tuned_model = (Path(__file__).parent / f"tabpfn-v2-{task_type}.ckpt")
            TabPFN_prediction_tensor, Lower, Upper, Variance = TabPFN_model_main(
                path_to_base_model="auto",
                save_path_to_fine_tuned_model=save_path_to_fine_tuned_model,
                X_train=YY_train_middle,
                y_train=OUTPUT[N_low:] - model_RCNN.afa * TabPFN_prediction_tensor_initial0.to('cpu'),
                X_test=YY_test_middle,
                n_classes=None,
                categorical_features_index=None,
                task_type=task_type,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
            Y_test_prediction_delt = TabPFN_prediction_tensor.detach().cpu().numpy() * Sigma_scaler
            Y_test_prediction_low_afa = model_RCNN.afa.detach().cpu().numpy() * TabPFN_prediction_tensor_initial.detach().cpu().numpy() * Sigma_scaler + Mu_scaler
            Y_test_prediction = (TabPFN_prediction_tensor.detach().cpu().numpy() + model_RCNN.afa.detach().cpu().numpy() * TabPFN_prediction_tensor_initial.detach().cpu().numpy()) * Sigma_scaler + Mu_scaler
            Prediction_sigma = np.sqrt(Variance.detach().cpu().numpy().ravel() * Sigma_scaler ** 2)
            Prediction_total_sigma = np.sqrt(model_RCNN.afa.detach().cpu().numpy() ** 2 * Prediction_sigma_initial ** 2 + Prediction_sigma ** 2)
            Y_test_prediction_MFTabPFN = Y_test_prediction.ravel()
            Y_test_prediction_TabPFN = TabPFN_prediction_initial.ravel() * Sigma_scaler + Mu_scaler
        else:
            reg.fit(xx_train, yy_train.ravel())
            TabPFN_prediction_initial_full = reg.predict(xx_total, output_type="full")
            TabPFN_prediction_initial = TabPFN_prediction_initial_full[f"mean"].reshape(-1, 1)
            Variance_initial = TabPFN_prediction_initial_full[f"variance"].reshape(-1, 1)
            Prediction_sigma_initial = np.sqrt(Variance_initial.ravel() * Sigma_scaler ** 2)
            Prediction_total_sigma = Prediction_sigma_initial
            Y_test_prediction = TabPFN_prediction_initial * Sigma_scaler + Mu_scaler
            Y_test_prediction_MFTabPFN = Y_test_prediction.ravel()
            Y_test_prediction_TabPFN = Y_test_prediction_MFTabPFN
            model_RCNN = None

    return Y_test_prediction_MFTabPFN, Prediction_total_sigma, Y_test_prediction_TabPFN, Prediction_sigma_initial, model_RCNN



