import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from tabpfn import TabPFNRegressor
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import scipy.stats as stats
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
import torch
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
import time
import gc
if not torch.cuda.is_available():
    raise SystemError('GPU device not found. For fast training, please enable GPU.')


def dis(Case):
    """
    Generate distribution parameters (mu, sigma, type) for different high-dimensional test cases.
    """
    if Case == 0:
        mu = np.full(100, 0.5)
        sigma = np.full(100, 1/np.sqrt(12))
        DisType = ['uniform'] * 100
    elif Case == 1:
        mu = np.full(200, 0.5)
        sigma = np.full(200, 1/np.sqrt(12))
        DisType = ['uniform'] * 200
    elif Case == 2:
        mu = np.full(300, 0.5)
        sigma = np.full(300, 1/np.sqrt(12))
        DisType = ['uniform'] * 300
    elif Case == 3:
        mu = np.full(400, 0.5)
        sigma = np.full(400, 1/np.sqrt(12))
        DisType = ['uniform'] * 400
    elif Case == 4:
        mu = np.full(500, 0.5)
        sigma = np.full(500, 1/np.sqrt(12))
        DisType = ['uniform'] * 500
    return mu, sigma, DisType

def func(x, Case):
    """
        High-dimensional synthetic test functions (benchmark problems) used in the experiments.
        Different cases correspond to different function forms with increasing dimensionality.
        """
    if Case == 0:
        ff = np.sum([(x[:, k] - 1) ** 4 for k in range(0, 99)], axis=0) + np.sum([np.sqrt(k + 1) * (x[:, k] - x[:, k - 1] ** 2) ** 2 for k in range(1, 100)], axis=0)
    elif Case == 1:
        ff = 10 * (x[:, 1] - x[:, 0] ** 2) ** 5 + np.sum([(x[:, k + 1] - x[:, k] ** 2) ** 5 for k in range(1, 199)], axis=0)
    elif Case == 2:
        ff = (x[:, 0] - 1) ** 2 + np.sum([k * (2 * x[:, k] ** 2 - x[:, k - 1]) ** 2 for k in range(1, 300)], axis=0)
    elif Case == 3:
        ff = (x[:, 0] ** 2 + 4) * (x[:, 1] - 1) / 20 + np.cos(x[:, 0]) + np.sum([x[:, k] ** 2 * x[:, k + 1] ** 2 for k in range(0, 399)], axis=0) - 100
    elif Case == 4:
        ff = np.sum([x[:, k] for k in range(0, 500)], axis=0) + 20 * x[:, 0] ** 2 * x[:, 1] ** 2 + np.sum([x[:, k] ** 2 * x[:, k + 1] ** 2 for k in range(1, 499)], axis=0) - np.sum([np.sin(x[:, k]) * np.exp(x[:, k] - 2) for k in range(0, 500)], axis=0) - 10
    return ff


def UtoX(u, mu, sigma, DisType):
    x = np.zeros_like(u)
    for i in range(len(mu)):
        if DisType[i] == 'normal':
            x[:, i] = u[:, i] * sigma[i] + mu[i]
        elif DisType[i] == 'uniform':
            a = mu[i] - np.sqrt(3) * sigma[i]
            b = mu[i] + np.sqrt(3) * sigma[i]
            x[:, i] = (b - a) * stats.norm.cdf(u[:, i]) + a
        elif DisType[i] == 'lognorm':
            sigma1 = np.sqrt(np.log(1 + sigma[i]**2 / mu[i]**2))
            mu1 = np.log(mu[i]) - 0.5 * sigma1**2
            x[:, i] = stats.lognorm.ppf(stats.norm.cdf(u[:, i]), s=sigma1, loc=0, scale=np.exp(mu1))
    return x
def XtoU(x, mu, sigma, DisType):
    u = np.zeros_like(x)
    for i in range(len(mu)):
        if DisType[i] == 'normal':
            u[:, i] = (x[:, i] - mu[i]) / sigma[i]
        elif DisType[i] == 'uniform':
            a = mu[i] - np.sqrt(3) * sigma[i]
            b = mu[i] + np.sqrt(3) * sigma[i]
            u[:, i] = stats.norm.ppf((x[:, i] - a) / (b - a))
        elif DisType[i] == 'lognorm':
            sigma1 = np.sqrt(np.log(1 + sigma[i] ** 2 / mu[i] ** 2))
            mu1 = np.log(mu[i]) - 0.5 * sigma1 ** 2
            u[:, i] = stats.norm.ppf(stats.lognorm.cdf(x[:, i], s=sigma1, loc=0, scale=np.exp(mu1)))
    return u

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


n_performances = 5
n_simulations = 3
YY_index = [[None for _ in range(n_simulations)] for _ in range(n_performances)]
YY_real_prediction = [[None for _ in range(n_simulations)] for _ in range(n_performances)]
random_seeds_train = np.array([111, 222, 333])

for j in range(0, n_performances):
    Case = j
    mu, sigma, DisType = dis(Case)
    nx = mu.shape[0]
    N_train = 5 * nx
    N = 5000
    np.random.seed(123)
    XX_total = np.random.normal(0, 1, size=(N, nx))
    XX_total_yuan = UtoX(XX_total, mu, sigma, DisType)
    YY_total_yuan = func(XX_total_yuan, Case)
    XX_total_yuan = pd.DataFrame(XX_total_yuan)
    YY_total_yuan = pd.Series(YY_total_yuan)
    XX_train_yuan_list = []
    XX_total_yuan_list = []
    YY_train_yuan_list = []
    YY_total_yuan_list = []
    for k in range(n_simulations):
        np.random.seed(random_seeds_train[k])
        XX_train = np.random.normal(0, 1, size=(N_train, nx))
        XX_train_yuan = UtoX(XX_train, mu, sigma, DisType)
        YY_train_yuan = func(XX_train_yuan, Case)
        XX_train_yuan = pd.DataFrame(XX_train_yuan)
        YY_train_yuan = pd.Series(YY_train_yuan)
        XX_train_yuan_list.append(XX_train_yuan)
        XX_total_yuan_list.append(XX_total_yuan)
        YY_train_yuan_list.append(YY_train_yuan)
        YY_total_yuan_list.append(YY_total_yuan)
    for k in range(n_simulations):
        print(j * 10 + k)
        XX_train_yuan = XX_train_yuan_list[k]
        XX_total_yuan = XX_total_yuan_list[k]
        YY_train_yuan = YY_train_yuan_list[k]
        YY_total_yuan = YY_total_yuan_list[k]
        preprocessor = PREPROCESSOR(XX_train_yuan)
        XX_train_yuan = preprocessor.fit_transform(XX_train_yuan)
        XX_total_yuan = preprocessor.transform(XX_total_yuan)
        YY_train_yuan = YY_train_yuan.to_numpy()
        YY_total_yuan = YY_total_yuan.to_numpy()

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
        ##########################################################################TabPFN
        reg_default = TabPFNRegressor(device="cuda", random_state=random_seeds)
        start_time = time.perf_counter()
        reg_default.fit(xx_train, yy_train.ravel())
        train_time_default = time.perf_counter() - start_time
        qidong = reg_default.predict(xx_total)
        start_time = time.perf_counter()
        YY_test_prediction_initial = reg_default.predict(xx_total)
        pred_time_default = time.perf_counter() - start_time
        YY_test_prediction_initial = reg_default.predict(xx_total, output_type="full")
        Y_test_prediction_initial_default = YY_test_prediction_initial[f"mean"] * Sigma_scaler + Mu_scaler
        Variance_initial = YY_test_prediction_initial[f"variance"].reshape(-1, 1)
        Prediction_sigma_initial_default = np.sqrt(Variance_initial.ravel() * Sigma_scaler ** 2)
        rmse_default = root_mean_squared_error(YY_total_yuan, Y_test_prediction_initial_default)
        mae_default = mean_absolute_error(YY_total_yuan, Y_test_prediction_initial_default)
        r2_default = r2_score(YY_total_yuan, Y_test_prediction_initial_default)
        rmse_normalized_default = 1 - rmse_default / (np.max(YY_total_yuan) - np.min(YY_total_yuan))
        mae_normalized_default = 1 - mae_default / (np.max(YY_total_yuan) - np.min(YY_total_yuan))
        results_list = []
        results_list.append({'Model': 'TabPFN (Default)', 'RMSE': rmse_default, 'R2': r2_default, 'MAE': mae_default, 'RMSE_N': rmse_normalized_default, 'MAE_N': mae_normalized_default, 'Time_Train': train_time_default, 'Time_Pred': pred_time_default})
        results_prediction = []
        results_prediction.append({'Model': 'Actual', 'Performance': YY_total_yuan})
        results_prediction.append({'Model': 'TabPFN (Default)', 'Performance': Y_test_prediction_initial_default, 'Prediction_sigma_initial_default': Prediction_sigma_initial_default})
        ##########################################################################MFTabPFN
        print(f"Operating model: MFTabPFN")
        criterion = torch.nn.MSELoss()
        start_time = time.perf_counter()
        TabPFN_prediction_initial = reg_default.predict(xx_train).reshape(-1, 1)
        YY_train_prediction_initial = torch.tensor(TabPFN_prediction_initial, dtype=torch.float)
        loss_train = criterion(OUTPUT, YY_train_prediction_initial)
        if loss_train > 0.01:
            if nx < 100:
                model_RCNN = MLP_S(task_type, INPUT, OUTPUT, input_TabPFN, bili, n_layer, lr, weight_decay, epochs, hidden_channels, activate_function, batch, TabPFN_prediction_initial, random_seeds)
            else:
                model_RCNN = RCNN_S(task_type, INPUT, OUTPUT, input_TabPFN, bili, n_layer, lr, weight_decay, epochs, hidden_channels, activate_function, batch, TabPFN_prediction_initial, random_seeds)
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
            Y_test_prediction = (TabPFN_prediction_tensor.detach().cpu().numpy() + TabPFN_prediction_tensor_initial.detach().cpu().numpy()) * Sigma_scaler + Mu_scaler
            pred_time = time.perf_counter() - start_time
            TabPFN_prediction_initial_full = reg_default.predict(xx_total, output_type="full")
            Variance_initial = TabPFN_prediction_initial_full[f"variance"].reshape(-1, 1)
            Prediction_sigma_initial = np.sqrt(Variance_initial.ravel() * Sigma_scaler ** 2)
            Prediction_sigma = np.sqrt(Variance.detach().cpu().numpy().ravel() * Sigma_scaler ** 2)
            Prediction_total_sigma = np.sqrt(Prediction_sigma_initial ** 2 + Prediction_sigma ** 2)
            rmse = root_mean_squared_error(YY_total_yuan, Y_test_prediction)
            mae = mean_absolute_error(YY_total_yuan, Y_test_prediction)
            r2 = r2_score(YY_total_yuan, Y_test_prediction)
            rmse_normalized = 1 - rmse / (np.max(YY_total_yuan) - np.min(YY_total_yuan))
            mae_normalized = 1 - mae / (np.max(YY_total_yuan) - np.min(YY_total_yuan))
            results_prediction.append({'Model': 'MFTabPFN (Default)', 'Performance': Y_test_prediction, 'Prediction_total_sigma': Prediction_total_sigma, 'Prediction_sigma_initial': Prediction_sigma_initial, 'Prediction_sigma': Prediction_sigma})
            del reg_default, model_RCNN
            torch.cuda.empty_cache()
            gc.collect()
        else:
            train_time = time.perf_counter() - start_time + train_time_default
            pred_time = pred_time_default
            rmse = rmse_default
            mae = mae_default
            r2 = r2_default
            rmse_normalized = rmse_normalized_default
            mae_normalized = mae_normalized_default
            results_prediction.append({'Model': 'MFTabPFN (Default)', 'Performance': Y_test_prediction_initial_default, 'Prediction_sigma_initial_default': Prediction_sigma_initial_default})
        results_list.append({'Model': 'MFTabPFN (Default)', 'RMSE': rmse, 'R2': r2, 'MAE': mae, 'RMSE_N': rmse_normalized, 'MAE_N': mae_normalized, 'Time_Train': train_time, 'Time_Pred': pred_time})
        ##########################################################################AutoGluon
        TRAIN_Autogluon = pd.DataFrame(xx_train, columns=[f'feature_{i}' for i in range(nx)])
        TRAIN_Autogluon['target'] = yy_train
        TEST_Autogluon = pd.DataFrame(xx_total, columns=[f'feature_{i}' for i in range(nx)])
        start_time = time.perf_counter()
        predictor = TabularPredictor(
            label="target",
            problem_type="regression",
        )
        predictor.fit(train_data=TRAIN_Autogluon, time_limit=3600, presets='best_quality')
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
            {'Model': 'AutoGluon', 'RMSE': rmse, 'R2': r2, 'MAE': mae, 'RMSE_N': rmse_normalized,
             'MAE_N': mae_normalized,
             'Time_Train': train_time, 'Time_Pred': pred_time})
        results_prediction.append({'Model': 'AutoGluon', 'Performance': Y_test_prediction_initial})
        YY_index[j][k] = pd.DataFrame(results_list)
        YY_real_prediction[j][k] = pd.DataFrame(results_prediction)
        with open('YY_index.pkl', 'wb') as f:
            pickle.dump(YY_index, f)
        with open('YY_real_prediction.pkl', 'wb') as f:
            pickle.dump(YY_real_prediction, f)
        del predictor
        del INPUT, OUTPUT
        del preprocessor, scaler_Y
        gc.collect()
        torch.cuda.empty_cache()

