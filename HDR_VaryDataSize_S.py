from tabpfn import TabPFNRegressor
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)
from RCNN_MFTabPFN_S import RCNN_S
from ANN_model import SingleANN_PM
from pathlib import Path
from TabPFN_model import TabPFN_model_main
from autogluon.tabular import TabularPredictor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
import pickle
import torch
if not torch.cuda.is_available():
    raise SystemError('GPU device not found. For fast training, please enable GPU.')

SAVE_DIR = './Datasets/Synthetic/HDR_varying_data_size'

n_train_number = 5
n_simulations = 10
Results_simulation_train_list = [[None for _ in range(n_train_number)] for _ in range(n_simulations)]
for j in range(0, n_simulations):
    print(j)
    Case = j
    nx = (Case + 1) * 100
    for k in range(0, n_train_number):
        N_train = (k + 1) * nx
        N = 1000

        # pkl_file = os.path.join(SAVE_DIR, f'XX_train_yuan_{(Case+1)*100}_{N_train}.pkl')
        # with open(pkl_file, 'wb') as f:
        #     pickle.dump(XX_train_yuan, f)
        # pkl_file = os.path.join(SAVE_DIR, f'XX_test_yuan_{(Case+1)*100}_{N_train}.pkl')
        # with open(pkl_file, 'wb') as f:
        #     pickle.dump(XX_total_yuan, f)
        # pkl_file = os.path.join(SAVE_DIR, f'YY_train_yuan_{(Case+1)*100}_{N_train}.pkl')
        # with open(pkl_file, 'wb') as f:
        #     pickle.dump(YY_train_yuan, f)
        # pkl_file = os.path.join(SAVE_DIR, f'YY_test_yuan_{(Case+1)*100}_{N_train}.pkl')
        # with open(pkl_file, 'wb') as f:
        #     pickle.dump(YY_total_yuan, f)

        pkl_file = os.path.join(SAVE_DIR, f'XX_train_yuan_{(Case + 1) * 100}_{N_train}.pkl')
        with open(pkl_file, 'rb') as f:
            XX_train_yuan = pickle.load(f)
        pkl_file = os.path.join(SAVE_DIR, f'XX_test_yuan_{(Case + 1) * 100}_{N_train}.pkl')
        with open(pkl_file, 'rb') as f:
            XX_total_yuan = pickle.load(f)
        pkl_file = os.path.join(SAVE_DIR, f'YY_train_yuan_{(Case + 1) * 100}_{N_train}.pkl')
        with open(pkl_file, 'rb') as f:
            YY_train_yuan = pickle.load(f)
        pkl_file = os.path.join(SAVE_DIR, f'YY_test_yuan_{(Case + 1) * 100}_{N_train}.pkl')
        with open(pkl_file, 'rb') as f:
            YY_total_yuan = pickle.load(f)

        scaler = StandardScaler()
        xx_train = scaler.fit_transform(XX_train_yuan)
        xx_total = scaler.transform(XX_total_yuan)
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
        task_type = "regression"
        batch = 256

        ##########################################################################TabPFN
        if nx > 500:
            reg = TabPFNRegressor(n_estimators=8, random_state=42, ignore_pretraining_limits=True)
        else:
            reg = TabPFNRegressor(n_estimators=8, random_state=42)
        reg.fit(xx_train, yy_train.ravel())
        YY_test_prediction_initial = reg.predict(xx_total, output_type="full")
        Y_test_prediction_initial = YY_test_prediction_initial[f"mean"] * Sigma_scaler + Mu_scaler
        rmse = 1 - root_mean_squared_error(YY_total_yuan, Y_test_prediction_initial) / (np.max(YY_total_yuan) - np.min(YY_total_yuan))
        mae = 1 - mean_absolute_error(YY_total_yuan, Y_test_prediction_initial) / (np.max(YY_total_yuan) - np.min(YY_total_yuan))
        r2 = r2_score(YY_total_yuan, Y_test_prediction_initial)
        results_list = []
        results_list.append({'Model': 'TabPFN', 'RMSE': rmse, 'R2': r2, 'MAE': mae})
        ##########################################################################ML
        models = {
            'SVR': SVR(),
            'RandomForest': RandomForestRegressor(random_state=42),
            'ExtraTrees': ExtraTreesRegressor(random_state=42),
            'KNN': KNeighborsRegressor(),
            'Ridge': Ridge(),
            'LightGBM': LGBMRegressor(verbose=-1),
            'CatBoost': CatBoostRegressor(verbose=0),
            'XGBoost': XGBRegressor()
        }
        for name, model in models.items():
            model.fit(xx_train, yy_train.ravel())
            Y_test_prediction_initial = model.predict(xx_total) * Sigma_scaler + Mu_scaler
            rmse = 1 - root_mean_squared_error(YY_total_yuan, Y_test_prediction_initial) / (np.max(YY_total_yuan) - np.min(YY_total_yuan))
            mae = 1 - mean_absolute_error(YY_total_yuan, Y_test_prediction_initial) / (np.max(YY_total_yuan) - np.min(YY_total_yuan))
            r2 = r2_score(YY_total_yuan, Y_test_prediction_initial)
            results_list.append({'Model': name, 'RMSE': rmse, 'R2': r2, 'MAE': mae})
        ##########################################################################AutoGluon
        TRAIN_Autogluon = pd.DataFrame(xx_train, columns=[f'feature_{i}' for i in range(nx)])
        TRAIN_Autogluon['target'] = yy_train
        predictor = TabularPredictor(
            label="target",
            problem_type="regression",
            path="autogluon_model"
        )
        predictor.fit(train_data=TRAIN_Autogluon, time_limit=600)
        TEST_Autogluon = pd.DataFrame(xx_total, columns=[f'feature_{i}' for i in range(nx)])
        Y_test_prediction_initial = predictor.predict(TEST_Autogluon).to_numpy() * Sigma_scaler + Mu_scaler
        rmse = 1 - root_mean_squared_error(YY_total_yuan, Y_test_prediction_initial) / (np.max(YY_total_yuan) - np.min(YY_total_yuan))
        mae = 1 - mean_absolute_error(YY_total_yuan, Y_test_prediction_initial) / (np.max(YY_total_yuan) - np.min(YY_total_yuan))
        r2 = r2_score(YY_total_yuan, Y_test_prediction_initial)
        results_list.append({'Model': 'AutoGluon', 'RMSE': rmse, 'R2': r2, 'MAE': mae})
        ##########################################################################MFTabPFN
        model_RCNN = RCNN_S(task_type, INPUT, OUTPUT, input_TabPFN, bili, n_layer, lr, weight_decay, epochs, hidden_channels, activate_function, batch)
        model_RCNN.eval()
        YY_train_middle = model_RCNN(torch.tensor(xx_train, dtype=torch.float))
        YY_test_middle = model_RCNN(torch.tensor(xx_total, dtype=torch.float))

        if nx > 500:
            reg_initial = TabPFNRegressor(n_estimators=8, random_state=42, ignore_pretraining_limits=True)
        else:
            reg_initial = TabPFNRegressor(n_estimators=8, random_state=42)
        reg_initial.fit(INPUT.detach().numpy(), OUTPUT.detach().numpy().ravel())
        TabPFN_prediction_initial_full = reg_initial.predict(xx_total, output_type="full")
        TabPFN_prediction_initial = TabPFN_prediction_initial_full[f"mean"].reshape(-1, 1)
        TabPFN_prediction_tensor_initial = torch.tensor(TabPFN_prediction_initial, dtype=torch.float)
        if nx > 500:
            reg_initial0 = TabPFNRegressor(n_estimators=8, random_state=42, ignore_pretraining_limits=True)
        else:
            reg_initial0 = TabPFNRegressor(n_estimators=8, random_state=42)
        reg_initial0.fit(INPUT.detach().numpy(), OUTPUT.detach().numpy().ravel())
        TabPFN_prediction_initial0 = reg_initial0.predict(INPUT.detach().numpy()).reshape(-1, 1)
        TabPFN_prediction_tensor_initial0 = torch.tensor(TabPFN_prediction_initial0, dtype=torch.float)

        save_path_to_fine_tuned_model = (Path(__file__).parent / f"tabpfn_model_{task_type}.ckpt")
        TabPFN_prediction_tensor, Lower, Upper, Variance = TabPFN_model_main(
            path_to_base_model="auto",
            save_path_to_fine_tuned_model=save_path_to_fine_tuned_model,
            X_train=YY_train_middle,
            y_train=OUTPUT - TabPFN_prediction_tensor_initial0.to('cpu'),
            X_test=YY_test_middle,
            n_classes=None,
            categorical_features_index=None,
            task_type=task_type,
            # device="cuda" if torch.cuda.is_available() else "cpu",
            device="cuda:0" if torch.cuda.is_available() else "cpu",
        )
        Y_test_prediction = (TabPFN_prediction_tensor.detach().cpu().numpy() + TabPFN_prediction_tensor_initial.detach().cpu().numpy()) * Sigma_scaler + Mu_scaler
        rmse = 1 - root_mean_squared_error(YY_total_yuan, Y_test_prediction) / (np.max(YY_total_yuan) - np.min(YY_total_yuan))
        mae = 1 - mean_absolute_error(YY_total_yuan, Y_test_prediction) / (np.max(YY_total_yuan) - np.min(YY_total_yuan))
        r2 = r2_score(YY_total_yuan, Y_test_prediction)
        results_list.append({'Model': 'MFTabPFN', 'RMSE': rmse, 'R2': r2, 'MAE': mae})
        ##########################################################################MLP
        epochs_ann = 500
        model_SingleANN = SingleANN_PM(INPUT, OUTPUT, bili, n_layer, lr, weight_decay, epochs_ann, hidden_channels, activate_function, batch)
        model_SingleANN.eval()
        ANN_prediction_tensor = model_SingleANN(torch.tensor(xx_total, dtype=torch.float))
        ANN_prediction = ANN_prediction_tensor.detach().cpu().numpy() * Sigma_scaler + Mu_scaler
        rmse = 1 - root_mean_squared_error(YY_total_yuan, ANN_prediction) / (np.max(YY_total_yuan) - np.min(YY_total_yuan))
        mae = 1 - mean_absolute_error(YY_total_yuan, ANN_prediction) / (np.max(YY_total_yuan) - np.min(YY_total_yuan))
        r2 = r2_score(YY_total_yuan, ANN_prediction)
        results_list.append({'Model': 'ANN', 'RMSE': rmse, 'R2': r2, 'MAE': mae})
        Results_simulation_train_list[j][k] = pd.DataFrame(results_list)
        # with open('Results_simulation_train_list.pkl', 'wb') as f:
        #     pickle.dump(Results_simulation_train_list, f)
        # print("Results_simulation_train_list is saved to 'Results_simulation_train_list.pkl'")

