import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from MFTabPFN_model import MFTabPFN_SY
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


def f_low(x):
    ff = x[:, 0] * np.sin(x[:, 0]) + np.cos(x[:, 0] * 20)
    return ff
def f_high(x):
    ff = f_low(x) + x[:, 0] ** 2 - 1
    return ff

random_seeds = 111
np.random.seed(random_seeds)

N_low = 20
N_high = 5
N_low0 = N_low
N_high0 = N_high
N_test = 200

XX_train_low = np.linspace(0, 1, N_low).reshape(-1, 1)
XX_train_high = np.linspace(0, 1, N_high).reshape(-1, 1)
XX_total = np.linspace(0, 1, N_test).reshape(-1, 1)
XX_train_yuan_low = XX_train_low
XX_train_yuan_high = XX_train_high
XX_total_yuan = XX_total
YY_train_yuan_low = f_low(XX_train_yuan_low)
YY_train_yuan_high = f_high(XX_train_yuan_high)
YY_total_yuan_low = f_low(XX_total_yuan)
YY_total_yuan_high = f_high(XX_total_yuan)

XX_train_yuan = np.concatenate((XX_train_yuan_low, XX_train_yuan_high), axis=0)
YY_train_yuan = np.concatenate((YY_train_yuan_low, YY_train_yuan_high), axis=0)

xx_train1 = pd.DataFrame(XX_train_yuan)
xx_total1 = pd.DataFrame(XX_total_yuan)
preprocessor = PREPROCESSOR(xx_train1)
xx_train1 = preprocessor.fit_transform(xx_train1)
xx_total1 = preprocessor.transform(xx_total1)
xx_train = xx_train1
xx_total = xx_total1

U = 20
Cost_low = 1
Cost_high = 1
Stop = np.zeros((U, 1), dtype=float)
Stop0 = 0.01

train = XX_train_low.ravel()
total = XX_total.ravel()
matches = np.any(np.isclose(train[:, None], total), axis=0)
common_indices_in_total = np.where(matches)[0]
Index_low_list = np.zeros(N_test, dtype=int)
m_low = len(common_indices_in_total)
Index_low_list[:m_low] = common_indices_in_total
train = XX_train_high.ravel()
total = XX_total.ravel()
matches = np.any(np.isclose(train[:, None], total), axis=0)
common_indices_in_total = np.where(matches)[0]
Index_high_list = np.zeros(N_test, dtype=int)
m_high = len(common_indices_in_total)
Index_high_list[:m_high] = common_indices_in_total

for i in range(U):
    print(i)
    if i > 0:
        Y_test_prediction_iteration = Y_test_prediction.copy()

    Prediction_MFTabPFN = MFTabPFN_SY(XX_train_yuan, YY_train_yuan, XX_total_yuan, random_seeds, N_low, N_high)
    Y_test_prediction = Prediction_MFTabPFN[0]  # Prediction mean of MFTabPFN
    Prediction_total_sigma = Prediction_MFTabPFN[1]  # Prediction standard deviation of MFTabPFN
    Y_prediction_LF = Prediction_MFTabPFN[2]  # Prediction mean of low-fidelity model
    Prediction_sigma_initial = Prediction_MFTabPFN[3]  # Prediction standard deviation of low-fidelity model
    model_encoder = Prediction_MFTabPFN[4]
    Prediction_sigma = np.sqrt(Prediction_total_sigma ** 2 - model_encoder.afa.detach().cpu().numpy() ** 2 * Prediction_sigma_initial ** 2)

    if i > 0:
        Stop[i] = np.mean(np.abs(Y_test_prediction - Y_test_prediction_iteration))/(np.max(Y_test_prediction) - np.min(Y_test_prediction))
        print("Convergence Criterion:", Stop[i][0])
        if Stop[i] <= Stop0:
            break

    Prediction_sigma_initial_mean = np.mean(model_encoder.afa.detach().cpu().numpy() * Prediction_sigma_initial)
    Prediction_sigma_mean = np.mean(Prediction_sigma)
    distances_low = np.sum(np.abs(xx_total[:, np.newaxis] - xx_train[:N_low, :]), axis=2)
    min_distances_low = np.min(distances_low, axis=1)
    distances_high = np.sum(np.abs(xx_total[:, np.newaxis] - xx_train[N_low:, :]), axis=2)
    min_distances_high = np.min(distances_high, axis=1)

    if i < 5:
        Prediction_sigma_stack = np.stack((min_distances_low * model_encoder.afa.detach().cpu().numpy() * Prediction_sigma_initial * Prediction_sigma_initial_mean / Cost_low, min_distances_high * Prediction_sigma * Prediction_sigma_mean / Cost_high), axis=1)
    else:
        Prediction_sigma_stack = np.stack((model_encoder.afa.detach().cpu().numpy() * Prediction_sigma_initial * Prediction_sigma_initial_mean / Cost_low, Prediction_sigma * Prediction_sigma_mean / Cost_high),axis=1)

    Prediction_sigma_stack[Index_low_list[:m_low], 0] = 0
    Prediction_sigma_stack[Index_high_list[:m_high], 1] = 0

    max_value = np.max(Prediction_sigma_stack)
    max_indices = np.where(Prediction_sigma_stack == max_value)
    index_fidelity = max_indices[1][0]
    index_max = max_indices[0][0]

    if index_fidelity == 0:
        Index_low_list[m_low] = index_max
        m_low = m_low + 1
    elif index_fidelity == 1:
        Index_high_list[m_high] = index_max
        m_high = m_high + 1

    XX_new = XX_total_yuan[index_max]
    if index_fidelity == 0:
        XX_train_yuan_low = np.concatenate((XX_train_yuan_low, XX_new.reshape(1, -1)), axis=0)
        YY_new = YY_total_yuan_low[index_max]
        YY_train_yuan_low = np.concatenate((YY_train_yuan_low, YY_new.ravel()), axis=0)
        N_low = N_low + 1
    elif index_fidelity == 1:
        XX_train_yuan_high = np.concatenate((XX_train_yuan_high, XX_new.reshape(1, -1)), axis=0)
        YY_new = YY_total_yuan_high[index_max]
        YY_train_yuan_high = np.concatenate((YY_train_yuan_high, YY_new.ravel()), axis=0)
        N_high = N_high + 1

    XX_train_yuan = np.concatenate((XX_train_yuan_low, XX_train_yuan_high), axis=0)
    YY_train_yuan = np.concatenate((YY_train_yuan_low, YY_train_yuan_high), axis=0)

    xx_train1 = pd.DataFrame(XX_train_yuan)
    xx_total1 = pd.DataFrame(XX_total_yuan)
    preprocessor = PREPROCESSOR(xx_train1)
    xx_train1 = preprocessor.fit_transform(xx_train1)
    xx_total1 = preprocessor.transform(xx_total1)
    xx_train = xx_train1
    xx_total = xx_total1


plt.figure(1)
plt.plot(XX_total_yuan, YY_total_yuan_low, linestyle='--', color='g', label='Reference low-fidelity solution', zorder=1)
plt.plot(XX_total_yuan, Y_prediction_LF, linestyle='-', color='g', label='Predicted low-fidelity solution', zorder=1)
plt.fill_between(XX_total_yuan.ravel(), Y_prediction_LF - Prediction_sigma_initial, Y_prediction_LF + Prediction_sigma_initial, color='g', edgecolor='none', alpha=0.4, zorder=1)
plt.plot(XX_total_yuan, YY_total_yuan_high, linestyle='--', color='b', label='Reference high-fidelity solution', zorder=2)
plt.plot(XX_total_yuan, Y_test_prediction, linestyle='-', color='b', label='Predicted high-fidelity solution', zorder=2)
plt.fill_between(XX_total_yuan.ravel(), Y_test_prediction - Prediction_total_sigma, Y_test_prediction + Prediction_total_sigma, color='b', edgecolor='none', alpha=0.4, zorder=2)
plt.scatter(XX_train_yuan_high[:N_high0], YY_train_yuan_high[:N_high0], color='b', marker='^', s=35, label="Initial high-fidelity training points", zorder=2)
plt.scatter(XX_train_yuan_low[:N_low0], YY_train_yuan_low[:N_low0], color='g', marker='o', s=35, label="Initial low-fidelity training points", zorder=1)
plt.scatter(XX_train_yuan_high[N_high0:], YY_train_yuan_high[N_high0:], color='r', marker='^', s=35, label="Added high-fidelity training points", zorder=2)
plt.scatter(XX_train_yuan_low[N_low0:], YY_train_yuan_low[N_low0:], color='r', marker='o', s=35, label="Added low-fidelity training points", zorder=1)
plt.xlabel('Input', fontsize=12)
plt.ylabel('Output', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=10)
plt.tight_layout()
# for ext in ['png', 'tiff', 'pdf']:
#     output_path = os.path.join(f'Toy_Active_plot.{ext}')
#     plt.savefig(output_path, dpi=400, bbox_inches='tight', format=ext)
plt.show(block=True)

