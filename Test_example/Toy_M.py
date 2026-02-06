import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from MFTabPFN_model import MFTabPFN_SY
import torch
if not torch.cuda.is_available():
    raise SystemError('GPU device not found. For fast training, please enable GPU.')


def f_low(x):
    ff = x[:, 0] ** 2 * np.cos(x[:, 0] * 20)
    return ff
def f_high(x):
    ff = f_low(x) + x[:, 0] ** 3
    return ff

random_seeds = 111
np.random.seed(random_seeds)

N_low = 40
N_high = 20
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

Prediction_MFTabPFN = MFTabPFN_SY(XX_train_yuan, YY_train_yuan, XX_total_yuan, random_seeds, N_low, N_high)
Y_prediction_MFTabPFN = Prediction_MFTabPFN[0] #Prediction mean of MFTabPFN
Y_prediction_SD_MFTabPFN = Prediction_MFTabPFN[1] #Prediction standard deviation of MFTabPFN
Y_prediction_LF = Prediction_MFTabPFN[2] #Prediction mean of low-fidelity model
Y_prediction_SD_LF = Prediction_MFTabPFN[3] #Prediction standard deviation of low-fidelity model

plt.figure(1)
plt.plot(XX_total_yuan, YY_total_yuan_low, linestyle='--', color='g', label='Reference low-fidelity solution', zorder=1)
plt.plot(XX_total_yuan, Y_prediction_LF, linestyle='-', color='g', label='Predicted low-fidelity solution', zorder=1)
plt.fill_between(XX_total_yuan.ravel(), Y_prediction_LF - Y_prediction_SD_LF, Y_prediction_LF + Y_prediction_SD_LF, color='g', edgecolor='none', alpha=0.4, zorder=1)
plt.plot(XX_total_yuan, YY_total_yuan_high, linestyle='--', color='b', label='Reference high-fidelity solution', zorder=2)
plt.plot(XX_total_yuan, Y_prediction_MFTabPFN, linestyle='-', color='b', label='Predicted high-fidelity solution', zorder=2)
plt.fill_between(XX_total_yuan.ravel(), Y_prediction_MFTabPFN - Y_prediction_SD_MFTabPFN, Y_prediction_MFTabPFN + Y_prediction_SD_MFTabPFN, color='b', edgecolor='none', alpha=0.4, zorder=2)
plt.scatter(XX_train_yuan_high, YY_train_yuan_high, color='b', marker='^', s=35, label="High-fidelity training points", zorder=2)
plt.scatter(XX_train_yuan_low, YY_train_yuan_low, color='g', marker='o', s=35, label="Low-fidelity training points", zorder=1)
plt.xlabel('Input', fontsize=12)
plt.ylabel('Output', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=10)
plt.tight_layout()
# for ext in ['png', 'tiff', 'pdf']:
#     output_path = os.path.join(f'Toy_M_plot.{ext}')
#     plt.savefig(output_path, dpi=400, bbox_inches='tight', format=ext)
plt.show(block=True)


