import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from MFTabPFN_model3 import MFTabPFN_SY3
import torch
if not torch.cuda.is_available():
    raise SystemError('GPU device not found. For fast training, please enable GPU.')


def f_low(x):
    ff = np.exp(x[:, 0]) * np.sin(x[:, 0] * 10)
    return ff
def f_high(x):
    ff = f_low(x) + 5 * x[:, 0] ** 2
    return ff
def f_mid(x):
    ff = (f_low(x) + f_high(x)) / 2
    return ff

random_seeds = 111
np.random.seed(random_seeds)

N_low = 40
N_mid = 10
N_high = 5
N_test = 200

XX_train_low = np.linspace(0, 1, N_low).reshape(-1, 1)
XX_train_mid = np.linspace(0, 1, N_mid).reshape(-1, 1)
XX_train_high = np.linspace(0, 1, N_high).reshape(-1, 1)
XX_total = np.linspace(0, 1, N_test).reshape(-1, 1)
XX_train_yuan_low = XX_train_low
XX_train_yuan_mid = XX_train_mid
XX_train_yuan_high = XX_train_high
XX_total_yuan = XX_total
YY_train_yuan_low = f_low(XX_train_yuan_low)
YY_train_yuan_mid = f_mid(XX_train_yuan_mid)
YY_train_yuan_high = f_high(XX_train_yuan_high)
YY_total_yuan_low = f_low(XX_total_yuan)
YY_total_yuan_mid = f_mid(XX_total_yuan)
YY_total_yuan_high = f_high(XX_total_yuan)

XX_train_yuan = np.concatenate((XX_train_yuan_low, XX_train_yuan_mid, XX_train_yuan_high), axis=0)
YY_train_yuan = np.concatenate((YY_train_yuan_low, YY_train_yuan_mid, YY_train_yuan_high), axis=0)

Prediction_MFTabPFN = MFTabPFN_SY3(XX_train_yuan, YY_train_yuan, XX_total_yuan, random_seeds, N_low, N_mid, N_high)
Y_prediction_MFTabPFN = Prediction_MFTabPFN[0] #Prediction mean of MFTabPFN
Y_prediction_MF = Prediction_MFTabPFN[1] #Prediction mean of mid-fidelity model
Y_prediction_LF = Prediction_MFTabPFN[2] #Prediction mean of low-fidelity model


plt.figure(1)
plt.plot(XX_total_yuan, YY_total_yuan_low, linestyle='--', color='g', label='Reference low-fidelity solution', zorder=1)
plt.plot(XX_total_yuan, Y_prediction_LF, linestyle='-', color='g', label='Predicted low-fidelity solution', zorder=1)
plt.plot(XX_total_yuan, YY_total_yuan_mid, linestyle='--', color='y', label='Reference mid-fidelity solution', zorder=2)
plt.plot(XX_total_yuan, Y_prediction_MF, linestyle='-', color='y', label='Predicted mid-fidelity solution', zorder=2)
plt.plot(XX_total_yuan, YY_total_yuan_high, linestyle='--', color='b', label='Reference high-fidelity solution', zorder=3)
plt.plot(XX_total_yuan, Y_prediction_MFTabPFN, linestyle='-', color='b', label='Predicted high-fidelity solution', zorder=3)
plt.scatter(XX_train_yuan_high, YY_train_yuan_high, color='b', marker='^', s=35, label="High-fidelity training points", zorder=3)
plt.scatter(XX_train_yuan_mid, YY_train_yuan_mid, color='y', marker='s', s=35, label="Mid-fidelity training points", zorder=2)
plt.scatter(XX_train_yuan_low, YY_train_yuan_low, color='g', marker='o', s=35, label="Low-fidelity training points", zorder=1)
plt.xlabel('Input', fontsize=12)
plt.ylabel('Output', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=10)
plt.tight_layout()
# for ext in ['png', 'tiff', 'pdf']:
#     output_path = os.path.join(f'Toy_M3_plot.{ext}')
#     plt.savefig(output_path, dpi=400, bbox_inches='tight', format=ext)
plt.show(block=True)


