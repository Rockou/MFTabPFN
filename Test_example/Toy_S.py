import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from MFTabPFN_model import MFTabPFN_SY
import torch
if not torch.cuda.is_available():
    raise SystemError('GPU device not found. For fast training, please enable GPU.')


def toy(x):
    y = x[:, 0] * np.sin(x[:, 0] * 5) * np.cos(x[:, 0]) + x[:, 0]
    return y

random_seeds = 111
np.random.seed(random_seeds)

N_train = 80
N_test = 500
XX_train_yuan = np.linspace(0, 20, N_train).reshape(-1, 1)
XX_total_yuan = np.linspace(0, 20, N_test).reshape(-1, 1)
YY_train_yuan = toy(XX_train_yuan).reshape(-1, 1)
YY_total_yuan = toy(XX_total_yuan).reshape(-1, 1)

Prediction_MFTabPFN = MFTabPFN_SY(XX_train_yuan, YY_train_yuan, XX_total_yuan, random_seeds, N_train)
Y_prediction_MFTabPFN = Prediction_MFTabPFN[0] #Prediction mean of MFTabPFN
Y_prediction_SD_MFTabPFN = Prediction_MFTabPFN[1] #Prediction standard deviation of MFTabPFN
Y_prediction_TabPFN = Prediction_MFTabPFN[2] #Prediction mean of TabPFN
Y_prediction_SD_TabPFN = Prediction_MFTabPFN[3] #Prediction standard deviation of TabPFN

plt.figure(1)
plt.plot(XX_total_yuan, Y_prediction_MFTabPFN, linestyle='-', color='b', linewidth=2, label='MFTabPFN')
plt.plot(XX_total_yuan, Y_prediction_TabPFN, linestyle='--', color='g', linewidth=2, label='TabPFN')
plt.fill_between(XX_total_yuan.ravel(), Y_prediction_MFTabPFN - Y_prediction_SD_MFTabPFN, Y_prediction_MFTabPFN + Y_prediction_SD_MFTabPFN, color='b', edgecolor='none', alpha=0.3)
plt.fill_between(XX_total_yuan.ravel(), Y_prediction_TabPFN - Y_prediction_SD_TabPFN, Y_prediction_TabPFN + Y_prediction_SD_TabPFN, color='g', edgecolor='none', alpha=0.3)
plt.plot(XX_total_yuan, YY_total_yuan, linestyle=':', color='k', linewidth=2, label='Reference')
plt.scatter(XX_train_yuan, YY_train_yuan, color='r', s=35, alpha=0.6, label='Training points')
plt.xlabel('Input', fontsize=12)
plt.ylabel('Output', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=10)
plt.tight_layout()
# for ext in ['png', 'tiff', 'pdf']:
#     output_path = os.path.join(f'Toy_S_plot.{ext}')
#     plt.savefig(output_path, dpi=400, bbox_inches='tight', format=ext)
plt.show(block=True)

