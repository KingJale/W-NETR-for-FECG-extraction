from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import scipy
from scipy import signal
from scipy.signal import butter, lfilter, filtfilt, find_peaks, savgol_filter
import numpy as np
import os

import wfdb
import scipy.io


sub = ['a03']
channel = ['I', 'II', 'III', 'IV']
fs = 250
units = ['mV']
sig_name =  ['I']
"""scipy.signal.decimate()：降采样函数
参数4：降采样因子，将采样率降低到原来的1/4
原始数据可能采样率是1000Hz，降采样后变为250Hz
降采样会先进行抗混叠滤波，然后抽取每第4个样本"""
maECG, fields = wfdb.rdsamp(sub[0])
maECG_I = scipy.signal.decimate(maECG[:,0],4)
maECG_II = scipy.signal.decimate(maECG[:,1],4)
maECG_III = scipy.signal.decimate(maECG[:,2],4)
maECG_IV = scipy.signal.decimate(maECG[:,3],4)

print(maECG.shape)
for kh in range(0,15):
    wfdb.wrsamp(sub[0]+'_'+'I'+'_'+str(kh), fs = fs, units=units, sig_name=sig_name, p_signal=np.expand_dims(maECG_I[1000*kh:1000*(kh+1)], axis=1), write_dir = 'maECG')
    wfdb.wrsamp(sub[0]+'_'+'II'+'_'+str(kh), fs = fs, units=units, sig_name=sig_name, p_signal=np.expand_dims(maECG_II[1000*kh:1000*(kh+1)], axis=1), write_dir = 'maECG')
    wfdb.wrsamp(sub[0]+'_'+'III'+'_'+str(kh), fs = fs, units=units, sig_name=sig_name, p_signal=np.expand_dims(maECG_III[1000*kh:1000*(kh+1)], axis=1), write_dir = 'maECG')
    wfdb.wrsamp(sub[0]+'_'+'IV'+'_'+str(kh), fs = fs, units=units, sig_name=sig_name, p_signal=np.expand_dims(maECG_IV[1000*kh:1000*(kh+1)], axis=1), write_dir = 'maECG')
    
"""wrsamp()：将数据保存为WFDB格式
参数说明：
filename：生成的文件名，如a03_I_0
fs：采样频率（250Hz）
units：信号单位（mV）
sig_name：信号名称（固定为'I'）
p_signal：实际信号数据
write_dir：保存目录

maECG/
├── a03_I_0.dat      # 导联I，片段0
├── a03_I_0.hea      # 对应的头文件
├── a03_I_1.dat
├── a03_I_1.hea
├── ...
├── a03_II_0.dat     # 导联II，片段0
├── a03_II_0.hea
├── ...
├── a03_IV_14.dat    # 导联IV，片段14
└── a03_IV_14.hea

"""

"""假设原始数据：
采样率：1000Hz
持续时间：60秒
样本数：60,000
降采样后：
采样率：250Hz
样本数：15,000
每个片段：4秒 × 250Hz = 1000样本
可分割片段数：15,000 ÷ 1000 = 15个片段"""