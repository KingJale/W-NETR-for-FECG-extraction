import mne
import matplotlib.pyplot as plt
import numpy as np
import wfdb

file_name = 'a03'
fecg, fields = wfdb.rdsamp(file_name)#作用：读取WFDB（Waveform Database）格式的生理信号数据文件//参数：file_name- 数据文件的名称（不包含扩展名）//返回值：fecg：一个二维numpy数组，包含实际的信号数据//fields：一个字典，包含数据的元信息（采样频率、单位、信号名称等）
print(fecg.shape)
fig, axs = plt.subplots(4)
axs[0].plot(fecg[0:4000,0])
axs[1].plot(fecg[0:4000,1])
axs[2].plot(fecg[0:4000,2])
axs[3].plot(fecg[0:4000,3])
plt.show()

"""索引说明：
fecg[0:4000, 0]：第一个0:4000表示时间索引（从0到3999），第二个0表示通道索引
使用切片[0:4000]只绘制前4000个数据点，方便查看细节
四个通道：
fecg[0:4000, 0]：通道1（索引0）
fecg[0:4000, 1]：通道2（索引1）
fecg[0:4000, 2]：通道3（索引2）
fecg[0:4000, 3]：通道4（索引3）"""
# file_name = 'ecgca244.edf'
# data = mne.io.read_raw_edf(file_name)
# raw_data = data.get_data()
# raw_data.shape
# fig, axs = plt.subplots(5)
# axs[0].plot(raw_data[0,0:4000])
# axs[1].plot(raw_data[1,0:4000])
# axs[2].plot(raw_data[2,0:4000])
# axs[3].plot(raw_data[3,0:4000])
# axs[4].plot(raw_data[4,0:4000])
# plt.show()

# axs[5].plot(raw_data[5,0:4000])