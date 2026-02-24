# -*- coding: utf-8 -*-
"""
    这个文件处理数据集，获取mECG，fECG和混合后的ECG(mixture)

    这个文件里面有滤波器设计的函数，标准化的部分，数据加载的部分，
    以及论文里面提到的所有预处理步骤把算是，也就是说这整个文件完成了数据的预处理工作，包括用的滑动窗口进行分割
    重采样的部分不在这个file里面，应该在数据集目录下的脚本
"""
# import libraries
import numpy as np #用于数值计算和数组操作
import os#用于文件路径操作
import torch.utils.data as data#pytorch的数据加载工具
from PIL import Image#虽然导入但未使用可能是用于图像处理

#提供数据预处理工具，如归一化和数据集划分
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
#提供信号处理工具，如滤波器设计
from scipy import signal
from scipy.signal import butter, lfilter, filtfilt, find_peaks, savgol_filter
#用于读取WFDB格式的生理信号数据(如ECG)
import wfdb


def butter_bandpass(lowcut, highcut, fs, order=5):#低截止频率、高截止频率，采样频率、滤波器阶数
    """这个函数用于设计巴特沃斯带通滤波器。它接受低截止频率、高截止频率、采样频率和滤波器阶数作为输入，返回滤波器的系数b和a，用于后续滤波操作。
具体步骤如下：
计算奈奎斯特频率，即采样频率的一半，这是数字滤波器设计的最高有效频率。
将截止频率归一化，除以奈奎斯特频率，将实际频率转换为数字频率。
调用butter函数设计带通滤波器，指定阶数、归一化频率范围和滤波器类型。
返回滤波器的分子系数b和分母系数a。
返回的b和a系数可以用于scipy.signal.lfilter等函数进行信号滤波。例如，在脑电信号处理中，可以用它提取特定频段如alpha波（8-13 Hz）的成分。"""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a#返回的是滤波器的两个参数

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

#计算amp的均方根值（RMS）
def cal_rms(amp):
    return np.sqrt(np.mean(np.square(amp), axis=-1))

"""# Load Dataset"""

class Dataset(data.Dataset):#获取aECG,mECG和fECG生成训练集

    def __init__(self, root='./', load_set='train', transform=None):
        self.root = root   #os.path.expanduser(root)
        self.transform = transform
        self.load_set = load_set  # 'train','val','test'
        

        #这里的npy文件是由每个数据集目录下面的代码生成，可以把npy作为经过python格式化过的数据集(数据包)
        self.fecg_paths = np.load(os.path.join(root, 'fecg_paths_%s.npy'%self.load_set))
        self.mecg_paths = np.load(os.path.join(root, 'mecg_paths_%s.npy'%self.load_set))
        self.mixture_paths = np.load(os.path.join(root, 'mixture_paths_%s.npy'%self.load_set))
        #代码执行过程：
        # 1.字符串格式化'fecg_paths_%s.npy'%self.load_set → 'fecg_paths_train.npy'
        # 2.路径拼接 os.path.join("./data", "fecg_paths_train.npy") → "./data/fecg_paths_train.npy"
        # 3.数据加载 np.load("./data/fecg_paths_train.npy") → 返回存储在文件中的 NumPy 数组
        # 4.属性赋值 self.fecg_paths = 加载的数组


        # self.points2d = np.load(os.path.join(root, 'points2d-%s.npy'%self.load_set))
        # self.points3d = np.load(os.path.join(root, 'points3d-%s.npy'%self.load_set))
        
        #if shuffle:
        #    random.shuffle(data)

    def __getitem__(self, index):        
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, points2D, points3D).
        """
        #这是一个元组解包赋值，将函数返回值赋给两个变量：
        # fecg：读取的心电信号数据（数值数组）
        # fields：数据的元信息/属性字典

        #返回值1：fecg（心电图数据）
        # 这是一个 NumPy 数组，形状通常是 (n_samples, n_channels)：
        # n_samples：采样点数（这里是1000个点）
        # n_channels：信号通道数
        # 返回值2：fields（数据属性字典）包含信号的详细描述信息
        fecg, fields = wfdb.rdsamp(self.fecg_paths[index], sampfrom=0, sampto=1000)
        mecg, fields = wfdb.rdsamp(self.mecg_paths[index], sampfrom=0, sampto=1000)
        mixture, fields = wfdb.rdsamp(self.mixture_paths[index], sampfrom=0, sampto=1000)
    
        mixture = mixture[0:992,0]#0:992：包括起始索引0，不包括结束索引992，所以实际取的是0到991共992行 0：只取第0列（索引为0的列）
        #结果： 
        #从原始形状 (1000, 2)变为 (992,)
        #从二维数组变成了一维数组
        #只保留了第一个通道（列索引0）的前992个采样点
        mecg = mecg[0:992,0]
        fecg = fecg[0:992,0]

        # print("mixture shape is -->>>>>>>>>>>>>>>>>",mixture.shape)
        mixture =butter_bandpass_filter(mixture, 3,90, 250, 3)#去3Hz到90Hz，信号采样率为250Hz，滤波器阶数为3
        fecg =butter_bandpass_filter(fecg, 3,90, 250, 3)
        mecg =butter_bandpass_filter(mecg, 3,90, 250, 3)
        
        mixture = (mixture-np.mean(mixture))/np.var(mixture) #这里是标准化，但不是z-score标准化。因为这里分母是方差不是标准差
        fecg = (fecg-np.mean(mixture))/np.var(mixture) 
        mecg = (mecg-np.mean(mixture))/np.var(mixture)
        
        
        
        # b = np.min(mixture)
        # mixture = (mixture - b) 
        # q = np.max(mixture)
        # mixture = mixture / q
        # fecg = (fecg - b) / q
        # mecg = (mecg - b) / q
        
        # print("max and min of mixuter is ->>>", np.min(mixture),"--------------", np.max(mixture))
        
        mixture = np.expand_dims(mixture, axis=1)#增加一个维度，相当于引入了通道的概念，就是列表外面再套了一层空壳可能原来只有行，现在多了列的概念。
        fecg = np.expand_dims(fecg, axis=1)#更多的目的是为了满足torch库接口的数据格式要求
        mecg = np.expand_dims(mecg, axis=1)



        # if self.transform is not None:
            # image = self.transform(image)

        return mixture, mecg, fecg

    def __len__(self):
        return len(self.fecg_paths)
