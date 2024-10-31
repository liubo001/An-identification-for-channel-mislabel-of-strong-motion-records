import numpy as np
import re
import matplotlib.pyplot as plt
from obspy.signal.cross_correlation import correlate, xcorr_max
from scipy.fft import fft, fftfreq

def konno_ohmachi_smoothing(signal, freq_array, smooth_coeff):
    """
    Original paper:
        K. Konno & T. Ohmachi (1998) "Ground-motion characteristics estimated
        from spectral ratio between horizontal and vertical components of
        microtremor." Bulletin of the Seismological Society of America.
        Vol.88, No.1, 228-241.
        
    Parameters:
    signal: Signal to be smoothed in frequency domain.
    freq_array: Frequency array corresponding to the signal (must have the same length as signal).
    smooth_coeff: A parameter determining the degree of smoothing. Lower values increase smoothing.
    
    Returns:
    y: Smoothed signal as a 1D numpy array.
    """
    
    x = np.array(signal)
    f = np.array(freq_array)
    f_shifted = f / (1 + 1e-4)
    L = len(x)
    y = np.zeros(L)

    for i in range(1, L - 1):
        z = f_shifted / f[i]
        w = (np.sin(smooth_coeff * np.log10(z)) / smooth_coeff / np.log10(z)) ** 4
        w[np.isnan(w)] = 0
        y[i] = np.dot(w, x) / np.sum(w)

    # Set the boundary values
    y[0] = y[1]
    y[-1] = y[-2]

    return y

def peer_read(file_name):
    # READ THE GROUND MOTION ACCELERATION RECORD OF PEER
    # Input:
    #    file_name: Contains path and file name of acceleration record
    # Output:
    #    dt: Time step of the ground motion acceleration record
    #    npts: Number of data points in the record
    #    data: Column vector of ground motion acceleration record
    #    units: cm/s^2

    ti_number = 4
    with open(file_name, 'r') as file:
        for i in range(1, ti_number + 1):
            line = file.readline()
            if i == 4:
                k = re.findall(r'\d+\.\d+|\d+', line)
                if k:
                    npts = int(k[0])
                    if len(k) > 1:
                        dt = float(k[1])
                    else:
                        dt = float(k[0])
                else:
                    raise ValueError("Unable to find numeric values in the fourth line.")

    with open(file_name, 'r') as file:
        # Skip the first 4 lines
        for _ in range(4):
            file.readline()

        data = [float(value) for value in file.read().split()]

    data = [value * 980 for value in data]

    return dt, npts, data

[dt, npts, Data] = peer_read(".\\PEER Original Format\\RSN11341_10275733_N5471HNE.AT2")
data = Data / np.max(np.abs(Data))
sampling_rate = 1/dt  # 采样率
time = np.arange(0, npts * dt, dt)

# 3. 绘制时域波形图
fig, ax = plt.subplots(figsize=(224/100, 224/100))  # 图像大小设置为 224x224 像素
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)  # 去除图像边距

# 绘制图像
ax.plot(time, data, 'k', linewidth=1)
ax.axis('off')  # 关闭坐标轴
# plt.show()
plt.savefig("./RSN11341_WF.png")

# 4. 计算傅里叶变换并绘制频谱图
n = len(data)
freq = fftfreq(n, d=1/sampling_rate)[:n//2]  # 仅考虑正频率部分
fft_values = np.abs(fft(data))[:n//2]  # 傅里叶变换后取绝对值


# 5. 使用 Konno-Ohmachi 平滑傅里叶谱
bandwidth = 40  # 平滑系数
smoothed_fft_values = konno_ohmachi_smoothing(fft_values, freq, bandwidth)

fig, ax = plt.subplots(figsize=(224/100, 224/100))  # 图像大小设置为 224x224 像素
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)  # 去除图像边距

# 绘制图像
ax.plot(freq, smoothed_fft_values, 'k', linewidth=1)
plt.xscale("log")  # 设置 x 轴为对数坐标
plt.yscale("log")  # 设置 y 轴为对数坐标
ax.axis('off')  # 关闭坐标轴
# plt.show()
plt.savefig("./RSN11341_FAS.png")




# [dt, npts, Data_H1] = peer_read(".\\PEER Original Format\\RSN11341_10275733_N5471HNE.AT2")
# [dt, npts, Data_H2] = peer_read(".\\PEER Original Format\\RSN11341_10275733_N5471HNN.AT2")
# [dt, npts, Data_UD] = peer_read(".\\PEER Original Format\\RSN11341_10275733_N5471HNZ.AT2")

# Data_EW = Data_H1 / np.max(np.abs(Data_H1))
# Data_NS = Data_H2 / np.max(np.abs(Data_H2))
# Data_UD = Data_UD / np.max(np.abs(Data_UD))

# a = len(Data_EW) - 1
# c = correlate(Data_H1, Data_H2, a)
# print(np.max(c))

# c2 = correlate(Data_H1, Data_UD, a)
# print(np.max(c2))

# c3 = correlate(Data_H2, Data_UD, a)
# print(np.max(c3))
