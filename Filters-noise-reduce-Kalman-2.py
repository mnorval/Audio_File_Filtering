audio='AudioClip-Noise-Added.wav' #location

import numpy as np
import matplotlib.pyplot as plt
from tsmoothie.smoother import *
from scipy.io import wavfile

#mu, sigma = 0, 500
#x = np.arange(1, 100, 0.1)  # x axis
#z = np.random.normal(mu, sigma, len(x))  # noise
#y = x ** 2 + z # data

sample_rate, signal = wavfile.read(audio)
y = signal[0:int(10* sample_rate)]
x = np.linspace(0, len(signal) / sample_rate, num=len(signal))

# operate smoothing
#smoother = ConvolutionSmoother(window_len=30, window_type='ones')
smoother = KalmanSmoother(component='level_trend',component_noise={'level':0.5, 'trend':0.5})
#KalmanSmoother
smoother.smooth(y)

# generate intervals
low, up = smoother.get_intervals('sigma_interval', n_sigma=3)


wavfile.write("AudioClip-Noise-Removed-Kalman-2.wav",sample_rate,smoother.data[0].astype(np.int16))

plt.figure(figsize=(15,6))
plt.subplot(3, 1, 1)
plt.title('Original Time Domain Signal')
plt.plot(x, y,'tab:blue')

plt.subplot(3, 1, 2)
plt.title('Noise Reduction Applied')
plt.plot(x, smoother.data[0],'tab:orange')

plt.subplot(3, 1, 3)
plt.title('Combined')
plt.plot(x, y,'tab:blue')
plt.plot(x, smoother.data[0],'tab:orange')



plt.tight_layout()

plt.show()
# plot the smoothed timeseries with intervals
#plt.figure(figsize=(11,6))
#plt.plot(smoother.data[0], color='orange')
#plt.plot(smoother.smooth_data[0], linewidth=3, color='blue')
#plt.fill_between(range(len(smoother.data[0])), low[0], up[0], alpha=0.3)

#plt.show()