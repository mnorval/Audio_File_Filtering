audio='AudioClip-Noise-Added.wav' #location
#audio='AudioClip_Short.wav'

import numpy as np
import pylab as pl
from scipy.io import wavfile
from matplotlib import pyplot as plt
from pykalman import KalmanFilter

rnd = np.random.RandomState(0)

# generate a noisy sine wave to act as our fake observations
sample_rate, signal = wavfile.read(audio)
signal = signal[0:int(10* sample_rate)]
Time = np.linspace(0, len(signal) / sample_rate, num=len(signal))
print("Starting")

n_timesteps = 100
#n_timesteps = signal[1]/sample_rate
#x = np.linspace(0, 3 * np.pi, n_timesteps)
#x = np.linspace(0, 3 * np.pi, n_timesteps)

#observations = 20 * (np.sin(x) + 0.5 * rnd.randn(n_timesteps))

#observations = signal[0:int(10* sample_rate)]
#observations = signal

#Read Audio File
sample_rate, signal = wavfile.read(audio)
n_timesteps = sample_rate
observations = signal[0:int(10* sample_rate)]
x = np.linspace(0, len(signal) / sample_rate, num=len(signal))




# create a Kalman Filter by hinting at the size of the state and observation
# space.  If you already have good guesses for the initial parameters, put them
# in here.  The Kalman Filter will try to learn the values of all variables.
print("Before filter")
kf = KalmanFilter(transition_matrices=np.array([[1, 1], [0, 1]]),transition_covariance=0.01 * np.eye(2))
#kf = KalmanFilter(transition_matrices = [[1, 1], [0, 1]], observation_matrices = [[0.1, 0.5], [-0.3, 0.0]])

# You can use the Kalman Filter immediately without fitting, but its estimates
# may not be as good as if you fit first.

print("After filter - 1")
#states_pred = kf.em(signal).smooth(signal)[0]
states_pred = kf.em(signal, n_iter=1).smooth(signal)[0]
#states_pred = kf.em(signal, n_iter=1)
print("After filter - 2")
#print('fitted model: {0}'.format(kf))



# Plot lines for the observations without noise, the estimated position of the
# target before fitting, and the estimated position after fitting.

wavfile.write("AudioClip-Noise-Removed-Kalman-1.wav",sample_rate,states_pred[:, 0].astype(np.int16))
#wavfile.write("AudioClip-Noise-Removed-2.wav",sample_rate,states_pred[:, 1].astype(np.int16))



plt.figure(figsize=(15,6))
plt.subplot(3, 1, 1)
plt.title('Original Time Domain Signal')
plt.plot(Time, signal,'tab:blue')

plt.subplot(3, 1, 2)
plt.title('Noise Reduction Applied')
plt.plot(Time, states_pred[:, 0],'tab:orange')

plt.subplot(3, 1, 3)
plt.title('Noise Reduction Applied')
plt.plot(Time, signal,'tab:blue')
plt.plot(Time, states_pred[:, 0],'tab:orange')

plt.tight_layout()

plt.show()
"""
pl.figure(figsize=(16, 6))
obs_scatter = pl.scatter(x, observations, marker='x', color='b',
                         label='observations')
position_line = pl.plot(x, states_pred[:, 0],
                        linestyle='-', marker='o', color='r',
                        label='position est.')
velocity_line = pl.plot(x, states_pred[:, 1],
                        linestyle='-', marker='o', color='g',
                        label='velocity est.')
pl.legend(loc='lower right')
pl.xlim(xmin=0, xmax=x.max())
pl.xlabel('time')
pl.show()
"""