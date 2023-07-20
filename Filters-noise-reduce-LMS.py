audio='AudioClip-Noise-Added.wav' #location

import numpy as np
import librosa
import librosa.display
from scipy.io import wavfile
from scipy.fftpack import dct
from matplotlib import pyplot as plt
import noisereduce as nr
import padasip as pa

def zs(a):
    """ 1d data z-score """
    a -= a.mean()
    return a / a.std()


SAMPLERATE = 44100
n = 300 # filter size
D = 200 # signal delay

#sample_rate, signal = wavfile.read(audio)
# open and process source data
fs, data = wavfile.read(audio)
y = data
y = y.astype("float64")
y = zs(y) / 10
N = len(y)

# contaminated with noise
q = np.sin(2*np.pi*1000/99*np.arange(N) + 10.1 * np.sin(2*np.pi/110*np.arange(N)))
d = y #+ q

# prepare data for simulation
x = pa.input_from_history(d, n)[:-D]
d = d[n+D-1:]
y = y[n+D-1:]
q = q[n+D-1:]

# create filter and filter
f = pa.filters.FilterNLMS(n=n, mu=0.05, w="zeros")
yp, e, w = f.run(d, x)

#signal = signal[0:int(10* sample_rate)]
#Time = np.linspace(0, len(signal) / sample_rate, num=len(signal))




#wavfile.write("AudioClip-Noise-Removed-Spectral Gating.wav",sample_rate,lms_signal.astype(np.int16))
wavfile.write("AudioClip-Noise-Removed-LMS.wav",fs,yp)

plt.figure(figsize=(15,6))
plt.subplot(3, 1, 1)
plt.title('Original Time Domain Signal')
plt.plot(y,'tab:blue')


plt.subplot(3, 1, 2)
plt.title('Filtere Record')
plt.plot(yp,'tab:orange')

plt.subplot(3, 1, 3)
plt.title('Combined')
plt.plot(y,'tab:blue')
plt.plot(yp,'tab:orange')

plt.tight_layout()

plt.show()

