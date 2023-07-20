audio='AudioClip-Noise-Added.wav' #location

import numpy as np
import librosa.display
from scipy.io import wavfile
from scipy.fftpack import dct
from matplotlib import pyplot as plt
import noisereduce as nr

sample_rate, signal = wavfile.read(audio)

signal = signal[0:int(10* sample_rate)]
Time = np.linspace(0, len(signal) / sample_rate, num=len(signal))
emphasized_signal = nr.reduce_noise(y=signal, sr=sample_rate)

wavfile.write("AudioClip-Noise-Removed-Generic.wav",sample_rate,emphasized_signal.astype(np.int16))

plt.figure(figsize=(15,6))
plt.subplot(3, 1, 1)
plt.title('Original Time Domain Signal')
plt.plot(Time, signal)

plt.subplot(3, 1, 2)
plt.title('Noise Reduction Applied')
plt.plot(Time, emphasized_signal,'tab:orange')

plt.subplot(3, 1, 3)
plt.title('Combined')
plt.plot(Time, signal)
plt.plot(Time, emphasized_signal)

plt.tight_layout()

plt.show()