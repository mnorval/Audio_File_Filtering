audio='AudioClip-Noise-Added.wav' #location

import numpy as np
import librosa
import librosa.display
from scipy.io import wavfile
from scipy.fftpack import dct
from matplotlib import pyplot as plt
import noisereduce as nr

import numpy as np
import librosa

def spectral_gating(signal, sr, threshold):
    # Compute the short-time Fourier transform (STFT)
    stft = librosa.stft(signal)

    # Compute the magnitude of the STFT
    magnitude, phase = librosa.magphase(stft)

    # Identify the indices where the magnitude is below the threshold
    low_values_indices = magnitude < threshold  # Where values are low
    magnitude[low_values_indices] = 0  # All low values set to 0

    # Reconstruct the signal
    cleaned_stft = magnitude * phase
    y = librosa.istft(cleaned_stft)

    return y

# Load an audio file
signal, sample_rate = librosa.load(audio)
Time = np.linspace(0, len(signal) / sample_rate, num=len(signal))

# Apply spectral gating with a threshold
cleaned_signal = spectral_gating(signal, sample_rate, threshold=3)
CleanedSignalTime = np.linspace(0, len(cleaned_signal) / sample_rate, num=len(cleaned_signal))

# Compare original and cleaned signal
#librosa.display.waveshow(signal, sr=sample_rate, alpha=0.7)
#librosa.display.waveshow(cleaned_signal, sr=sample_rate, color='r', alpha=0.5)

wavfile.write("AudioClip-Noise-Removed-Spectral Gating.wav",sample_rate,cleaned_signal)


plt.figure(figsize=(15,6))
plt.subplot(3, 1, 1)
plt.title('Original Time Domain Signal')
plt.plot(Time, signal,'tab:blue')

plt.subplot(3, 1, 2)
plt.title('Spectral Gating Applied')
plt.plot(CleanedSignalTime, cleaned_signal,'tab:orange')

plt.subplot(3, 1, 3)
plt.title('Combined')
plt.plot(Time, signal,'tab:blue')
plt.plot(CleanedSignalTime, cleaned_signal,'tab:orange')

plt.tight_layout()

plt.show()
