audio='AudioClip-Noise-Added.wav' 

import numpy as np
import librosa
from scipy.io import wavfile
from scipy.fftpack import dct
from matplotlib import pyplot as plt
import noisereduce as nr
import soundfile as sf



# Load an audio file
signal, sample_rate = librosa.load(audio)
Time = np.linspace(0, len(signal) / sample_rate, num=len(signal))

# Compute short-time Fourier transform (STFT)
D = librosa.stft(signal)

# Compute the average spectrum (this will be our noise estimate)
avg_spectrum = np.mean(np.abs(D), axis=1)

# Generate a mask by comparing the spectrum of each frame to the average spectrum
mask = (np.abs(D) > avg_spectrum[:, np.newaxis])

# Apply the mask to the STFT, effectively subtracting the average spectrum
D_denoised = D * mask

# Compute the inverse STFT to convert back to the time domain
signal_denoised = librosa.istft(D_denoised)
Time_denoised = np.linspace(0, len(signal_denoised) / sample_rate, num=len(signal_denoised))

# Write out the denoised audio
sf.write('AudioClip-Noise-Removed-Spectral-Subtract.wav', signal_denoised, sample_rate)


plt.figure(figsize=(15,6))
plt.subplot(3, 1, 1)
plt.title('Original Time Domain Signal')
plt.plot(Time, signal,'tab:blue')

plt.subplot(3, 1, 2)
plt.title('Spectral Gating Applied')
plt.plot(Time_denoised, signal_denoised,'tab:orange')

plt.subplot(3, 1, 3)
plt.title('Combined')
plt.plot(Time, signal,'tab:blue')
plt.plot(Time_denoised, signal_denoised,'tab:orange')

plt.tight_layout()

plt.show()










"""
# load input file, and stft (Short-time Fourier transform)
signal, sample_rate = librosa.load( audio, sr=None, mono=True) # keep native sr (sampling rate) and trans into mono
#sample_rate, signal = wavfile.read(audio)
signal_fourier= librosa.stft(signal)                     # Short-time Fourier transform
signal_fourier_magnitude= np.abs(signal_fourier)         # get magnitude
signal_fourier_angle= np.angle(signal_fourier)           # get phase
signal_fourier_phase=np.exp(1.0j* signal_fourier_angle)  # use this phase information when Inverse Transform


# load noise only file, stft, and get mean
#noise_sample_rate, noise_signal = wavfile.read(audio_noise)
noise_signal, noise_sample_rate = librosa.load( audio_noise, sr=None, mono=True) # keep native sr (sampling rate) and trans into mono
noise_signal_fourier= librosa.stft(noise_signal) 
noise_signal_fourier_magnitude= np.abs(noise_signal_fourier)
noise_signal_fourier_mean = np.mean(noise_signal_fourier_magnitude, axis=1) # get mean

# subtract noise spectral mean from input spectral, and istft (Inverse Short-Time Fourier Transform)
sa= signal_fourier_magnitude - noise_signal_fourier_mean.reshape((noise_signal_fourier_mean.shape[0],1))  # reshape for broadcast to subtract
sa0= sa * signal_fourier_phase  # apply phase information
y= librosa.istft(sa0) # back to time domain signal



plt.subplot(3, 2, 1)
plt.ylim([-1, 1])
plt.rcParams['figure.figsize'] = [2, 2]
plt.title('Original Signal')
plt.plot(signal,'tab:blue')

plt.subplot(3, 2, 2)
plt.ylim([0, 80])
plt.rcParams['figure.figsize'] = [2, 2]
plt.title('Original Signal Spectrum')
plt.plot(signal_fourier_magnitude,'tab:blue')


plt.subplot(3, 2, 3)
plt.ylim([-1, 1])
plt.rcParams['figure.figsize'] = [2, 2]
plt.title('Background Noise Signal')
plt.plot(noise_signal,'tab:orange')

plt.subplot(3, 2, 4)
plt.ylim([0, 80])
plt.rcParams['figure.figsize'] = [2, 2]
plt.title('Background Noise Signal Spectrum')
plt.plot(noise_signal_fourier_magnitude,'tab:orange')



plt.subplot(3, 2, 5)
plt.ylim([-1, 1])
plt.title('Noise Subtracted')
plt.plot(signal,'tab:blue')
plt.plot(y,'tab:orange')


plt.subplot(3, 2, 6)
plt.ylim([0, 80])
plt.rcParams['figure.figsize'] = [2, 2]
plt.title('Noise Subtracted Spectrum')
plt.plot(signal_fourier_magnitude,'tab:blue')
plt.plot(noise_signal_fourier_magnitude,'tab:orange')


plt.tight_layout()

plt.show()
"""