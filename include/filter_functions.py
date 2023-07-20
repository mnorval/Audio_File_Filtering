import numpy as np
import librosa.display
from scipy.io import wavfile
from scipy.fftpack import dct
from matplotlib import pyplot as plt
import noisereduce as nr
import os
from pykalman import KalmanFilter
from tsmoothie.smoother import *
from scipy.signal import butter, lfilter
from pydub import AudioSegment
from pydub.silence import split_on_silence
from functools import reduce

def filter_noise_remove_generic(dirpath, file):
    sample_rate, signal = wavfile.read(os.path.join(dirpath, file))

    signal = signal[0:int(10* sample_rate)]
    Time = np.linspace(0, len(signal) / sample_rate, num=len(signal))
    emphasized_signal = nr.reduce_noise(y=signal, sr=sample_rate)

    output_filename = os.path.join(dirpath, "filter_noise_remove_generic_"+file)
    print("Write File:" + "_"+output_filename)
    wavfile.write(output_filename,sample_rate,emphasized_signal.astype(np.int16))
    return 

def filter_noise_remove_kalman_1(dirpath, file):
    
    sample_rate, signal = wavfile.read(os.path.join(dirpath, file))
    n_timesteps = sample_rate
    observations = signal[0:int(10* sample_rate)]
    x = np.linspace(0, len(signal) / sample_rate, num=len(signal))
    
    kf = KalmanFilter(transition_matrices=np.array([[1, 1], [0, 1]]),transition_covariance=0.01 * np.eye(2))
    states_pred = kf.em(signal, n_iter=1).smooth(signal)[0]

    output_filename = os.path.join(dirpath, "filter_noise_remove_kalman_1_"+file)
    print("Write File:" + "_"+output_filename)
    wavfile.write(output_filename,sample_rate,states_pred[:, 0].astype(np.int16))
    return 

def filter_noise_remove_kalman_2(dirpath, file):
    sample_rate, signal = wavfile.read(os.path.join(dirpath, file))
    
    signal = signal[0:int(10* sample_rate)]
    Time = np.linspace(0, len(signal) / sample_rate, num=len(signal))

    smoother = KalmanSmoother(component='level_trend',component_noise={'level':0.5, 'trend':0.5})
    smoother.smooth(signal)


    output_filename = os.path.join(dirpath, "filter_noise_remove_kalman_2_"+file)
    print("Write File:" + "_"+output_filename)
    wavfile.write(output_filename,sample_rate,signal.astype(np.int16))
    return 


#*****************************************
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
#*****************************************

def filter_noise_remove_spectral_gating(dirpath, file):
    signal, sample_rate = librosa.load(os.path.join(dirpath, file))
    
    Time = np.linspace(0, len(signal) / sample_rate, num=len(signal))

    # Apply spectral gating with a threshold
    cleaned_signal = spectral_gating(signal, sample_rate, threshold=3)
    CleanedSignalTime = np.linspace(0, len(cleaned_signal) / sample_rate, num=len(cleaned_signal))

    output_filename = os.path.join(dirpath, "filter_noise_remove_spectral_gating_"+file)
    print("Write File:" + "_"+output_filename)
    wavfile.write(output_filename,sample_rate,cleaned_signal)
    return 


def filter_noise_remove_spectral_subtract(dirpath, file):
    signal, sample_rate = librosa.load(os.path.join(dirpath, file))    
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

    output_filename = os.path.join(dirpath, "filter_noise_remove_spectral_subtract_"+file)
    print("Write File:" + "_"+output_filename)
    wavfile.write(output_filename,sample_rate,signal_denoised)    
    return 

#********************************************************
def butter_bandpass(lowcut, highcut, fs, order=5):
    return butter(order, [lowcut, highcut], fs=fs, btype='band')

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y
#********************************************************


def filter_bandpass(dirpath, file):
    signal, sample_rate = librosa.load(os.path.join(dirpath, file))    
     # Sample rate and desired cutoff frequencies (in Hz).
    fs = 44100.0
    lowcut = 500.0
    highcut = 8000.0

    filtered_signal = butter_bandpass_filter(signal, lowcut, highcut, fs, order=6)

    output_filename = os.path.join(dirpath, "filter_bandpass_"+file)
    print("Write File:" + "_"+output_filename)
    wavfile.write(output_filename,sample_rate,filtered_signal)    
    return 


def filter_silence_remove(dirpath, file):
    audio = AudioSegment.from_wav(os.path.join(dirpath, file))    
    
    original_signal, original_sample_rate = librosa.load(os.path.join(dirpath, file))
    original_Time_signal = np.linspace(0, len(original_signal) / original_sample_rate, num=len(original_signal))

    try:
        # Split audio into chunks where silence is detected.
        chunks = split_on_silence(
        audio,

        # Specify that a silent chunk must be at least 2 seconds or 2000 ms long.
        min_silence_len=100,

        # Consider a chunk silent if it's quieter than -16 dBFS.
        silence_thresh=-30
    )

        # Now we just concatenate all the non-silent chunks.
        non_silent_audio = reduce(lambda a, b: a + b, chunks)

        # Saving the result.
        non_silent_audio.export("AudioClip-Silence-Removed.wav", format="wav")


        output_filename = os.path.join(dirpath, "filter_silence_remove_"+file)
        print("Write File:" + "_"+output_filename)
        non_silent_audio.export(output_filename, format="wav")  
    except:
        print("An exception occurred: " + os.path.join(dirpath, file))

    return 