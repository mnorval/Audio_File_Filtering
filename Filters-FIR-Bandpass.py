audio='AudioClip-Noise-Added.wav' #location

from scipy.signal import butter, lfilter
import numpy as np
import librosa
import soundfile as sf

def butter_bandpass(lowcut, highcut, fs, order=5):
    return butter(order, [lowcut, highcut], fs=fs, btype='band')

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import freqz

    # Sample rate and desired cutoff frequencies (in Hz).
    fs = 44100.0
    lowcut = 500.0
    highcut = 8000.0

    signal, sample_rate = librosa.load(audio)
    Time = np.linspace(0, len(signal) / sample_rate, num=len(signal))

    filtered_signal = butter_bandpass_filter(signal, lowcut, highcut, fs, order=6)
    

    # Write out the denoised audio
    sf.write('AudioClip-Bandpass_Filter.wav', filtered_signal, sample_rate)


    plt.figure(figsize=(15,6))
    plt.subplot(3, 1, 1)
    plt.title('Original Time Domain Signal')
    plt.plot(Time, signal,'tab:blue')

    plt.subplot(3, 1, 2)
    plt.title('Bandpass Filter Applied')
    plt.plot(Time, filtered_signal,'tab:orange')

    plt.subplot(3, 1, 3)
    plt.title('Combined')
    plt.plot(Time, signal,'tab:blue')
    plt.plot(Time, filtered_signal,'tab:orange')

    plt.tight_layout()

    plt.show()


    """
    # Plot the frequency response for a few different orders.
    plt.figure(1)
    plt.clf()
    for order in [3, 6, 9]:
        b, a = butter_bandpass(lowcut, highcut, fs, order=order)
        w, h = freqz(b, a, fs=fs, worN=2000)
        plt.plot(w, abs(h), label="order = %d" % order)

    plt.plot([0, 0.5 * fs], [np.sqrt(0.5), np.sqrt(0.5)],
             '--', label='sqrt(0.5)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Gain')
    plt.grid(True)
    plt.legend(loc='best')

    # Filter a noisy signal.
    T = 0.05
    nsamples = T * fs
    t = np.arange(0, nsamples) / fs
    a = 0.02
    f0 = 600.0
    x = 0.1 * np.sin(2 * np.pi * 1.2 * np.sqrt(t))
    x += 0.01 * np.cos(2 * np.pi * 312 * t + 0.1)
    x += a * np.cos(2 * np.pi * f0 * t + .11)
    x += 0.03 * np.cos(2 * np.pi * 2000 * t)
    plt.figure(2)
    plt.clf()
    plt.plot(t, x, label='Noisy signal')

    y = butter_bandpass_filter(x, lowcut, highcut, fs, order=6)
    plt.plot(t, y, label='Filtered signal (%g Hz)' % f0)
    plt.xlabel('time (seconds)')
    plt.hlines([-a, a], 0, T, linestyles='--')
    plt.grid(True)
    plt.axis('tight')
    plt.legend(loc='upper left')

    plt.show()
    """