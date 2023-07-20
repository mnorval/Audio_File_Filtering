audio_clip='AudioClip-Silence-Added.wav' #location

import librosa
import soundfile as sf
import numpy as np
from matplotlib import pyplot as plt
from pydub import AudioSegment
from pydub.silence import split_on_silence
from functools import reduce

import os

original_signal, original_sample_rate = librosa.load(audio_clip)
original_Time_signal = np.linspace(0, len(original_signal) / original_sample_rate, num=len(original_signal))

# Load your audio.
audio = AudioSegment.from_wav(audio_clip)

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

signal_silence_removed, sample_rate_silence_removed = librosa.load('AudioClip-Silence-Removed.wav')
Time_signal_silence_removed = np.linspace(0, len(signal_silence_removed) / sample_rate_silence_removed, num=len(signal_silence_removed))


# Write out the denoised audio
#sf.write('AudioClip_Silence_Removed.wav', audio_trimmed, sample_rate)


plt.figure(figsize=(15,6))
plt.subplot(3, 1, 1)
plt.title('Original Time Domain Signal')
plt.plot(original_Time_signal,original_signal,'tab:blue')

plt.subplot(3, 1, 2)
plt.title('Silent Sections Applied')
plt.plot(Time_signal_silence_removed, signal_silence_removed,'tab:orange')


#plt.subplot(3, 1, 3)
#plt.title('Combined')
#plt.plot(Time, signal,'tab:blue')
#plt.plot(Time_trimmed, audio_trimmed,'tab:orange')

plt.tight_layout()

plt.show()


"""
# Clearing the Screen
os.system('cls')


#read wav data
audio, sr = librosa.load(audio, sr= 8000, mono=True)
print(audio.shape, sr)

clips = librosa.effects.split(audio, top_db=1.6)
print(clips)

audio_silence_removed = []
for c in clips:
    #print(c)
    data = audio[c[0]: c[1]]
    audio_silence_removed.extend(data)




plt.figure(figsize=(15,6))


plt.subplot(3, 1, 1)
plt.title('Original Audio Clip')
plt.plot(audio)

plt.xlim(-0, 7000)
plt.subplot(3, 1, 2)
plt.title('Silence Removal Applied')
plt.plot(audio_silence_removed,'tab:orange')
plt.xlim(-0, 7000)

plt.subplot(3, 1, 3)
plt.title('Combined')
plt.plot(audio_silence_removed,'tab:orange')
plt.plot(audio)
plt.xlim(-0, 7000)

plt.tight_layout()


plt.show()
"""