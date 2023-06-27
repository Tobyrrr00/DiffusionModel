import numpy as np
import librosa, librosa.display
import matplotlib.pyplot as plt

file = "LecturesFollow\\blues.00000.wav"
FIG_SIZE = (15,10)
# load audio file with Librosa
#SR = sample rate. t = time of soundfile
signal, sr = librosa.load(file, sr = 22050) #sr * T -> 22050 * 30
#Note: waveshow is called waveplot in older version of librosa
librosa.display.waveshow(signal)
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.show()

#fft (fast fourier transform) -> power spectrum
fft = np.fft.fft(signal)
# calculate abs values on complex numbers to get magnitude
spectrum = np.abs(fft)
# create frequency variable
f = np.linspace(0, sr, len(spectrum))
# take half of the spectrum and frequency
left_spectrum = spectrum[:int(len(spectrum)/2)]
left_f = f[:int(len(spectrum)/2)]

plt.plot(left_f, left_spectrum)
plt.xlabel("Frequency")
plt.ylabel("Magnitude")
plt.show()

#stft (Short-time Fourier transform) -> spectogram
hop_length = 512 # Size of window we are doing stft on 
n_fft = 2048 # Number of samples considering
stft = librosa.core.stft(signal, hop_length=hop_length, n_fft=n_fft)
# calculate abs values on complex numbers to get magnitude
spectrogram = np.abs(stft)


librosa.display.specshow(spectrogram, sr=sr, hop_length=hop_length)
plt.xlabel("Time")
plt.ylabel("Frequency")
#plt.colorbar() # note, this will throw a bug on Matplotlib v. > 3.7.0. But 3.6.0 works
plt.title("Spectrogram")
plt.show()

# MFCCs
MFCCs = librosa.feature.mfcc(y=signal, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mfcc=13)
librosa.display.specshow(MFCCs, sr=sr, hop_length=hop_length)
plt.xlabel("Time")
plt.ylabel("MFCC")
#plt.colorbar
plt.show()