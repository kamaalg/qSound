import numpy as np
import pyaudio
import time
import librosa
import matplotlib.pyplot as plot
from scipy.fft import fft, fftfreq

SPEED_OF_SOUND = 343

#p = pyaudio.PyAudio()
#info = p.get_host_api_info_by_index(0)
#numdevices = info.get('deviceCount')
#
#for i in range(0, numdevices):
#    if (p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
#        print("Input Device id ", i, " - ", p.get_device_info_by_host_api_device_index(0, i).get('name'))


class AudioHandler(object):
    def __init__(self):
        self.FORMAT = pyaudio.paFloat32
        self.CHANNELS = 1
        self.RATE = 48000
        self.CHUNK = 1024
        self.p = None
        self.stream = None
        self.metastream: np.ndarray

    def start(self):
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(input_device_index=2, #TODO: this needs to not be hardcoded
                                  format=self.FORMAT,
                                  channels=self.CHANNELS,
                                  rate=self.RATE,
                                  input=True,
                                  output=False,
                                  stream_callback=self.callback,
                                  frames_per_buffer=self.CHUNK)

    def stop(self):
        self.stream.close()
        self.p.terminate()

    def callback(self, in_data, frame_count, time_info, flag):
        print(in_data)
        numpy_array = np.frombuffer(in_data, dtype=np.float32)
        #plot.specgram(numpy_array)
        #plot.show()
        print(numpy_array)
        #bpm = librosa.feature.tempo(y=numpy_array) -> record peaks from this iteration only
        # or build meta-stream arraylist??
        #print(f"BPM: {bpm}") #debug
        rms = librosa.feature.rms(y=numpy_array)
        print(f"Root mean square: {rms})")
        spectral_centroid = librosa.feature.spectral_centroid(y=numpy_array)
        print(f"Spectral centroid: {spectral_centroid}")
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y=numpy_array)
        print(f"Zero crossing rate: {zero_crossing_rate}")

        amplitude = np.max(np.abs(numpy_array))
        print(f"Amplitude: {amplitude}")

        # Compute FFT to find dominant frequency
        fft_values = fft(numpy_array)
        fft_magnitudes = np.abs(fft_values)
        freqs = fftfreq(len(numpy_array), 1 / self.RATE)

        # Get the frequency with the highest magnitude in the positive frequency range
        idx = np.argmax(fft_magnitudes[:len(fft_magnitudes) // 2])
        frequency = abs(freqs[idx])  # Dominant frequency
        print(f"Dominant Freq: {frequency}")

        # Calculate time period and wavelength
        time_period = 1 / frequency if frequency != 0 else float('inf')
        print(f"Time period: {time_period}")
        wavelength = SPEED_OF_SOUND / frequency if frequency != 0 else float('inf')
        print(f"Wavelength: {wavelength}")
        return None, pyaudio.paContinue

    def mainloop(self):
        while (self.stream.is_active()):
            time.sleep(2.0)


audio = AudioHandler()
audio.start()
audio.mainloop()
audio.stop()