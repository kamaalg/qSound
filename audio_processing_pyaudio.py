import numpy as np
import pyaudio
import time
import librosa
import matplotlib.pyplot as plot

class AudioHandler(object):
    def __init__(self):
        self.FORMAT = pyaudio.paFloat32
        self.CHANNELS = 1
        self.RATE = 48000
        self.CHUNK = 1024 * 2
        self.p = None
        self.stream = None
        self.metastream: np.ndarray

    def start(self):
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(input_device_index=2,
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
        numpy_array = np.frombuffer(in_data, dtype=np.float32)
        #plot.specgram(numpy_array)
        #plot.show()
        print(numpy_array)
        #bpm = librosa.feature.tempo(y=numpy_array) -> record peaks from this iteration only
        # or build meta-stream arraylist??
        #print(f"BPM: {bpm}") #debug
        rms = librosa.feature.rms(y=numpy_array)
        print(f"RMS: {rms})")
        spectral_centroid = librosa.feature.spectral_centroid(y=numpy_array)
        print(f"SC: {spectral_centroid}")
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y=numpy_array)
        print(f"ZCR: {zero_crossing_rate}")
        return None, pyaudio.paContinue

    def mainloop(self):
        while (self.stream.is_active()):
            time.sleep(2.0)


audio = AudioHandler()
audio.start()
audio.mainloop()
audio.stop()