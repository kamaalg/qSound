import numpy as np
import pyaudio
import time
from scipy.fft import fft, fftfreq

SPEED_OF_SOUND = 343

class AudioHandler(object):
    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.FORMAT = pyaudio.paFloat32
        self.CHANNELS, self.DEVICE_INDEX = self.get_audio_device()  # Set to match BlackHole's 16 channels
        self.RATE = 48000
        self.CHUNK = int(self.RATE * 0.1)  # 100ms
        self.buffer = np.array([], dtype=np.float32)  # Initialize buffer

    def get_audio_device(self):
        for i in range(0, self.p.get_host_api_info_by_index(0).get('deviceCount')):
            device_name: str = self.p.get_device_info_by_host_api_device_index(0,i).get('name')
            # if windows using vb_cable pick lower index - probably won't fix since doesn't matter
            if ("BlackHole" in device_name) or ("CABLE In 16ch" in device_name):
                device_index: int = i
                break

        channels: int = self.p.get_device_info_by_host_api_device_index(0,device_index).get('maxOutputChannels')

        return channels, device_index

    def start(self):
        self.stream = self.p.open(input_device_index=self.DEVICE_INDEX,  # BlackHole 16ch
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

        # Convert multi-channel audio to mono by averaging all channels
        numpy_array = numpy_array.reshape(-1, self.CHANNELS).mean(axis=1)

        # Append new data to the buffer
        self.buffer = np.concatenate((self.buffer, numpy_array))

        # Limit buffer size to last 10 seconds
        max_buffer_size = self.RATE * 10
        if len(self.buffer) > max_buffer_size:
            self.buffer = self.buffer[-max_buffer_size:]

        # Compute RMS (Root Mean Square)
        rms = np.sqrt(np.mean(self.buffer ** 2))
        print(f"Root Mean Square: {rms}")

        # Compute Zero Crossing Rate
        zero_crossings = np.where(np.diff(np.sign(self.buffer)))[0]
        zero_crossing_rate = len(zero_crossings) / len(self.buffer)
        print(f"Zero Crossing Rate: {zero_crossing_rate}")

        # Compute Amplitude
        amplitude = np.max(np.abs(self.buffer))
        print(f"Amplitude: {amplitude}")

        # Compute FFT to find dominant frequency
        fft_values = fft(self.buffer)
        fft_magnitudes = np.abs(fft_values)
        freqs = fftfreq(len(self.buffer), 1 / self.RATE)

        # Get the frequency with the highest magnitude in the positive frequency range
        positive_freqs = freqs[:len(freqs) // 2]
        positive_magnitudes = fft_magnitudes[:len(fft_magnitudes) // 2]
        idx = np.argmax(positive_magnitudes)
        frequency = positive_freqs[idx]
        print(f"Dominant Frequency: {frequency} Hz")

        # Calculate Spectral Centroid
        spectral_centroid = np.sum(positive_freqs * positive_magnitudes) / np.sum(positive_magnitudes)
        print(f"Spectral Centroid: {spectral_centroid} Hz")

        # Calculate Time Period and Wavelength
        time_period = 1 / frequency if frequency != 0 else float('inf')
        print(f"Time Period: {time_period} s")
        wavelength = SPEED_OF_SOUND / frequency if frequency != 0 else float('inf')
        print(f"Wavelength: {wavelength} m")

        # Estimate BPM (Beats Per Minute)
        bpm = self.estimate_bpm()
        print(f"Estimated BPM: {bpm}")

        return None, pyaudio.paContinue

    def estimate_bpm(self):
        # Parameters for onset detection
        window_size = 1024
        hop_size = 512

        # Compute the energy of each frame
        energy = []
        for i in range(0, len(self.buffer) - window_size, hop_size):
            frame = self.buffer[i:i + window_size]
            frame_energy = np.sum(frame ** 2)
            energy.append(frame_energy)
        energy = np.array(energy)

        # Normalize energy
        energy = energy / np.max(energy)

        # Detect peaks in energy
        peaks = []
        threshold = 0.6  # You may need to adjust this threshold
        for i in range(1, len(energy) - 1):
            if energy[i] > threshold and energy[i] > energy[i - 1] and energy[i] > energy[i + 1]:
                peaks.append(i)

        # Calculate intervals between peaks
        if len(peaks) > 1:
            peak_times = np.array(peaks) * hop_size / self.RATE
            intervals = np.diff(peak_times)
            avg_interval = np.mean(intervals)
            bpm = 60 / avg_interval
        else:
            bpm = 0  # Not enough peaks to estimate BPM

        return bpm

    def mainloop(self):
        while self.stream.is_active():
            time.sleep(0.1)

audio = AudioHandler()
audio.start()
audio.mainloop()
audio.stop()