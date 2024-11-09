import sounddevice as sd
import numpy as np
import sys
from scipy.signal import find_peaks
from scipy.fft import fft, fftfreq
import pandas as pd
import librosa as lr

keys = [
    "amplitude",
    "frequency",
    "time_period",
    "wavelength",
    "tempo",
    "rms",
    "spectral_centroid",
    "zero_crossing_rate"
]
this_run_data = {key: np.zeros(64) for key in keys}
current_iteration = 0

# Audio settings
SAMPLE_RATE = 48000  # Must be one of webrtcvad's supported rates
BUFFER_DURATION = 0.016  # 16ms per buffer
BUFFER_SIZE = int(SAMPLE_RATE * BUFFER_DURATION)

# Constants
SPEED_OF_SOUND = 343  # Speed of sound in air in meters/second

# Find the loopback input device
def get_blackhole_input_device():
    devices = sd.query_devices()
    for idx, device in enumerate(devices):
        if 'CABLE' in device['name'] and device['max_input_channels'] > 0:
            return idx
    print("BlackHole input device not found.")
    sys.exit(1)

# Function to process and retrieve audio parameters every 16ms
def get_audio_parameters(buffer, sample_rate):
    # Calculate amplitude as the maximum absolute value in the buffer
    amplitude = np.max(np.abs(buffer))

    # Compute FFT to find dominant frequency
    fft_values = fft(buffer)
    fft_magnitudes = np.abs(fft_values)
    freqs = fftfreq(len(buffer), 1 / sample_rate)

    # Get the frequency with the highest magnitude in the positive frequency range
    idx = np.argmax(fft_magnitudes[:len(fft_magnitudes) // 2])
    frequency = abs(freqs[idx])  # Dominant frequency

    # Calculate time period and wavelength
    time_period = 1 / frequency if frequency != 0 else float('inf')
    wavelength = SPEED_OF_SOUND / frequency if frequency != 0 else float('inf')

    # Debug statement for buffer shape
    print(f"Buffer shape before reshape: {buffer}")

    # Ensure buffer is a 2D array for librosa functions
    buffer_reshaped = buffer.reshape(1, -1)

    # Calculate bpm from avg difference between peaks in this sample
    bpm = lr.beat.tempo(buffer_reshaped)[0]

    # Debug statement for reshaped buffer
    print(f"Buffer shape after reshape: {buffer_reshaped}")

    rms = lr.feature.rms(y=buffer_reshaped)
    spectral_centroid = lr.feature.spectral_centroid(y=buffer_reshaped, sr=sample_rate)
    zero_crossing_rate = lr.feature.zero_crossing_rate(y=buffer_reshaped)

    return amplitude, frequency, time_period, wavelength, bpm, rms, spectral_centroid, zero_crossing_rate

# Callback function for audio processing - runs every 16ms on buffer
def audio_callback(indata, frames, time, status):
    global current_iteration

    if status:
        print(f"Status: {status}", file=sys.stderr)

    # Convert to 16-bit PCM
    buffer_int16 = (indata[:, 0] * 32767).astype(np.int16)

    # Calculate audio parameters
    amplitude, frequency, time_period, wavelength, bpm, rms, spectral_centroid, zero_crossing_rate = get_audio_parameters(indata[:, 0], SAMPLE_RATE)

    this_run_data["amplitude"][current_iteration] = amplitude
    this_run_data["frequency"][current_iteration] = frequency
    this_run_data["time_period"][current_iteration] = time_period
    this_run_data["wavelength"][current_iteration] = wavelength
    this_run_data["tempo"][current_iteration] = bpm
    this_run_data["rms"][current_iteration] = rms
    this_run_data["spectral_centroid"][current_iteration] = spectral_centroid
    this_run_data["zero_crossing_rate"][current_iteration] = zero_crossing_rate

    current_iteration += 1

def main():
    input_device = get_blackhole_input_device()
    print(f"Using input device: {sd.query_devices(input_device)['name']}")

    while current_iteration != 64:
        try:
            with sd.InputStream(
                    samplerate=SAMPLE_RATE,
                    device=input_device,
                    channels=1,
                    dtype='float32',
                    blocksize=BUFFER_SIZE,
                    callback=audio_callback
            ):
                while True:
                    sd.sleep(int(BUFFER_DURATION * 1000))
        except KeyboardInterrupt:
            print("\nStopped by user.")
            break
        except Exception as e:
            print(f"Error: {e}")

    # Ensure DataFrame creation after loop ends
    df = pd.DataFrame.from_dict(this_run_data)
    print(df)

if __name__ == "__main__":
    main()
