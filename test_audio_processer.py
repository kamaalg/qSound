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

    reshaped_buffer = buffer.reshape((1,-1))

    bpm = lr.feature.tempo(y=reshaped_buffer)

    rms = lr.feature.rms(y=reshaped_buffer)[0][0][0]
    
    spectral_centroid = lr.feature.spectral_centroid(y=reshaped_buffer, sr=sample_rate)[0][0][0]

    zero_crossing_rate = lr.feature.zero_crossing_rate(y=reshaped_buffer)[0][0][0]

    return amplitude, frequency, time_period, wavelength, bpm, rms, spectral_centroid, zero_crossing_rate

# Callback function for audio processing - runs every 16ms on buffer
def audio_callback(indata, frames, time, status):

    current_iteration = 0

    while current_iteration < 63:

        if status:
            print(f"Status: {status}", file=sys.stderr)

        # Convert to 16-bit PCM
        buffer_int16 = (indata[:, 0] * 32767).astype(np.int16)

        # Calculate audio parameters
        this_run_data["amplitude"][current_iteration],
        this_run_data["frequency"][current_iteration],
        this_run_data["time_period"][current_iteration],
        this_run_data["wavelength"][current_iteration],
        this_run_data["tempo"][current_iteration],
        this_run_data["rms"][current_iteration],
        this_run_data["spectral_centroid"][current_iteration], 
        this_run_data["zero_crossing_rate"][current_iteration] = get_audio_parameters(indata[:, 0], SAMPLE_RATE)

        current_iteration += 1

def main():
    input_device = get_blackhole_input_device()
    print(f"Using input device: {sd.query_devices(input_device)['name']}")
    
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
    except Exception as e:
        print(f"Error: {e}")

    # Ensure DataFrame creation after loop ends
    df = pd.DataFrame.from_dict(this_run_data)
    print(df)

if __name__ == "__main__":
    main()
