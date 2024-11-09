import sounddevice as sd
import numpy as np
import webrtcvad
import sys
from scipy.fft import fft, fftfreq

# Audio settings
SAMPLE_RATE = 48000  # Must be one of webrtcvad's supported rates
BUFFER_DURATION = 0.02  # 20ms per buffer
BUFFER_SIZE = int(SAMPLE_RATE * BUFFER_DURATION)

# Constants
SPEED_OF_SOUND = 343  # Speed of sound in air in meters/second

# Initialize Voice Activity Detector
vad = webrtcvad.Vad()
vad.set_mode(2)


# Find the BlackHole input device
def get_blackhole_input_device():
    devices = sd.query_devices()
    for idx, device in enumerate(devices):
        if 'BlackHole' in device['name'] and device['max_input_channels'] > 0:
            return idx
    print("BlackHole input device not found.")
    sys.exit(1)


# Function to process and retrieve audio parameters
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

    return amplitude, frequency, time_period, wavelength


# Callback function for audio processing
def audio_callback(indata, frames, time, status):
    if status:
        print(f"Status: {status}", file=sys.stderr)

    # Check if voice is detected in the audio buffer
    buffer_int16 = (indata[:, 0] * 32767).astype(np.int16)  # Convert to 16-bit PCM
    if vad.is_speech(buffer_int16.tobytes(), SAMPLE_RATE):
        # Calculate audio parameters
        amplitude, frequency, time_period, wavelength = get_audio_parameters(indata[:, 0], SAMPLE_RATE)

        # Print the results
        print(f"Amplitude: {amplitude:.2f}")
        print(f"Frequency: {frequency:.2f} Hz")
        print(f"Time Period: {time_period:.4f} s")
        print(f"Wavelength: {wavelength:.4f} m")
    else:
        print("No voice detected")


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
            print("Listening... Press Ctrl+C to stop.")
            while True:
                sd.sleep(int(BUFFER_DURATION * 1000))
    except KeyboardInterrupt:
        print("\nStopped by user.")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
