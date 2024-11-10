import math

def calculate_song_intensity(amplitude, frequency, phase, spectral_centroid, bpm, rms_energy):
    # Normalize inputs to 0-1 range where applicable
    normalized_amplitude = max(0, min(1.0, amplitude))  # Clamp between 0 and 1
    normalized_frequency = (min(max(frequency, 20), 20000) - 20) / (20000 - 20)  # Scale frequency
    normalized_spectral_centroid = (min(max(spectral_centroid, 200), 8000) - 200) / (
                8000 - 200)  # Scale spectral centroid
    normalized_bpm = min(max(bpm, 40), 300) / 300  # Normalize bpm to a range from 40 to 300
    normalized_rms_energy = max(0, min(1.0, rms_energy))  # Assume rms_energy is between 0 and 1

    # Modulation factor based on phase (using cosine to oscillate intensity)
    phase_modulation = 0.5 * (1 + math.cos(phase))  # Ranges from 0 to 1

    # Calculate intensity with adjusted weights for the additional factors
    intensity = (
            0.3 * normalized_amplitude +
            0.2 * normalized_frequency +
            0.2 * normalized_spectral_centroid +
            0.15 * normalized_bpm +
            0.1 * normalized_rms_energy +
            0.05 * phase_modulation
    )

    # Map intensity from [0, 1] to [-1, 1]
    intensity = 6 * (intensity - 0.5) + 0.7

    # Clamp intensity to [-1.0, 1.0]
    return max(-1.0, min(1.0, intensity))


# Example usage
amplitude = 0.8
frequency = 1000
phase = math.pi / 4
spectral_centroid = 1500
bpm = 120
rms_energy = 0.7

intensity = calculate_song_intensity(amplitude, frequency, phase, spectral_centroid, bpm, rms_energy)
print(f"The intensity of the song is: {intensity}")
