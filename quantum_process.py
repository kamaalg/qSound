# Import necessary libraries
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Statevector, DensityMatrix, partial_trace
import matplotlib.pyplot as plt

# Function to apply sound transformations to qubits
def apply_sound_transformations(qc, qubits, amplitude, phase, frequency):
    # Map amplitude to rotation angle (0 to pi)
    theta = amplitude * np.pi

    # Map frequency to rotation speed (for simulation purposes)
    freq_factor = frequency * np.pi

    for qubit in qubits:
        # Apply rotation around Y-axis
        qc.ry(theta, qubit)
        # Apply rotation around X-axis influenced by frequency
        qc.rx(freq_factor, qubit)
        # Apply phase shift around Z-axis
        qc.rz(phase, qubit)

# Function to entangle groups of qubits
def entangle_groups(qc, control_qubits, target_qubits):
    for c_qubit, t_qubit in zip(control_qubits, target_qubits):
        qc.cx(c_qubit, t_qubit)

# Function to get the reduced density matrix of a qubit
def get_qubit_density_matrix(statevector, qubit_index, num_total_qubits):
    # Convert statevector to density matrix
    rho = DensityMatrix(statevector)
    # List of qubits to trace out (all except qubit_index)
    qubits_to_trace_out = [i for i in range(num_total_qubits) if i != qubit_index]
    # Reduce the density matrix to the qubit of interest
    reduced_rho = partial_trace(rho, qubits_to_trace_out)
    return reduced_rho

# Initialize quantum registers for each group (2 qubits each)
qr_plus = QuantumRegister(2, 'plus')
qr_minus = QuantumRegister(2, 'minus')
qr_zero = QuantumRegister(2, 'zero')
qr_one = QuantumRegister(2, 'one')
qr_y_plus = QuantumRegister(2, 'y_plus')
qr_y_minus = QuantumRegister(2, 'y_minus')

# Create a quantum circuit combining all qubits
qc = QuantumCircuit(qr_plus, qr_minus, qr_zero, qr_one, qr_y_plus, qr_y_minus)

# Initialize qubits in different basis states
# Group 1: |+> state
for qubit in qr_plus:
    qc.h(qubit)  # Hadamard gate transforms |0⟩ to |+⟩

# Group 2: |-> state
for qubit in qr_minus:
    qc.h(qubit)
    qc.z(qubit)  # Z gate changes |+⟩ to |−⟩

# Group 3: |0⟩ state (already initialized by default)

# Group 4: |1⟩ state
for qubit in qr_one:
    qc.x(qubit)  # X gate flips |0⟩ to |1⟩

# Group 5: |i+⟩ state (Y basis positive eigenstate)
for qubit in qr_y_plus:
    qc.s(qubit)  # S gate (phase gate)
    qc.h(qubit)

# Group 6: |i−⟩ state (Y basis negative eigenstate)
for qubit in qr_y_minus:
    qc.sdg(qubit)  # S† gate (inverse phase gate)
    qc.h(qubit)

# Assume we are given amplitude, phase, and frequency at a given point
# These values are provided as inputs to the script
# For example:
input_amplitude = 0.8        # Value between 0 and 1
input_phase = np.pi / 3      # Phase in radians
input_frequency = 0.6        # Normalized frequency between 0 and 1

# Define sound wave attributes for each group based on the given inputs
# We can vary these slightly for each group to create diversity
group_sound_attributes = {
    'plus': {'amplitude': input_amplitude * 1.0, 'phase': input_phase, 'frequency': input_frequency},
    'minus': {'amplitude': input_amplitude * 0.9, 'phase': input_phase + 0.1, 'frequency': input_frequency + 0.05},
    'zero': {'amplitude': input_amplitude * 0.8, 'phase': input_phase + 0.2, 'frequency': input_frequency + 0.1},
    'one': {'amplitude': input_amplitude * 0.7, 'phase': input_phase + 0.3, 'frequency': input_frequency + 0.15},
    'y_plus': {'amplitude': input_amplitude * 0.6, 'phase': input_phase + 0.4, 'frequency': input_frequency + 0.2},
    'y_minus': {'amplitude': input_amplitude * 0.5, 'phase': input_phase + 0.5, 'frequency': input_frequency + 0.25}
}

# Apply transformations to each group with sound attributes
apply_sound_transformations(qc, qr_plus, **group_sound_attributes['plus'])
apply_sound_transformations(qc, qr_minus, **group_sound_attributes['minus'])
apply_sound_transformations(qc, qr_zero, **group_sound_attributes['zero'])
apply_sound_transformations(qc, qr_one, **group_sound_attributes['one'])
apply_sound_transformations(qc, qr_y_plus, **group_sound_attributes['y_plus'])
apply_sound_transformations(qc, qr_y_minus, **group_sound_attributes['y_minus'])

# Entangle qubits between groups
entangle_groups(qc, qr_plus, qr_minus)
entangle_groups(qc, qr_zero, qr_one)
entangle_groups(qc, qr_y_plus, qr_y_minus)

# Get the statevector
statevector = Statevector.from_instruction(qc)
num_total_qubits = qc.num_qubits

# Collect data for all qubits
labels = []
group_names = ['plus', 'minus', 'zero', 'one', 'y_plus', 'y_minus']
group_qubits = [qr_plus, qr_minus, qr_zero, qr_one, qr_y_plus, qr_y_minus]
group_colors = {
    'plus': (1.0, 0.0, 0.0),      # Red
    'minus': (0.0, 1.0, 0.0),     # Green
    'zero': (0.0, 0.0, 1.0),      # Blue
    'one': (1.0, 1.0, 0.0),       # Yellow
    'y_plus': (1.0, 0.0, 1.0),    # Magenta
    'y_minus': (0.0, 1.0, 1.0)    # Cyan
}

# Prepare a list to store data for each qubit
qubit_data = []

for group_name, qr in zip(group_names, group_qubits):
    for qubit in qr:
        qubit_index = qc.find_bit(qubit).index
        labels.append(f"{group_name}_{qubit_index}")

        # Get the reduced density matrix of the qubit
        rho_qubit = get_qubit_density_matrix(statevector, qubit_index, num_total_qubits)
        rho_matrix = rho_qubit.data

        # Calculate probabilities of |0⟩ and |1⟩
        p0 = np.real(rho_matrix[0, 0])
        p1 = np.real(rho_matrix[1, 1])

        # Calculate amplitude as sqrt of probability of |1⟩, scaled by input amplitude
        input_amplitude_group = group_sound_attributes[group_name]['amplitude']
        amplitude_wave = input_amplitude_group * np.sqrt(p1)

        # Calculate phase as angle of the off-diagonal element
        phase_wave = np.angle(rho_matrix[0, 1])

        # Use the group's frequency for visualization
        frequency_wave = group_sound_attributes[group_name]['frequency']

        qubit_data.append({
            'label': f"{group_name}_{qubit_index}",
            'amplitude': amplitude_wave,
            'frequency': frequency_wave,
            'phase': phase_wave,
            'color': group_colors[group_name]
        })

        print(f"Qubit {labels[-1]}: Amplitude = {amplitude_wave:.4f}, "
              f"Frequency = {frequency_wave:.4f}, Phase = {phase_wave:.4f} rad")

# send 'qubit_data' to OpenGL renderer here, example following




# Generate and plot wave for all qubits
# Define frequency range for mapping (adjusted for higher frequencies)
f_min = 200    # Minimum frequency in Hz
f_max = 2000   # Maximum frequency in Hz  (Increased for higher frequencies)
sampling_rate = 44100  # Standard audio sampling rate
duration = 1.0         # Duration in seconds
t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)

# Generate and plot individual waves
for data in qubit_data:
    amplitude_wave = data['amplitude']
    frequency_wave = data['frequency']
    phase_wave = data['phase']

    # Map normalized frequency to actual frequency in Hz
    frequency_hz = f_min + frequency_wave * (f_max - f_min)

    # Generate the wave for this qubit
    wave = amplitude_wave * np.sin(2 * np.pi * frequency_hz * t + phase_wave)

    # Plot the wave for this qubit
    plt.figure(figsize=(12, 6))
    plt.plot(t, wave)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title(f'Waveform from Qubit {data["label"]}')
    plt.xlim(0, 0.01)  # Zoom in to the first 10 ms for better visibility
    plt.show()
