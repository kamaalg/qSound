# Import necessary libraries
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Statevector, Pauli
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

# Function to get the Bloch vector of a qubit without constructing the full density matrix
def get_qubit_bloch_vector(statevector, qubit_index, num_qubits):
    # Define Pauli operators acting on the qubit_index
    paulis = {'X': Pauli('I' * qubit_index + 'X' + 'I' * (num_qubits - qubit_index - 1)),
              'Y': Pauli('I' * qubit_index + 'Y' + 'I' * (num_qubits - qubit_index - 1)),
              'Z': Pauli('I' * qubit_index + 'Z' + 'I' * (num_qubits - qubit_index - 1))}

    # Compute expectation values
    bloch_x = statevector.expectation_value(paulis['X']).real
    bloch_y = statevector.expectation_value(paulis['Y']).real
    bloch_z = statevector.expectation_value(paulis['Z']).real

    return np.array([bloch_x, bloch_y, bloch_z])

# Initialize quantum registers for each group (2 qubits each)
qr_plus = QuantumRegister(2, 'plus')
qr_minus = QuantumRegister(2, 'minus')
qr_zero = QuantumRegister(2, 'zero')
qr_one = QuantumRegister(2, 'one')
qr_y_plus = QuantumRegister(2, 'y_plus')
qr_y_minus = QuantumRegister(2, 'y_minus')

# Create a quantum circuit combining all qubits
qc = QuantumCircuit(qr_plus, qr_minus, qr_zero, qr_one, qr_y_plus, qr_y_minus)

# Initialize qubits
# Group 1: |+> state
for qubit in qr_plus:
    qc.h(qubit)  # Hadamard gate transforms |0> to |+>

# Group 2: |-> state
for qubit in qr_minus:
    qc.h(qubit)
    qc.z(qubit)  # Z gate changes |+> to |->

# Group 3: |0> state (already initialized by default)

# Group 4: |1> state
for qubit in qr_one:
    qc.x(qubit)  # X gate flips |0> to |1>

# Group 5: |i+> state (Y basis positive eigenstate)
for qubit in qr_y_plus:
    qc.s(qubit)  # S gate (phase gate)
    qc.h(qubit)

# Group 6: |i-> state (Y basis negative eigenstate)
for qubit in qr_y_minus:
    qc.sdg(qubit)  # S† gate (inverse phase gate)
    qc.h(qubit)

# Simulate sound wave attributes (for demonstration)
amplitude = 0.8          # Normalized amplitude (0 to 1)
phase = np.pi / 4        # Phase in radians
frequency = 0.5          # Normalized frequency (0 to 1)

# Apply transformations to each group
apply_sound_transformations(qc, qr_plus, amplitude, phase, frequency)
apply_sound_transformations(qc, qr_minus, amplitude, phase, frequency)
apply_sound_transformations(qc, qr_zero, amplitude, phase, frequency)
apply_sound_transformations(qc, qr_one, amplitude, phase, frequency)
apply_sound_transformations(qc, qr_y_plus, amplitude, phase, frequency)
apply_sound_transformations(qc, qr_y_minus, amplitude, phase, frequency)

# Entangle qubits between groups to simulate interference
entangle_groups(qc, qr_plus, qr_minus)
entangle_groups(qc, qr_zero, qr_one)
entangle_groups(qc, qr_y_plus, qr_y_minus)

# Get the statevector
statevector = Statevector.from_instruction(qc)
num_total_qubits = qc.num_qubits

# Collect data for all qubits
coordinates = []
labels = []
colors = []
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

# Prepare a dictionary to store data for each entangled set
entangled_data = {'Entangled Set 1': [], 'Entangled Set 2': [], 'Entangled Set 3': []}

for group_name, qr in zip(group_names, group_qubits):
    for qubit in qr:
        qubit_index = qc.find_bit(qubit).index
        bloch_vector = get_qubit_bloch_vector(statevector, qubit_index, num_total_qubits)
        x, y, z = bloch_vector.real
        labels.append(f"{group_name}_{qubit_index}")
        colors.append(group_colors[group_name])  # Assign color based on initial group

        # Calculate Bloch sphere angles
        theta = np.arccos(z)  # θ ranges from 0 to π
        phi = np.arctan2(y, x)  # φ ranges from -π to π

        # Map to wave parameters
        amplitude_wave = np.sin(theta / 2)  # Amplitude is sin(θ/2)
        phase_wave = phi  # Phase is φ

        # Map phase_wave to frequency (e.g., map φ from [-π, π] to a frequency range)
        # For simplicity, we'll fix the frequency and use phase_wave as the phase
        frequency_wave = 0.5  # Fixed normalized frequency for all qubits

        # Store data in the appropriate entangled set
        if group_name in ['plus', 'minus']:
            entangled_set_label = 'Entangled Set 1'
        elif group_name in ['zero', 'one']:
            entangled_set_label = 'Entangled Set 2'
        elif group_name in ['y_plus', 'y_minus']:
            entangled_set_label = 'Entangled Set 3'

        entangled_data[entangled_set_label].append({
            'label': f"{group_name}_{qubit_index}",
            'amplitude': amplitude_wave,
            'frequency': frequency_wave,
            'phase': phase_wave,
            'color': group_colors[group_name]
        })

        print(f"Qubit {labels[-1]}: Amplitude = {amplitude_wave:.4f}, Phase = {phase_wave:.4f} rad")

# Now you can send 'entangled_data' to your OpenGL renderer or process it further

# Generate and plot wave for each entangled set separately
# Define frequency range for mapping (if varying frequencies)
f_base = 440    # Base frequency in Hz (A4 note)
sampling_rate = 44100  # Standard audio sampling rate
duration = 1.0         # Duration in seconds
t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)

# Iterate over each entangled set
for entangled_set_label, qubit_data_list in entangled_data.items():
    # Initialize the composite wave for this entangled set
    composite_wave = np.zeros_like(t)

    # Generate and sum waves for qubits in this entangled set
    for data in qubit_data_list:
        amplitude_wave = data['amplitude']
        frequency_wave = data['frequency']
        phase_wave = data['phase']

        # Map normalized frequency to actual frequency in Hz
        frequency_hz = f_base * frequency_wave  # Since frequency_wave is fixed at 0.5, frequency_hz = f_base * 0.5

        # Generate the wave for this qubit
        wave = amplitude_wave * np.sin(2 * np.pi * frequency_hz * t + phase_wave)
        composite_wave += wave

    # Normalize the composite wave
    composite_wave /= len(qubit_data_list)

    # Plot the composite wave for this entangled set
    plt.figure(figsize=(12, 6))
    plt.plot(t, composite_wave)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title(f'Waveform from {entangled_set_label}')
    plt.xlim(0, 0.01)  # Zoom in to the first 10 ms for better visibility
    plt.show()
