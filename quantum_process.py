# quantum_process.py

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Statevector, DensityMatrix, partial_trace

def generate_qubit_data(input_amplitude, input_phase, input_frequency):
    """
    Generates qubit data for OpenGL rendering based on input amplitude, phase, and frequency.
    
    Parameters:
    - input_amplitude: float, the input amplitude (0 to 1)
    - input_phase: float, the input phase in radians
    - input_frequency: float, the normalized input frequency (0 to 1)
    
    Returns:
    - qubit_data: list of dictionaries, each containing data for a qubit
      Each dictionary contains:
        - 'label': str, label of the qubit
        - 'amplitude': float, amplitude for visualization
        - 'frequency': float, frequency for visualization
        - 'phase': float, phase for visualization
        - 'color': tuple, RGB color of the qubit
    """
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

    # Define sound wave attributes for each group based on the given inputs
    # Vary these slightly for each group to create diversity, i.e. shift amplitude superpositions, rest fairly arbitrary
    group_sound_attributes = {
        'plus': {'amplitude': input_amplitude * 1.0, 'phase': input_phase, 'frequency': input_frequency},
        'minus': {'amplitude': input_amplitude * 0.9, 'phase': input_phase + 0.1, 'frequency': input_frequency + 0.05},
        'zero': {'amplitude': input_amplitude * 0.8, 'phase': input_phase + 0.2, 'frequency': input_frequency + 0.1},
        'one': {'amplitude': input_amplitude * 0.7, 'phase': input_phase + 0.3, 'frequency': input_frequency + 0.15},
        'y_plus': {'amplitude': input_amplitude * 0.6, 'phase': input_phase + 0.4, 'frequency': input_frequency + 0.2},
        'y_minus': {'amplitude': input_amplitude * 0.5, 'phase': input_phase + 0.5, 'frequency': input_frequency + 0.25}
    }

    # Apply transformations to each group with sound attributes
    for group_name, qr in zip(group_sound_attributes.keys(), [qr_plus, qr_minus, qr_zero, qr_one, qr_y_plus, qr_y_minus]):
        amplitude = group_sound_attributes[group_name]['amplitude']
        phase = group_sound_attributes[group_name]['phase']
        frequency = group_sound_attributes[group_name]['frequency']
        apply_sound_transformations(qc, qr, amplitude, phase, frequency)

    # Entangle qubits between groups to observe interesting behavior
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

    return qubit_data

def apply_sound_transformations(qc, qubits, amplitude, phase, frequency):
    """
    Applies quantum transformations to qubits based on sound attributes.
    
    Parameters:
    - qc: QuantumCircuit, the quantum circuit to apply transformations to
    - qubits: QuantumRegister, the qubits to transform
    - amplitude: float, amplitude for rotation (0 to 1)
    - phase: float, phase shift in radians
    - frequency: float, frequency factor for rotation
    """
    # Map amplitude to rotation angle (0 to pi)
    theta = amplitude * np.pi

    # Map frequency to rotation angle
    freq_factor = frequency * np.pi

    for qubit in qubits:
        # Apply rotation around Y-axis
        qc.ry(theta, qubit)
        # Apply rotation around X-axis influenced by frequency
        qc.rx(freq_factor, qubit)
        # Apply phase shift around Z-axis
        qc.rz(phase, qubit)

def entangle_groups(qc, control_qubits, target_qubits):
    """
    Entangles pairs of qubits between two groups using CNOT gates.
    
    Parameters:
    - qc: QuantumCircuit, the quantum circuit to apply entanglement to
    - control_qubits: QuantumRegister, the control qubits
    - target_qubits: QuantumRegister, the target qubits
    """
    for c_qubit, t_qubit in zip(control_qubits, target_qubits):
        qc.cx(c_qubit, t_qubit)

def get_qubit_density_matrix(statevector, qubit_index, num_total_qubits):
    """
    Obtains the reduced density matrix of a single qubit from the statevector.
    
    Parameters:
    - statevector: Statevector, the statevector of the quantum system
    - qubit_index: int, the index of the qubit to extract
    - num_total_qubits: int, total number of qubits in the system
    
    Returns:
    - reduced_rho: DensityMatrix, the reduced density matrix of the qubit
    """
    # Convert statevector to density matrix
    rho = DensityMatrix(statevector)
    # List of qubits to trace out (all except qubit_index)
    qubits_to_trace_out = [i for i in range(num_total_qubits) if i != qubit_index]
    # Reduce the density matrix to the qubit of interest
    reduced_rho = partial_trace(rho, qubits_to_trace_out)
    return reduced_rho
