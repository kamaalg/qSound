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

# Initialize quantum registers for each group
qr_plus = QuantumRegister(3, 'plus')
qr_minus = QuantumRegister(3, 'minus')
qr_zero = QuantumRegister(3, 'zero')
qr_one = QuantumRegister(3, 'one')

# Create a quantum circuit combining all qubits
qc = QuantumCircuit(qr_plus, qr_minus, qr_zero, qr_one)

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

# Simulate sound wave attributes (for demonstration)
# In a real application, these would come from real-time audio processing
amplitude = 0.8          # Normalized amplitude (0 to 1)
phase = np.pi / 4        # Phase in radians
frequency = 0.5          # Normalized frequency (0 to 1)

# Apply transformations to each group
apply_sound_transformations(qc, qr_plus, amplitude, phase, frequency)
apply_sound_transformations(qc, qr_minus, amplitude, phase, frequency)
apply_sound_transformations(qc, qr_zero, amplitude, phase, frequency)
apply_sound_transformations(qc, qr_one, amplitude, phase, frequency)

# Entangle qubits between groups to simulate interference
entangle_groups(qc, qr_plus, qr_minus)
entangle_groups(qc, qr_zero, qr_one)

# Get the statevector
statevector = Statevector.from_instruction(qc)
num_total_qubits = qc.num_qubits

# Function to get the Bloch vector of a qubit
def get_qubit_bloch_vector(statevector, num_qubits, qubit_index):
    # Convert statevector to density matrix
    rho = DensityMatrix(statevector)
    # Trace out all other qubits except the one at qubit_index
    qubits_to_trace_out = [i for i in range(num_qubits) if i != qubit_index]
    reduced_rho = partial_trace(rho, qubits_to_trace_out)  # Use partial_trace function
    
    # Access the matrix data
    reduced_rho_matrix = reduced_rho.data

    # Calculate Bloch vector components for a single-qubit density matrix
    bloch_x = 2 * np.real(reduced_rho_matrix[0, 1])
    bloch_y = 2 * np.imag(reduced_rho_matrix[0, 1])
    bloch_z = np.real(reduced_rho_matrix[0, 0] - reduced_rho_matrix[1, 1])
    
    return np.array([bloch_x, bloch_y, bloch_z])


# Collect coordinates for all qubits
# Collect coordinates for all qubits
coordinates = []
labels = []
group_names = ['plus', 'minus', 'zero', 'one']
group_qubits = [qr_plus, qr_minus, qr_zero, qr_one]

for group_name, qr in zip(group_names, group_qubits):
    for qubit in qr:
        # Use find_bit to locate the qubit index in the circuit
        qubit_index = qc.find_bit(qubit).index
        bloch_vector = get_qubit_bloch_vector(statevector, num_total_qubits, qubit_index)
        x, y, z = bloch_vector.real  # Ensure values are real numbers
        coordinates.append((x, y, z))
        labels.append(f"{group_name}_{qubit_index}")
# Visualize the qubits on the Bloch sphere
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

# Draw the sphere
u, v = np.mgrid[0:2*np.pi:200j, 0:np.pi:100j]
xs = np.cos(u)*np.sin(v)
ys = np.sin(u)*np.sin(v)
zs = np.cos(v)
ax.plot_wireframe(xs, ys, zs, color="lightgray", linewidth=0.5, alpha=0.2)

# Plot qubit vectors
for idx, (x, y, z) in enumerate(coordinates):
    ax.quiver(0, 0, 0, x, y, z, length=1.0, color='b', arrow_length_ratio=0.1)
    ax.text(x, y, z, labels[idx], fontsize=9)

# Set plot limits and labels
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Qubit States on the Bloch Sphere')

plt.show()
