
import numpy as np
from quantum_process import generate_qubit_data

# Input params
input_amplitude = 0.9        
input_phase = np.pi / 4     
input_frequency = 0.7       

qubit_data = generate_qubit_data(input_amplitude, input_phase, input_frequency)

for qubit in qubit_data:

    label = qubit['label']
    amplitude = qubit['amplitude']
    frequency = qubit['frequency']
    phase = qubit['phase']
    color = qubit['color']
    print(qubit)    

    # Render qubit in OpenGL
