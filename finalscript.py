from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
from intensityto import *
from particle import Particle, create_particle
from quantum_process import generate_qubit_data
from realwave import draw_wave
import random
import math
import time
import threading
from files_for_nn.nn import final_nn
import numpy as np
from intensity import calculate_song_intensity
# import correct audio handler for platform
import platform
system = platform.system()
print(f"Launching {system} audio handler")
if system == "Darwin":
    from audio_processing_pyaudio import AudioHandler
elif system == "Windows":
    # there are a couple of minor tweaks in this file I think
    # I don't want to port them back to Darwin and break things
    from audio_processing_pyaudio_win import AudioHandler

# global intensity
intensity = -1.0
features = []
intensities = []


#global
qubits = []
qubit_datas = []
qubit_live= []
counter = 0


# global wave values
amplitude, frequency, phase, spectral_centroid = 0.0, 0.0, 0.0, 0.0
rms, bpm = 0.0, 0.0
amp, freq, pha = 1, 1, 1
start_time = time.time()

# List to store particles
particles = []


def spawn():
    global particles, intensity
    # Create a new particle occasionally for demonstration
    if len(particles) < 100000:
        particles.append(create_particle(intensity))
        if intensity > 0.3:
            particles.append(create_particle(intensity))
        if intensity > 0.6:
            particles.append(create_particle(intensity))
        if intensity > 0.8:
            particles.append(create_particle(intensity))

def update_wave():
    global amplitude, frequency, phase, amp, freq, pha

    # if abs(amplitude) > 1.0:
    #     amp = -1.0*amp
    #
    # if abs(phase) > 1*math.pi:
    #     pha = -1.0*pha
    # if abs(frequency) > 1.0:
    #     freq = -1.0*freq
    # phase += .001*pha
    # amplitude += .001*amp
    # frequency += .001*freq


def init():
    glEnable(GL_DEPTH_TEST)  # Enable depth testing for 3D rendering
    glClearColor(0.0, 0.0, 0.0, 1.0)  # Set background color to black

    # Set up the projection matrix
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45.0, 1.7777, 0.1, 50.0)  # Set up a perspective projection
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()


def draw():
    global particles, qubits, counter
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)  # Clear color and depth buffers
    glLoadIdentity()
    gluLookAt(0.0, 0.0, 5.0,  # Eye position
              0.0, 0.0, 0.0,  # Look-at position
              0.0, 1.0, 0.0)  # Up vector

    # Update and draw particles, remove dead ones
    particles = [particle for particle in particles if particle.update()]
    for particle in particles:
        particle.draw()

    for index, (qubit, data, live) in enumerate(zip(qubits, qubit_datas, qubit_live)):
        # Smoothly transition 'live' values towards 'qubit' values
        transition_speed = 1.0 / 120  # ~12 frames to complete transition (200ms at 60 FPS)

        live['amplitude'] += (qubit['amplitude'] - live['amplitude']) * transition_speed
        temp = (qubit['frequency'] - live['frequency'])/(10)
        if temp > 1.0:
            temp = 1.0

        if temp < -1.0:
            temp = -1.0

        #print(temp)
        live['frequency'] += temp /500
        # print("Live Freq")
        # print(live['frequency'])
        # print("Q Freq")
        # print(qubit['frequency'])
        live['phase'] += (qubit['phase'] - live['phase']) * transition_speed

        # Optional: Slight damping to stabilize values near the target
        # live['amplitude'] *= 0.99
        # live['frequency'] *= 0.99
        # live['phase'] *= 0.99
        #live['frequency']*live['amplitude']/4
        tfreq = abs(( live['frequency']) / 4)
        draw_wave(start_time, live['amplitude'], live['frequency'], live['phase'], intensity)

    glFlush()
    glutSwapBuffers()


def update(value):
    global counter, intensity, amplitude, frequency, phase, spectral_centroid, rms, bpm
    # get audio output, output the current now, shift array

    # take averages for input layer

    # process intensity in NN, update intensity global val

    if len(features) > 0:
        #intensity = -1*calculate_song_intensity(amplitude, frequency, phase, spectral_centroid)
        tintensity = final_nn(features)
        print("INTENSE SET")
        print(tintensity)
        tintensity = tintensity*12+.7
        #tintensity = calculate_song_intensity(amplitude, frequency, phase, spectral_centroid, rms, bpm)
        print(tintensity)
        if(len(intensities)>20):
            intensities.pop(0)
        intensities.append(tintensity)
        #print(intensities)
        #print(features)
        intensity = np.mean(intensities)
        #print(intensity)

    # quantum math

    spawn()  # create particles

    update_wave()  # update waves
    counter += 1

    glutPostRedisplay()
    glutTimerFunc(16, update, 0)


def qubit_thread():
    global qubits, amplitude, frequency, phase, qubit_datas, qubit_live, counter, bpm
    while True:
        print("BPM2: ", bpm)
        bpm_map = (bpm - 1200) / 100
        if bpm_map < 0:
            bpm_map = 0.2

        qubits_temp = generate_qubit_data((amplitude**2)*3 * bpm_map, frequency*2 * bpm_map, phase*2*bpm_map)

        if(len(qubits) != 0):
            #qubit_live = qubits
            qubit_datas = qubits
        qubits = qubits_temp
        counter = 0
        if(len(qubit_datas) == 0):
            qubit_datas = qubits
            qubit_live = qubits
        time.sleep(0.01)  # Update qubits every 100 milliseconds



def audio_thread():
    global qubits, amplitude, frequency, phase, qubit_datas, qubit_live, counter, features, spectral_centroid, rms, bpm
    audio = AudioHandler()
    audio.start()
    while audio.stream.is_active():
        amplitude = audio.amplitude
        frequency = audio.frequency
        phase = audio.phase
        spectral_centroid = audio.spectral_centroid
        rms = audio.rms
        bpm = audio.bpm
        prev_features = features
        features = audio.features

        # print("INTENSITY")
        # print(intensity)
        #print(phase)
        # print("Prev")
        # print(prev_features)
        # print("Feat")
        # print(features)
        time.sleep(0.16)
    audio.stop()
    time.sleep(0.16)  # Update qubits every 100 milliseconds


def main():
    glutInit()
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
    glutInitWindowSize(1920, 1080)
    glutCreateWindow("Particle System with Lifetime")
    init()
    glutDisplayFunc(draw)
    glutTimerFunc(16, update, 0)

    # Start the qubit data generation thread
    qubit_thread_instance = threading.Thread(target=qubit_thread)
    qubit_thread_instance.daemon = True  # Ensures the thread exits when the main program ends
    qubit_thread_instance.start()

    #Start the audio handler thread
    audio_thread_instance = threading.Thread(target=audio_thread)
    audio_thread_instance.daemon = True  # Ensures the thread exits when the main program ends
    audio_thread_instance.start()

    #Start the GLUT main loop
    glutMainLoop()

main()
