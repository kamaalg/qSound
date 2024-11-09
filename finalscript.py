from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
from intensityto import *
from particle import Particle, create_particle
from realwave import draw_wave
import random
import math
import time

# global intensity
intensity = -1.0

# global wave values
amplitude, frequency, phase = 0.0, 0.0, 0.0
start_time = time.time()

# List to store particles
particles = []


def spawn():
    global particles, intensity
    # Create a new particle occasionally for demonstration
    if len(particles) < 100000:
        particles.append(create_particle(intensity))
    intensity += .001
    if intensity >= 1.0:
        intensity = -1.0


def update_wave():
    global amplitude, frequency, phase
    if amplitude > 1.0: amplitude = 0
    if phase > 1*math.pi: phase = -1*math.pi
    if frequency > 1.0: frequency = 0
    phase += .001
    amplitude += .001
    frequency += .001


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
    global particles
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)  # Clear color and depth buffers
    glLoadIdentity()
    gluLookAt(0.0, 0.0, 5.0,  # Eye position
              0.0, 0.0, 0.0,  # Look-at position
              0.0, 1.0, 0.0)  # Up vector

    # Update and draw particles, remove dead ones
    particles = [particle for particle in particles if particle.update()]
    for particle in particles:
        particle.draw()
    draw_wave(start_time, amplitude, frequency, phase, intensity)
    glFlush()
    glutSwapBuffers()


def update(value):
    # get audio output, output the current now, shift array

    # take averages for input layer

    # process intensity in NN, update intensity global val

    # quantum math

    spawn()  # create particles

    update_wave()  # update waves

    glutPostRedisplay()
    glutTimerFunc(16, update, 0)

def main():
    glutInit()
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
    glutInitWindowSize(1920, 1080)
    glutCreateWindow("Particle System with Lifetime")
    init()
    glutDisplayFunc(draw)
    glutTimerFunc(16, update, 0)
    glutMainLoop()

main()
