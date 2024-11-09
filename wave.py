from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import math
import time

amplitude, frequency, phase = 0.0, 0.0, 0.0
start_time = time.time()

def init():
    glEnable(GL_DEPTH_TEST)  # Enable depth testing for 3D rendering
    glClearColor(0.0, 0.0, 0.0, 1.0)  # Set background color to black

    # Set up the projection matrix
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluOrtho2D(-1.0, 1.0, -1.0, 1.0)  # 2D orthographic projection
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

def generate_wave_points():
    global phase
    points = []
    num_points = 100  # Number of points to draw the wave
    current_time = time.time() - start_time  # Elapsed time for animation

    for i in range(num_points):
        x = -1.0 + 2.0 * i / (num_points - 1)  # x-coordinate ranges from -1 to 1
        y = amplitude * math.sin(2 * math.pi * frequency * (x + current_time) + phase)  # Sine wave formula
        points.append((x, y))
    return points

def draw_wave():
    points = generate_wave_points()
    glColor3f(1.0, 1.0, 1.0)  # Blue color for the wave
    glBegin(GL_LINE_STRIP)
    for x, y in points:
        glVertex2f(x, y)
    glEnd()

def draw():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)  # Clear color and depth buffers
    glLoadIdentity()

    draw_wave()  # Draw the wave

    glFlush()
    glutSwapBuffers()

def update(value):
    global amplitude, frequency, phase
    if amplitude > 1.0: amplitude = 0
    if phase > 1*math.pi: phase = -1*math.pi
    if frequency > 1.0: frequency = 0
    phase += .001
    amplitude += .001
    frequency += .001
    glutPostRedisplay()
    glutTimerFunc(16, update, 0)  # Call this function again after 16 milliseconds (60 FPS)

def main():
    glutInit()
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
    glutInitWindowSize(1920, 1080)
    glutCreateWindow("Moving Wave with Amplitude, Frequency, and Phase")
    init()
    glutDisplayFunc(draw)
    glutTimerFunc(16, update, 0)
    glutMainLoop()

main()
