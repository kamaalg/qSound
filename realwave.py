from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

import math
import time

def generate_wave_points(start_time, amplitude, frequency, phase):
    points = []
    num_points = 400  # Number of points to draw the wave
    current_time = time.time() - start_time  # Elapsed time for animation

    for i in range(num_points):
        # Adjust x to span the entire 1.77 aspect ratio frame
        x = -3.77 + 2 * 3.77 * i / (num_points - 1)  # x-coordinate ranges from -1.77 to 1.77
        y = amplitude * math.sin(2 * math.pi * frequency * (x + current_time) + phase)  # Sine wave formula
        points.append((x, y))
    return points

def draw_wave(start_time, amplitude, frequency, phase):
    points = generate_wave_points(start_time, amplitude, frequency, phase)
    glColor3f(1.0, 1.0, 1.0)  # Blue color for the wave
    glBegin(GL_LINE_STRIP)
    for x, y in points:
        glVertex2f(x, y)
    glEnd()

