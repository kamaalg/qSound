from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
from intensityto import intensity_to_color, intensity_to_wave_color
import math
import time

sr, sg, sb = 0.0, 0.0, 0.0

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

def draw_wave(start_time, amplitude, frequency, phase, intensity):
    r, g, b = intensity_to_wave_color(intensity)
    points = generate_wave_points(start_time, amplitude, frequency, phase)
    glColor3f(r, g, b)  # Blue color for the wave
    glLineWidth(4)
    glBegin(GL_LINE_STRIP)
    for x, y in points:
        glVertex2f(x, y)
    glEnd()

