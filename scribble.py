from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import random
import numpy as np

points = []  # List to store generated points

def init():
    glEnable(GL_DEPTH_TEST)  # Enable depth testing for 3D rendering
    glClearColor(0.0, 0.0, 0.0, 1.0)  # Set background color to black

    # Set up the projection matrix
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45.0, 1.0, 0.1, 50.0)  # Set up a perspective projection
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

def generate_new_point():
    # Generate a new random point in 3D space within a limited range
    x = random.uniform(-1.0, 1.0)
    y = random.uniform(-1.0, 1.0)
    z = random.uniform(-1.0, 1.0)
    points.append((x, y, z))
    if len(points) > 20:  # Limit the number of points to keep it manageable
        points.pop(0)

def catmull_rom_spline(p0, p1, p2, p3, num_points=20):
    """
    Generates points on a Catmull-Rom spline given four control points.
    p0, p1, p2, p3 are 3D points (tuples of x, y, z).
    num_points specifies the number of points to generate between p1 and p2.
    """
    curve = []
    for i in range(num_points):
        t = i / (num_points - 1)  # Normalized parameter t ranges from 0 to 1
        t2 = t * t
        t3 = t2 * t

        # Catmull-Rom basis matrix
        x = 0.5 * ((2 * p1[0]) +
                   (-p0[0] + p2[0]) * t +
                   (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t2 +
                   (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t3)
        y = 0.5 * ((2 * p1[1]) +
                   (-p0[1] + p2[1]) * t +
                   (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t2 +
                   (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t3)
        z = 0.5 * ((2 * p1[2]) +
                   (-p0[2] + p2[2]) * t +
                   (2 * p0[2] - 5 * p1[2] + 4 * p2[2] - p3[2]) * t2 +
                   (-p0[2] + 3 * p1[2] - 3 * p2[2] + p3[2]) * t3)
        curve.append((x, y, z))
    return curve

def draw_curve():
    if len(points) < 4:
        return  # Need at least 4 points for Catmull-Rom spline

    glColor3f(0.0, 1.0, 0.0)  # Green color for the curve
    glBegin(GL_LINE_STRIP)
    for i in range(1, len(points) - 2):
        p0 = points[i - 1]
        p1 = points[i]
        p2 = points[i + 1]
        p3 = points[i + 2]
        curve = catmull_rom_spline(p0, p1, p2, p3)
        for point in curve:
            glVertex3f(point[0], point[1], point[2])  # Add points to the curve
    glEnd()

def draw():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)  # Clear color and depth buffers
    glLoadIdentity()
    gluLookAt(0.0, 0.0, 5.0,  # Eye position
              0.0, 0.0, 0.0,  # Look-at position
              0.0, 1.0, 0.0)  # Up vector

    draw_curve()  # Draw the smooth curve connecting the points

    glFlush()
    glutSwapBuffers()

def update(value):
    generate_new_point()  # Generate a new point every update
    glutPostRedisplay()
    glutTimerFunc(1000, update, 0)  # Call this function again after 1000 milliseconds (1 second)

def main():
    glutInit()
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
    glutInitWindowSize(800, 600)
    glutCreateWindow("3D Smooth Curve with Catmull-Rom Spline")
    init()
    glutDisplayFunc(draw)
    glutTimerFunc(1000, update, 0)
    glutMainLoop()

main()
