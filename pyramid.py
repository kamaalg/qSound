from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

# Global variables for base color
base_r, base_g, base_b = 1.0, 0.5, 0.2  # Initial color
back_r, back_g, back_b = 0.0, 0.0, 0.0  # Initial background
angle = 0  # Global variable for rotation angle


def init():
    global back_r, back_g, back_b
    glEnable(GL_DEPTH_TEST)  # Enable depth testing for 3D rendering
    glClearColor(back_r, back_g, back_b, 1.0)  # Set background color to black

    # Set up the projection matrix
    glMatrixMode(GL_PROJECTION)  # Switch to the projection matrix
    glLoadIdentity()  # Reset any previous transformations
    gluPerspective(45.0, 1.777, 0.1, 50.0)  # Set up a perspective projection
    # Parameters: field of view (45 degrees), aspect ratio (1.777 for 16:9), near and far planes (0.1 to 50.0)

    # Switch to the modelview matrix
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()  # Reset any previous transformations


def draw():
    global angle
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)  # Clear color and depth buffers
    glLoadIdentity()  # Reset transformations
    gluLookAt(0.0, 1.0, 3.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)  # Set the camera view

    # Apply rotation around the y-axis
    glRotatef(angle, 0.0, 1.0, 0.0)  # Rotate the pyramid

    # Draw a pyramid
    glBegin(GL_TRIANGLES)

    # Front face (bright)
    glColor3f(base_r, base_g, base_b)  # Base color
    glVertex3f(0.0, 1.0, 0.0)  # Apex
    glVertex3f(-1.0, -1.0, 1.0)  # Bottom-left
    glVertex3f(1.0, -1.0, 1.0)  # Bottom-right

    # Right face (slightly darker)
    glColor3f(base_r * 0.7, base_g * 0.7, base_b * 0.7)  # Darken color by multiplying
    glVertex3f(0.0, 1.0, 0.0)  # Apex
    glVertex3f(1.0, -1.0, 1.0)  # Bottom-left
    glVertex3f(1.0, -1.0, -1.0)  # Bottom-right

    # Back face (darker)
    glColor3f(base_r * 0.5, base_g * 0.5, base_b * 0.5)  # Further darken color
    glVertex3f(0.0, 1.0, 0.0)  # Apex
    glVertex3f(1.0, -1.0, -1.0)  # Bottom-left
    glVertex3f(-1.0, -1.0, -1.0)  # Bottom-right

    # Left face (even darker)
    glColor3f(base_r * 0.3, base_g * 0.3, base_b * 0.3)  # Further darken color
    glVertex3f(0.0, 1.0, 0.0)  # Apex
    glVertex3f(-1.0, -1.0, -1.0)  # Bottom-left
    glVertex3f(-1.0, -1.0, 1.0)  # Bottom-right

    glEnd()

    # Draw the base (optional)
    glBegin(GL_QUADS)
    glColor3f(base_r * 0.6, base_g * 0.6, base_b * 0.6)  # Slightly darker than the front
    glVertex3f(-1.0, -1.0, 1.0)  # Bottom-left
    glVertex3f(1.0, -1.0, 1.0)  # Bottom-right
    glVertex3f(1.0, -1.0, -1.0)  # Top-right
    glVertex3f(-1.0, -1.0, -1.0)  # Top-left
    glEnd()

    glFlush()
    glutSwapBuffers()  # Swap the front and back buffers


def update(value):
    global angle
    angle += 1  # Increment the angle for rotation
    if angle > 360:  # Reset the angle to avoid overflow
        angle -= 360
    global base_r, base_g, base_b

    # Update the color (simple cycling logic)
    base_r += 0.01
    base_g += 0.02
    base_b += 0.03
    # Keep the color values in the range [0, 1]
    if base_r > 1.0: base_r -= 1.0
    if base_g > 1.0: base_g -= 1.0
    if base_b > 1.0: base_b -= 1.0

    glutPostRedisplay()  # Request a redraw
    glutTimerFunc(16, update, 0)  # Call this function again after 16 milliseconds


def main():
    glutInit()
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)  # Enable double buffering and depth testing
    glutInitWindowSize(1920, 1080)
    glutCreateWindow("3D Pyramid with Rotation")
    glutDisplayFunc(draw)
    init()
    glutTimerFunc(16, update, 0)
    glutMainLoop()


main()
