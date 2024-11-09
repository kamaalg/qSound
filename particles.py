from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import random

intensity = -1.0

def intensity_to_color(intensity):
    r, g, b = 0.0, 0.0, 0.0
    if(intensity<0):
        mr = 0.5+-1*intensity/10
        r = random.uniform(0.0, mr)

        mg = 0.5+(1+intensity)/2
        sg = (1+intensity)*4/5
        g = random.uniform(sg, mg)

        mb = .6 + -1*intensity*2/5
        sb = -1*intensity*4/5
        b = random.uniform(sb, mb)
    if(intensity>0):
        mr = intensity/2+0.5
        sr = intensity*4/5
        r = random.uniform(sr, mr)

        mg = 1 - intensity*4/5
        sg = 0.8-(intensity)*3/5
        g = random.uniform(sg, mg)

        mb = .6-(intensity/3)
        b = random.uniform(0.0, mb)
    return r, g, b


def intensity_to_speed(intensity):
    min = (intensity+1)/100
    max = -min
    return min, max



def intensity_to_radius(intensity):
    min = ((intensity-1)/1.5-.3)*4
    max = -min
    return min, max
# Particle structure with position, velocity, acceleration, and life span
class Particle:
    def __init__(self, x, y, z, dx, dy, dz, ax, ay, az, r, g, b):
        self.x = x  # Initial x position
        self.y = y  # Initial y position
        self.z = z  # Initial z position
        self.dx = dx  # Velocity vector x
        self.dy = dy  # Velocity vector y
        self.dz = dz  # Velocity vector z
        self.ax = ax  # Acceleration vector x
        self.ay = ay  # Acceleration vector y
        self.az = az  # Acceleration vector z
        self.r = r
        self.g = g
        self.b = b
        self.life = 2000  # Particle life span in milliseconds

    def update(self):
        # Decrease the life of the particle
        self.life -= 16  # Subtracting approximately 16 milliseconds (60 FPS)
        if self.life <= 0:
            return False  # Indicate that the particle should be removed

        # Update velocity using acceleration
        self.dx += self.ax
        self.dy += self.ay
        self.dz += self.az

        # Update position using velocity
        self.x += self.dx
        self.y += self.dy
        self.z += self.dz

        return True  # Indicate that the particle is still alive

    def draw(self):
        # Only draw the particle if it is alive
        glPushMatrix()
        glTranslatef(self.x, self.y, self.z)
        glColor3f(self.r, self.g, self.b)  # Orange color
        glutSolidSphere(0.02, 10, 10)  # Draw a small sphere to represent the particle
        glPopMatrix()

# List to store particles
particles = []

# Function to create a new particle
def create_particle():
    global intensity
    r, g, b = intensity_to_color(intensity)
    s_min, s_max = intensity_to_radius(intensity)
    d_min, d_max = intensity_to_speed(intensity)
    a_min, a_max = d_min/10, d_max/10
    return Particle(
        random.uniform(s_min, s_max), random.uniform(s_min, s_max), random.uniform(s_min/10, s_max/10),  # Initial position
        random.uniform(d_min, d_max), random.uniform(d_min, d_max), random.uniform(d_min, d_max),  # Initial velocity
        random.uniform(a_min, a_max), random.uniform(a_min, a_max), random.uniform(a_min, a_max),  # Acceleration
        r, g, b
    )

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

    glFlush()
    glutSwapBuffers()

def update(value):
    global particles, intensity
    # Create a new particle occasionally for demonstration
    if len(particles) < 100000:
        particles.append(create_particle())
    intensity += .001
    if intensity >= 1.0:
        intensity = -1.0
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
