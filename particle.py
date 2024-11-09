from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
from intensityto import *

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


# Function to create a new particle
def create_particle(intensity):
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
