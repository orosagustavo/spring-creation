import os
import numpy as np
import random
from particle import Particle

# Create a .xyz file to visualize ()
def arquivo_xyz(file, particles, step):
    with open(file, "a") as f:
        total_particles = len(particles)
        f.write(f"{total_particles}\n")
        f.write(f"Step {step}\n")
        # Run all particles and save positions
        for p in particles.T:
            f.write(f"P {p[0]:.4f} {p[1]:.4f} 0.0\n")
    

# Generate a random particle
def generate_particle(x_range, y_range):
    x = random.uniform(*x_range)
    y = random.uniform(*y_range)

    return Particle(x,y)