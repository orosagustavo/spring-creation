import os
import numpy as np
import random
from datetime import datetime
from particle import Particle

def create_output_folder():
    output_folder = "data"
    os.makedirs(output_folder, exist_ok=True)
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_file = os.path.join(output_folder, f"particles_2D_{current_time}.xyz")
    return output_file

# Create a .xyz file to visualize ()    
def xyz_file_writer(file, particles, step):
    with open(file, "a") as f:
        total_particles = len(particles)
        f.write(f"{total_particles}\n")
        f.write(f"Step {step}\n")
        # Run all particles and save positions
        for p in particles:
            f.write(f"{p.type_mol} {p.x:.4f} {p.y:.4f} {p.vx:.4f} {p.vy:.4f} \n")
    

# Generate a random particle
def generate_particle(x_range, y_range, type_mol):
    x = random.uniform(*x_range)
    y = random.uniform(*y_range)

    return Particle(x, y, type_mol)

# Calculate spring force between 2 particles
def spring_force(particle1, particle2, natural_distance, k):
    # Relative vector
    dx, dy = particle1.relative_position(particle2)
    particles_distance = np.sqrt(dx**2 + dy**2)
    delta_distance = particles_distance - natural_distance

    # Calculate force
    force = -k * delta_distance # Hook Law
    fx = force * (dx / particles_distance) # Normalized directional vector
    fy = force * (dy / particles_distance)

    return fx, fy

# Function to calculate and update positions
def update_positions(particles, forces, dt, masses):
    for i, particle in enumerate(particles):
        fx, fy = forces[i]
        
        # Calculate aceleration
        ax, ay = fx / masses[i], fy / masses[i]

        # Update velicity
        particle.vx += ax * dt
        particle.vy += ay * dt

        # Update position
        particle.x += particle.vx * dt
        particle.y += particle.vy * dt 
