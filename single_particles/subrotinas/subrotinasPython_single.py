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
def generate_particle(x_range, y_range, type_mol, vx=0, vy=0):
    x = random.uniform(*x_range)
    y = random.uniform(*y_range)

    return Particle(x, y, vx, vy, type_mol)

# Calculate spring force between 2 particles
def spring_force(particle1, particle2, natural_distance, k):
    # Relative vector
    dx, dy = particle2.relative_position(particle1)
    particles_distance = np.sqrt(dx**2 + dy**2)
    delta_distance = particles_distance - natural_distance
    
    if particles_distance == 0:
        # Evita divisão por zero
        return 0, 0

    # Calculate force
    force = -k * delta_distance # Hook Law
    fx = force * (dx / particles_distance) # Normalized directional vector
    fy = force * (dy / particles_distance)

    return fx, fy

# Function to calculate and update positions
def update_positions(particles, forces, dt, masses):
    for i, particle in enumerate(particles):
        fx, fy = forces[i]
        
        # Calculate acceleration
        ax, ay = fx / masses[i], fy / masses[i]

        # Update velicity
        particle.vx += ax * dt
        particle.vy += ay * dt

        # Update position
        particle.x += particle.vx * dt
        particle.y += particle.vy * dt 

def damped_string_acc(particles, natural_distance, k, b, masses):
    # Spring force calc
    fx, fy = spring_force(particles[0], particles[1], natural_distance, k)

    # Damping forces
    f_damps = [(b * particle.vx, b * particle.vy) for particle in particles]

    # Loop to update accelerations
    for i, particle in enumerate(particles):
            # Accelerations
            particle.ax = (fx - f_damps[i][0]) / masses[i] if i == 0 else (-fx - f_damps[i][0]) / masses[i]
            particle.ay = (fy - f_damps[i][1]) / masses[i] if i == 0 else (-fy - f_damps[i][1]) / masses[i]

def update_damped_positions(particles, dt):
    # Calculating positions
    for particle in particles:
        particle.x += particle.vx * dt + 0.5 * particle.ax * dt**2
        particle.y += particle.vy * dt + 0.5 * particle.ay * dt**2

def velocities_damped(particles, dt, natural_distance, k, b, masses):
    # Storaging old accelerations
    old_accs = [(particle.ax, particle.ay) for particle in particles]
    
    # Calulating new accelerations
    damped_string_acc(particles, natural_distance, k, b, masses)
    
    # Loop to calculate velocities
    for i, particle in enumerate(particles):
        particle.vx += 0.5 * (particle.ax + old_accs[i][0]) * dt
        particle.vy += 0.5 * (particle.ay + old_accs[i][1]) * dt

    