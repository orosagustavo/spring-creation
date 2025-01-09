import numpy as np
import subrotinas.subrotinasPython as SRP
from particle import Particle

# Defining time steps
DT = 0.01
TOTAL_STEPS = 1000

# Defining start random position
X_INTERVAL = (-5, 5)
Y_INTERVAL = (-5, 5)

# Defining spring constants
NATURAL_DISTANCE = 4 # cm
K_SPRING = 10 # Nm-1

# Defining particles
# particle1 = SRP.generate_particle(X_INTERVAL, Y_INTERVAL, type_mol=1)
particle1 = Particle(-3.000, 0.000, type_mol=3)
mass1 = 1

# particle2 = SRP.generate_particle(X_INTERVAL, Y_INTERVAL, type_mol=3)
particle2 = Particle(0.000000000, 0.00000000)
mass2 = 1

particles = [particle1, particle2]
masses = [mass1, mass2]

# Create an output folder
output_file = SRP.create_output_folder()
open(output_file, "w").close() # Clean output file

# Start loop
for step in range(TOTAL_STEPS):
    fx, fy = SRP.spring_force(particle1, particle2,
                              NATURAL_DISTANCE, K_SPRING)
    forces = [(fx, fy), (-fx, -fy)]

    SRP.update_positions(particles, forces, DT, masses)

    SRP.xyz_file_writer(output_file, particles, step)

    print(step)

