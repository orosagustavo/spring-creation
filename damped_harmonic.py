import numpy as np
import subrotinas.subrotinasPython as SRP
from particle import Particle

# Defining time steps
DT = 0.1
TOTAL_STEPS = 1000

# Defining start random position
X_INTERVAL = (-5, 5)
Y_INTERVAL = (-5, 5)

# Defining spring constants
NATURAL_DISTANCE = 9
K_SPRING = 20

B_DAMP = 0.1

# Defining particles
# particle1 = SRP.generate_particle(X_INTERVAL, Y_INTERVAL, type_mol=1)
particle1 = Particle(3.000, 0.000)
mass1 = 1

# particle2 = SRP.generate_particle(X_INTERVAL, Y_INTERVAL, type_mol=3)
particle2 = Particle(1.000, 2.000)
mass2 = 1

particles = [particle1, particle2]
masses = [mass1, mass2]

# Create an output folder
output_file = SRP.create_output_folder()
open(output_file, "w").close() # Clean output file

# Start loop
for step in range(TOTAL_STEPS):
    if step == 0:
        SRP.xyz_file_writer(output_file, particles, step)
    
    # Calulate accelerations
    SRP.damped_string_acc(particles, NATURAL_DISTANCE, K_SPRING, B_DAMP, masses)

    # Calculate velocities
    SRP.velocities_damped(particles, DT, NATURAL_DISTANCE, K_SPRING, B_DAMP, masses)
    
    # Update positions
    SRP.update_damped_positions(particles, DT)


    SRP.xyz_file_writer(output_file, particles, step)

    print(step)

