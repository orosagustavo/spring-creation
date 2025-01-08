import numpy as np
import subrotinas.subrotinasPython as SRP

# Defining start random position
X_INTERVAL = (-5, 5)
Y_INTERVAL = (-5,5)

NATURAL_DISTANCE = 3 # cm
K_SPRING = 5 # Kg*s^-1

# Particle test
particle1 = SRP.generate_particle(X_INTERVAL, Y_INTERVAL)
particle2 = SRP.generate_particle(X_INTERVAL, Y_INTERVAL)


