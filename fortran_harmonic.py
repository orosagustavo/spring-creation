import subrotinas.subrotinasFortran as SRF
import subrotinas.subrotinasPython as SRP
from particle import Particle

NATURAL_DISTANCE = 4
K_SPRING = 20 
B_SPRING = 0.1

DT = 0.1
TOTAL_STEPS = 100

particle1 = Particle(3, 0)
mass1 = 1
particle2 = Particle(1, 2)
mass2 = 1

masses = [mass1, mass2]

particles = [particle1, particle2]




for step in range(TOTAL_STEPS):
    # Recebe os parâmetros antigos
    vx_0 = [particle.vx for particle in particles]
    vy_0 = [particle.vy for particle in particles]

    ax_0 = [particle.ax for particle in particles]
    ay_0 = [particle.ay for particle in particles]

    x_0 = [particle.x for particle in particles]
    y_0 = [particle.y for particle in particles]


    # Calcula os novos parâmetros
    dx, dy = SRF.subrotinas.relative_vector(
        particle1.x, particle1.y,
        particle2.x, particle2.y
    )
    fx, fy = SRF.subrotinas.spring_force(
        dx, dy, NATURAL_DISTANCE, K_SPRING
    )
    ax, ay = SRF.subrotinas.damped_string_acc(
        vx_0, vy_0, fx, fy, B_SPRING, masses, 2
    )

    vx, vy = SRF.subrotinas.velocities(
        ax, ay, ax_0, ay_0, vx_0, vy_0, DT, 2
    )

    # Atualiza os parâmetros
    for i, particle in enumerate(particles):
        particle.ax = ax[i]
        particle.ay = ay[i]

        particle.vx = vx[i]
        particle.vy = vy[i]

