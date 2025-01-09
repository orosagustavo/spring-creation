import subrotinas.subrotinasFortran as SRF
from particle import Particle

NATURAL_DISTANCE = 4

particle1 = Particle(3, 0)
particle2 = Particle(1, 2)

particles = [particle1, particle2]

dx, dy = SRF.subrotinas.relative_vector(
    particle1.x, particle1.y,
    particle2.x, particle2.y
)

fx, fy = SRF.subrotina.spring_force(dx, dy, NATURAL_DISTANCE, K)

vx = [particle.vx for particle in particles]
vy = [particle.vy for particle in particles]

ax, ay = SRF.subrotina.damped_string_acc()