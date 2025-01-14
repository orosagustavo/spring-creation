import subrotinas.subrotinasFortran as SRF
import subrotinas.subrotinasPython as SRP
from particle import Particle

# Constantes da mola
NATURAL_DISTANCE = 9
K_SPRING = 20 
B_SPRING = 0.1

# Constantes de tempo
DT = 0.1
TOTAL_STEPS = 1000

# Inicialização de partículas
particle1 = Particle(3, 0)
mass1 = 1

particle2 = Particle(1, 2)
mass2 = 1

# Definição de listas
masses = [mass1, mass2]
particles = [particle1, particle2]

# Cria a pasta de saída
output_file = SRP.create_output_folder()

# Iteração dos frames da simulação
for step in range(TOTAL_STEPS):
    if step == 0:
        SRP.xyz_file_writer(output_file, particles, step)
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
    # Novas forças
    fx, fy = SRF.subrotinas.spring_force(
        dx, dy, NATURAL_DISTANCE, K_SPRING
    )
    # Novas acelerações
    ax, ay = SRF.subrotinas.damped_string_acc(
        vx_0, vy_0, fx, fy, B_SPRING, masses, 2
    )
    # Novas velocidades
    vx, vy = SRF.subrotinas.velocities(
        ax, ay, ax_0, ay_0, vx_0, vy_0, DT, 2
    )
    # Novas posições
    x, y = SRF.subrotinas.position(
        x_0, y_0, vx, vy, ax, ay, DT, 2
    )

    # Escreve o xyz
    print(step)

    # Atualiza os parâmetros
    for i, particle in enumerate(particles):
        particle.ax = ax[i]
        particle.ay = ay[i]

        particle.vx = vx[i]
        particle.vy = vy[i]

        particle.x = x[i]
        particle.y = y[i]
        
    SRP.xyz_file_writer(output_file, particles, step)

