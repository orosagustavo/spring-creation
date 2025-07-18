import numpy as np
import subrotinas.subrotinasPython_multiple as SRP
from tqdm import tqdm
from particle import Particle

# Definindo passos de tempo
DT = 0.001
TOTAL_STEPS = 10000

# Definindo o número de partículas
NUM_PARTICLES = 120

# Definindo constantes da mola
NATURAL_DISTANCE = 7.0 # Distância natural da mola
K_SPRING = 2.0        # Constante da mola
B_DAMP = 0.1           # Constante de amortecimento

# Distância mínima entre partículas para aplicar força repulsiva de "colisão"
MIN_PARTICLE_DISTANCE = 1.0 

# Inicializando partículas em uma linha
particles = []
masses = []
initial_x = 0.0
mass_per_particle = 1.0 # Massa para cada partícula

for i in range(NUM_PARTICLES):
    particles.append(Particle(initial_x + i * NATURAL_DISTANCE, 0.0, vx=0, vy=0, type_mol=1))
    masses.append(mass_per_particle)

# -- Adicionar perturbação inicial --
particles[0].vx = 10.0
particles[0].vy = 0.0

# Listas para armazenar dados para o gráfico
time_points = []
relative_distances_history = []

# Cria uma pasta de saída e limpa o arquivo de saída
output_file = SRP.create_output_folder()
open(output_file, "w").close() # Limpa o arquivo de saída

print(f"Simulando {NUM_PARTICLES} partículas conectadas em linha...")

# -- Inicia o loop de simulação --
for step in tqdm(range(TOTAL_STEPS)):
    # Escreve o estado inicial no arquivo .xyz
    if step == 0:
        SRP.xyz_file_writer(output_file, particles, step)
    
    # Calcula as acelerações de todas as partículas
    SRP.calculate_chain_accelerations(particles, NATURAL_DISTANCE, K_SPRING, B_DAMP, masses, MIN_PARTICLE_DISTANCE)

    # Calcula as novas velocidades de todas as partículas usando as acelerações
    SRP.velocities_damped(particles, DT, NATURAL_DISTANCE, K_SPRING, B_DAMP, masses)
    
    # Atualiza as posições de todas as partículas
    SRP.update_damped_positions(particles, DT)

    # Escreve o estado atual no arquivo .xyz
    SRP.xyz_file_writer(output_file, particles, step)

    # Coleta dados para o gráfico da posição relativa (entre a primeira e a segunda partícula)
    if NUM_PARTICLES >= 2:
        dx_rel = particles[1].x - particles[0].x
        dy_rel = particles[1].y - particles[0].y
        current_relative_distance = np.sqrt(dx_rel**2 + dy_rel**2)
        relative_distances_history.append(current_relative_distance)
        time_points.append(step * DT)

print("Simulação concluída!")

# Geração do gráfico da posição relativa
if NUM_PARTICLES >= 2:
    SRP.plot_relative_distance(NATURAL_DISTANCE, time_points, relative_distances_history, "posicao_relativa_p1_p2.png")
    print("Gráfico da posição relativa salvo em 'graphs/posicao_relativa_p1_p2.png'")
else:
    print("Não há partículas suficientes para gerar o gráfico de posição relativa (mínimo 2 partículas).")

