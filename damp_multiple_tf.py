import random
import tensorflow as tf
import numpy as np
import subrotinas.subrotinas_tf as SRTF 
import subrotinas.subrotinas_hybrid as SRH
from tqdm import tqdm
from particle import Particle 

# --- Configurações da Simulação ---
DT = 0.01
TOTAL_STEPS = 7000
NUM_PARTICLES = 100
NATURAL_DISTANCE = 5.0
# K_SPRINGS = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0001]
K_SPRING = 0.1
B_DAMP = 0.1
MIN_PARTICLE_DISTANCE = 1.0

# --- Inicialização do Sistema via Objetos ---
#Crie a lista de objetos Particle
particles = []
initial_x = 0.0
for i in range(NUM_PARTICLES):
    if i == 0:
        particles.append(Particle(-3, 5, type_mol=1))
    # random_number = random.uniform(0, 2)
    particles.append(Particle(initial_x + i *  NATURAL_DISTANCE, 0, type_mol=1))

# Adicionar perturbação inicial (modificando o objeto diretamente)
particles[0].vx = - 15.0
particles[0].vy = 25

# 2. Crie os tensores do TensorFlow a partir da lista de objetos.
positions_tf, velocities_tf, accelerations_tf, masses_tf = SRH.initialize_tensors_from_objects(particles)

# Listas para o gráfico
time_points = []
relative_distances_history = []

# Setup do arquivo de saída
output_file = SRH.create_output_folder()
open(output_file, "w").close()

print(f"Simulando {NUM_PARTICLES} partículas com a abordagem Híbrida (OO + TF)...")

# --- Loop de Simulação Híbrido ---
for step in tqdm(range(TOTAL_STEPS)):

    SRH.xyz_file_writer(output_file, particles, step)
    
    new_pos, new_vel, new_acc = SRTF.tf_integration_step(
        positions_tf, velocities_tf, accelerations_tf, masses_tf, DT,
        NATURAL_DISTANCE, K_SPRING, B_DAMP, MIN_PARTICLE_DISTANCE
    )
    
    # Escreve o estado atual no arquivo .xyz usando a lista de objetos.
    # Atualize os tensores principais com os novos resultados.
    positions_tf.assign(new_pos)
    velocities_tf.assign(new_vel)
    accelerations_tf.assign(new_acc)

    # if step == 2500:
    #     velocities_tf[0].assign([50, 15])

    # Sincronização: Atualize os objetos Python com os novos dados dos tensores.
    SRH.sync_tensors_to_objects(particles, positions_tf, velocities_tf, accelerations_tf)

    # Coleta de dados para o gráfico
    if NUM_PARTICLES >= 2:
        p1 = particles[5]
        p2 = particles[6]
        dx_rel = p2.x - p1.x
        dy_rel = p2.y - p1.y
        current_relative_distance = np.sqrt(dx_rel**2 + dy_rel**2)
        relative_distances_history.append(current_relative_distance)
        time_points.append(step * DT)

# Geração do gráfico
if NUM_PARTICLES >= 2:
    filename = f'relative_distance_hybrid.png'
    SRH.plot_relative_distance(NATURAL_DISTANCE, time_points, relative_distances_history, filename)
    print(f"Gráfico salvo em 'graphs/relative_distance_hybrid.png'")

print("Simulação Híbrida concluída!")

