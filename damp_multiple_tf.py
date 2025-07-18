import tensorflow as tf
import numpy as np
import subrotinas.subrotinas_tf as SRTF
from tqdm import tqdm

# --- Configurações da Simulação ---
# Definindo passos de tempo
DT = 0.001
TOTAL_STEPS = 10000

# Definindo o número de partículas
NUM_PARTICLES = 120

# Definindo constantes da mola
NATURAL_DISTANCE = 7.0
K_SPRING = 2.0
B_DAMP = 0.1

# Distância mínima para força repulsiva
MIN_PARTICLE_DISTANCE = 1.0

# --- Inicialização do Sistema com TensorFlow ---

# Posições iniciais: Partículas em linha no eixo X
initial_x = tf.range(NUM_PARTICLES, dtype=tf.float32) * NATURAL_DISTANCE
initial_y = tf.zeros(NUM_PARTICLES, dtype=tf.float32)
# `positions` é um tensor tf.Variable de shape (NUM_PARTICLES, 2)
positions = tf.Variable(tf.stack([initial_x, initial_y], axis=1))

# Velocidades iniciais (todas em repouso)
velocities = tf.Variable(tf.zeros((NUM_PARTICLES, 2), dtype=tf.float32))

# Massas das partículas
mass_per_particle = 1.0
masses = tf.constant([mass_per_particle] * NUM_PARTICLES, dtype=tf.float32)

# -- Adicionar perturbação inicial --
# Modifica a velocidade da primeira partícula
# Usa-se `tf.tensor_scatter_nd_update` para atualizar um elemento específico do tensor
indices = tf.constant([[0]]) # Índice da partícula 0
updates = tf.constant([[10.0, 0.0]]) # Nova velocidade [vx, vy]
velocities.assign(tf.tensor_scatter_nd_update(velocities, indices, updates))

# Aceleração inicial (começa em zero, será calculada no primeiro passo)
accelerations = tf.Variable(tf.zeros((NUM_PARTICLES, 2), dtype=tf.float32))

# Listas para armazenar dados para o gráfico
time_points = []
relative_distances_history = []

# Cria arquivo de saída
output_file = SRTF.create_output_folder()
open(output_file, "w").close() # Limpa o arquivo

print(f"Simulando {NUM_PARTICLES} partículas com TensorFlow...")

# --- Loop de Simulação ---
for step in tqdm(range(TOTAL_STEPS)):
    # Escreve o estado no arquivo .xyz no primeiro e nos passos seguintes
    SRTF.xyz_file_writer(output_file, positions, velocities, step)
    
    # Executa um passo de integração completo
    # A função `tf_integration_step` é compilada pelo TensorFlow para alta performance
    new_pos, new_vel, new_acc = SRTF.tf_integration_step(
        positions, velocities, accelerations, masses, DT, 
        NATURAL_DISTANCE, K_SPRING, B_DAMP, MIN_PARTICLE_DISTANCE
    )
    
    # Atualiza o estado das variáveis da simulação
    positions.assign(new_pos)
    velocities.assign(new_vel)
    accelerations.assign(new_acc)

    # Coleta dados para o gráfico
    if NUM_PARTICLES >= 2:
        # tf.norm calcula a distância euclidiana diretamente
        relative_vector = positions[1] - positions[0]
        current_relative_distance = tf.norm(relative_vector)
        # .numpy() converte o tensor de volta para um valor Python/NumPy
        relative_distances_history.append(current_relative_distance.numpy())
        time_points.append(step * DT)

print("Simulação com TensorFlow concluída!")

# Geração do gráfico da posição relativa
if NUM_PARTICLES >= 2:
    SRTF.plot_relative_distance(NATURAL_DISTANCE, time_points, relative_distances_history, "posicao_relativa_p1_p2_tf.png")
    print("Gráfico salvo em 'graphs/posicao_relativa_p1_p2_tf.png'")