import tensorflow as tf
import os
from datetime import datetime
import matplotlib.pyplot as plt

# Funções de I/O e Plotagem (mantidas do original, mas adaptadas para receber tensores)
def create_output_folder():
    """Cria a pasta 'data' para armazenar os arquivos de saída e retorna o caminho do arquivo."""
    output_folder = "data"
    os.makedirs(output_folder, exist_ok=True)
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_file = os.path.join(output_folder, f"particles_tf_{current_time}.xyz")
    return output_file

def create_graphs_folder():
    """Cria a pasta 'graphs' para armazenar os gráficos de saída e retorna o caminho da pasta."""
    graphs_folder = "graphs"
    os.makedirs(graphs_folder, exist_ok=True)
    return graphs_folder

def xyz_file_writer(file, particles_pos, particles_vel, step):
    """Escreve as posições e tipos das partículas em um arquivo .xyz."""
    with open(file, "a") as f:
        total_particles = particles_pos.shape[0]
        f.write(f"{total_particles}\n")
        f.write(f"Step {step}\n")
        # Converte tensores para numpy para iteração e escrita
        pos_np = particles_pos.numpy()
        vel_np = particles_vel.numpy()
        for i in range(total_particles):
            # Assumindo tipo 'P' para todas as partículas, como no script original
            f.write(f"P {pos_np[i, 0]:.4f} {pos_np[i, 1]:.4f} {vel_np[i, 0]:.4f} {vel_np[i, 1]:.4f}\n")

def plot_relative_distance(natural_distance, time_steps, relative_distances, filename="relative_distance_tf.png"):
    """Gera e salva um gráfico da distância relativa ao longo do tempo."""
    graphs_folder = create_graphs_folder()
    filepath = os.path.join(graphs_folder, filename)
    plt.figure(figsize=(10, 6))
    plt.plot(time_steps, relative_distances, label='Distância Relativa entre P1 e P2 (TF)')
    plt.axhline(y=natural_distance, color='r', linestyle='--', label='Distância Natural da Mola')
    plt.title('Distância Relativa (TensorFlow)')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Distância Relativa')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()

# --- Funções de Simulação com TensorFlow ---

@tf.function
def tf_spring_force(pos1, pos2, natural_distance, k, min_distance):
    """
    Calcula a força da mola entre dois conjuntos de partículas de forma vetorizada.
    A força calculada é a que 'pos2' exerce sobre 'pos1'.
    """
    epsilon = 1e-9
    relative_vectors = pos2 - pos1
    distances = tf.norm(relative_vectors, axis=1)
    
    # Adiciona epsilon para evitar divisão por zero
    distances_safe = distances + epsilon
    
    # Vetores de direção normalizados
    unit_vectors = relative_vectors / distances_safe[:, tf.newaxis]
    
    # Magnitude da Força (Lei de Hooke)
    delta_distance = distances - natural_distance
    force_magnitude = -k * delta_distance
    
    # Força repulsiva de curto alcance para evitar colapso
    # tf.where(condition, x, y) é o equivalente a 'if' vetorizado
    repulsive_force = 10.0 * (min_distance / distances_safe)**12
    force_magnitude = tf.where(distances < min_distance, force_magnitude + repulsive_force, force_magnitude)
    
    # Força final como vetor (componentes x, y)
    # tf.newaxis expande a dimensão para permitir a multiplicação elemento a elemento
    force_vectors = unit_vectors * force_magnitude[:, tf.newaxis]
    
    return force_vectors

@tf.function
def tf_calculate_accelerations(positions, velocities, masses, natural_distance, k, b, min_distance):
    """
    Calcula as acelerações para todas as partículas em uma cadeia de forma vetorizada.
    """
    num_particles = positions.shape[0]
    
    # Inicializa o tensor de forças totais com zeros
    total_forces = tf.zeros_like(positions)

    # --- Forças dos vizinhos à esquerda (para partículas 1 até N-1) ---
    # pos_current são as partículas de 1 a N-1
    # pos_left são as partículas de 0 a N-2
    force_left = tf_spring_force(
        positions[1:], positions[:-1], natural_distance, k, min_distance
    )
    # Adiciona a força às partículas de 1 a N-1
    total_forces = tf.tensor_scatter_nd_add(total_forces, tf.range(1, num_particles)[:, tf.newaxis], force_left)
    # Adiciona a força de reação (3ª Lei de Newton) às partículas de 0 a N-2
    total_forces = tf.tensor_scatter_nd_add(total_forces, tf.range(0, num_particles - 1)[:, tf.newaxis], -force_left)
    
    # --- Forças dos vizinhos à direita (para partículas 0 até N-2) ---
    # pos_current são as partículas de 0 a N-2
    # pos_right são as partículas de 1 a N-1
    force_right = tf_spring_force(
        positions[:-1], positions[1:], natural_distance, k, min_distance
    )
    # Adiciona a força às partículas de 0 a N-2
    total_forces = tf.tensor_scatter_nd_add(total_forces, tf.range(0, num_particles - 1)[:, tf.newaxis], force_right)
    # Adiciona a força de reação (3ª Lei de Newton) às partículas de 1 a N-1
    total_forces = tf.tensor_scatter_nd_add(total_forces, tf.range(1, num_particles)[:, tf.newaxis], -force_right)

    # Força de Amortecimento
    damping_force = b * velocities
    
    # Força Total Resultante
    # No seu código original, a força de mola e a de amortecimento são invertidas
    # F_total = -(F_mola + F_amortecimento)
    # Vou manter essa convenção.
    net_force = -(total_forces + damping_force)
    
    # Aceleração (F=ma => a=F/m)
    # masses[:, tf.newaxis] garante a divisão correta das componentes (x,y)
    accelerations = net_force / masses[:, tf.newaxis]
    
    return accelerations

@tf.function
def tf_integration_step(positions, velocities, accelerations, masses, dt, natural_distance, k, b, min_distance):
    """
    Executa um passo de integração de Verlet completo e retorna os novos estados.
    Esta função é decorada com @tf.function para compilação em um grafo otimizado.
    """
    # 1. Armazena a aceleração antiga (a(t))
    old_accelerations = accelerations
    
    # 2. Atualiza a velocidade para v(t+dt) - conforme seu código original
    # v(t+dt) = v(t) + a(t) * dt
    new_velocities = velocities + old_accelerations * dt

    # 3. Atualiza a posição para r(t+dt) - conforme seu código original
    # r(t+dt) = r(t) + v(t+dt)*dt + 0.5*a(t)*dt^2
    new_positions = positions + new_velocities * dt + 0.5 * old_accelerations * dt**2
    
    # 4. Calcula a nova aceleração a(t+dt) usando as novas posições e velocidades
    new_accelerations = tf_calculate_accelerations(new_positions, new_velocities, masses, natural_distance, k, b, min_distance)
    
    return new_positions, new_velocities, new_accelerations