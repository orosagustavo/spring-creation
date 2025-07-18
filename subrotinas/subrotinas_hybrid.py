import tensorflow as tf
import os
from datetime import datetime
import matplotlib.pyplot as plt

# --- Funções de Sincronização Objeto <-> Tensor ---

def initialize_tensors_from_objects(particles):
    """
    Extrai o estado inicial de uma lista de objetos Particle e cria os tensores do TensorFlow.
    Retorna tensores tf.Variable para posições, velocidades e acelerações, e um tf.constant para massas.
    """
    # Extrai os dados da lista de objetos para listas Python
    positions_list = [[p.x, p.y] for p in particles]
    velocities_list = [[p.vx, p.vy] for p in particles]
    accelerations_list = [[p.ax, p.ay] for p in particles]
    # Assumindo massa 1.0 para todos, como no script original.
    masses_list = [1.0] * len(particles)

    # Converte as listas Python para tensores do TensorFlow
    positions_tf = tf.Variable(positions_list, dtype=tf.float32)
    velocities_tf = tf.Variable(velocities_list, dtype=tf.float32)
    accelerations_tf = tf.Variable(accelerations_list, dtype=tf.float32)
    masses_tf = tf.constant(masses_list, dtype=tf.float32)

    return positions_tf, velocities_tf, accelerations_tf, masses_tf

def sync_tensors_to_objects(particles, positions_tf, velocities_tf, accelerations_tf):
    """
    Atualiza os atributos dos objetos na lista 'particles' com os dados dos tensores.
    Esta é a ponte do "motor" TensorFlow de volta para a sua estrutura de objetos.
    """
    # Converte tensores para NumPy arrays para uma iteração mais fácil e eficiente
    pos_np = positions_tf.numpy()
    vel_np = velocities_tf.numpy()
    acc_np = accelerations_tf.numpy()

    # Itera sobre a lista de partículas e atualiza cada objeto
    for i, p in enumerate(particles):
        p.x = pos_np[i, 0]
        p.y = pos_np[i, 1]
        
        p.vx = vel_np[i, 0]
        p.vy = vel_np[i, 1]
        
        p.ax = acc_np[i, 0]
        p.ay = acc_np[i, 1]

# --- Funções de I/O e Plotagem ---

def create_output_folder(output_name = ''):
    """Cria a pasta 'data' para armazenar os arquivos de saída e retorna o caminho do arquivo."""
    output_folder = "data"
    os.makedirs(output_folder, exist_ok=True)
    if output_name != '':
        return os.path.join(output_folder, f'{output_name}.xyz')
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_file = os.path.join(output_folder, f"particles_hybrid_{current_time}.xyz")
    return output_file

def xyz_file_writer(file, particles, step):
    """
    Escreve as posições e tipos das partículas em um arquivo .xyz usando a lista de objetos.
    """
    with open(file, "a") as f:
        total_particles = len(particles)
        f.write(f"{total_particles}\n")
        f.write(f"Step {step}\n")
        # Itera sobre a lista de objetos Particle e salva seus dados
        for p in particles:
            f.write(f"{p.type_mol} {p.x:.4f} {p.y:.4f} {p.vx:.4f} {p.vy:.4f} \n")

def create_graphs_folder():
    """Cria a pasta 'graphs' para armazenar os gráficos de saída e retorna o caminho da pasta."""
    graphs_folder = "graphs"
    os.makedirs(graphs_folder, exist_ok=True)
    return graphs_folder

def plot_relative_distance(natural_distance, time_steps, relative_distances, filename="relative_distance_hybrid.png"):
    """Gera e salva um gráfico da distância relativa ao longo do tempo."""
    graphs_folder = create_graphs_folder()
    filepath = os.path.join(graphs_folder, filename)
    plt.figure(figsize=(10, 6))
    plt.plot(time_steps, relative_distances, label='Distância Relativa (Híbrido)')
    plt.axhline(y=natural_distance, color='r', linestyle='--', label='Distância Natural')
    plt.title('Distância Relativa (Abordagem Híbrida OO + TF)')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Distância Relativa')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()
