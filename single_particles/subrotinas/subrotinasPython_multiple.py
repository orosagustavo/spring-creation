import os
import numpy as np
import random
import matplotlib.pyplot as plt
from datetime import datetime
from particle import Particle

def create_output_folder():
    """Cria a pasta 'data' para armazenar os arquivos de saída e retorna o caminho do arquivo."""
    output_folder = "data"
    os.makedirs(output_folder, exist_ok=True)
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_file = os.path.join(output_folder, f"particles_2D_{current_time}.xyz")
    return output_file

def create_graphs_folder():
    """Cria a pasta 'graphs' para armazenar os gráficos de saída e retorna o caminho da pasta."""
    graphs_folder = "graphs"
    os.makedirs(graphs_folder, exist_ok=True)
    return graphs_folder

def xyz_file_writer(file, particles, step):
    """Escreve as posições e tipos das partículas em um arquivo .xyz para visualização."""
    with open(file, "a") as f:
        total_particles = len(particles)
        f.write(f"{total_particles}\n")
        f.write(f"Step {step}\n")
        # Itera sobre todas as partículas e salva suas posições, velocidades e tipo
        for p in particles:
            f.write(f"{p.type_mol} {p.x:.4f} {p.y:.4f} {p.vx:.4f} {p.vy:.4f} \n")
    

def generate_particle(x_range, y_range, type_mol, vx=0, vy=0):
    """Gera uma partícula com posição aleatória dentro de um dado intervalo."""
    x = random.uniform(*x_range)
    y = random.uniform(*y_range)
    return Particle(x, y, vx, vy, type_mol)

def spring_force(particle1, particle2, natural_distance, k, min_distance):
    """
    Calcula a força da mola entre duas partículas.
    Retorna as componentes x e y da força que particle2 exerce sobre particle1.
    """
    # Vetor relativo de particle1 para particle2
    epsilon = 1e-9
    dx = particle2.x - particle1.x
    dy = particle2.y - particle1.y
    
    particles_distance = np.sqrt(dx**2 + dy**2)

    if particles_distance < epsilon: # Usa epsilon para verificar se a distância é muito pequena
        return 0.0, 0.0
    
    delta_distance = particles_distance - natural_distance
    
    # Calcula a magnitude da força (Lei de Hooke)
    force_magnitude = -k * delta_distance 

    # Adiciona uma força repulsiva de curto alcance se as partículas estiverem muito próximas
    if particles_distance < min_distance:
        # Aumenta a força repulsiva drasticamente quando a distância é menor que min_distance
        repulsive_force_magnitude = 10.0 * (min_distance / particles_distance)**12 
        # Garante que a força repulsiva seja sempre para afastar as partículas
        force_magnitude += repulsive_force_magnitude 
    
    # Calcula as componentes x e y da força
    fx = force_magnitude * (dx / particles_distance)
    fy = force_magnitude * (dy / particles_distance)

    return fx, fy

def calculate_chain_accelerations(particles, natural_distance, k, b, masses, min_distance):
    """
    Calcula e atualiza as acelerações para todas as partículas em uma cadeia linear.
    Cada partícula interage apenas com seus vizinhos imediatos.
    """
    num_particles = len(particles)

    # Zera as acelerações atuais para acumular as novas forças
    for p in particles:
        p.ax = 0.0
        p.ay = 0.0

    # Loop para calcular as forças de mola e amortecimento para cada partícula
    for i in range(num_particles):
        current_particle = particles[i]
        
        # Força da mola do vizinho da esquerda
        if i > 0:
            left_neighbor = particles[i-1]
            # Força que o vizinho da esquerda exerce sobre a partícula atual
            fx_left, fy_left = spring_force(current_particle, left_neighbor, natural_distance, k, min_distance)
            
            # Adiciona a força à partícula atual
            current_particle.ax += fx_left
            current_particle.ay += fy_left

            # Pela 3ª Lei de Newton, a força oposta atua no vizinho da esquerda
            left_neighbor.ax -= fx_left
            left_neighbor.ay -= fy_left

        # Força da mola do vizinho da direita
        if i < num_particles - 1:
            right_neighbor = particles[i+1]
            # Força que o vizinho da direita exerce sobre a partícula atual
            fx_right, fy_right = spring_force(current_particle, right_neighbor, natural_distance, k, min_distance)
            
            # Adiciona a força à partícula atual
            current_particle.ax += fx_right
            current_particle.ay += fy_right

            # Pela 3ª Lei de Newton, a força oposta atua no vizinho da direita
            right_neighbor.ax -= fx_right
            right_neighbor.ay -= fy_right

    # Aplica a força de amortecimento e calcula a aceleração final para todas as partículas
    for i, particle in enumerate(particles):
        # Força de amortecimento
        f_damp_x = b * particle.vx
        f_damp_y = b * particle.vy
        
        # Aceleração final = (Soma das forças de mola - Força de amortecimento) / Massa
        particle.ax = -(particle.ax + f_damp_x) / masses[i]
        particle.ay = -(particle.ay + f_damp_y) / masses[i]

def update_damped_positions(particles, dt):
    """
    Atualiza as posições de todas as partículas usando o método de Verlet.
    """
    for particle in particles:
        # Posição = Posição_antiga + Velocidade*dt + 0.5*Aceleração*dt^2
        particle.x += particle.vx * dt + 0.5 * particle.ax * dt**2
        particle.y += particle.vy * dt + 0.5 * particle.ay * dt**2

def velocities_damped(particles, dt, natural_distance, k, b, masses):
    """
    Atualiza as velocidades de todas as partículas usando o método de Verlet.
    """
    # Armazena as acelerações antigas antes de recalcular as novas
    old_accs = [(particle.ax, particle.ay) for particle in particles]
    
    # Loop para calcular as novas velocidades
    for i, particle in enumerate(particles):
        # Velocidade = Velocidade_antiga + 0.5 * (Aceleração_nova + Aceleração_antiga) * dt
        particle.vx += 0.5 * (particle.ax + old_accs[i][0]) * dt
        particle.vy += 0.5 * (particle.ay + old_accs[i][1]) * dt

def plot_relative_distance(natural_distance, time_steps, relative_distances, filename="relative_distance.png"):
    """
    Gera e salva um gráfico da distância relativa entre duas partículas ao longo do tempo.
    """
    # Cria a pasta 'graphs' se não existir
    graphs_folder = create_graphs_folder()
    filepath = os.path.join(graphs_folder, filename)

    plt.figure(figsize=(10, 6))
    plt.plot(time_steps, relative_distances, label='Distância Relativa entre P1 e P2')
    plt.axhline(y=natural_distance, color='r', linestyle='--', label='Distância Natural da Mola') # Adiciona linha para distância natural
    plt.title('Distância Relativa entre a Primeira e a Segunda Partícula no Tempo')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Distância Relativa')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close() # Fecha a figura para liberar memória