# Define particle
class Particle:
    def __init__(self, x, y):
        # Positions
        self.x = x
        self.y = y

        # Speed
        self.vx = 0
        self.vy = 0
        

    # Visualization on print
    def __repr__(self):
        return f"Particle(x={self.x:.2f}, y={self.y:.2f})"