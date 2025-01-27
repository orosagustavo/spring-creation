# Define particle
class Particle:
    def __init__(self, x, y, vx=0, vy=0, type_mol = 1, ax = 0, ay = 0):
        if type_mol == 1:
            # Particle Type
            self.type_mol = 'P'
        if type_mol == 3:
            self.type_mol = 'S'
        
        # Positions
        self.x = x
        self.y = y

        # Speed
        self.vx = vx
        self.vy = vy

        # Accelerations
        self.ax = ax
        self.ay = ay
        
    # Visualization on print
    def __repr__(self):
        return [self.x, self.y, self.vx, self.vy]
    
    # Calculate relative positions
    def relative_position(self, other):
        dx = other.x - self.x
        dy = other.y - self.y
        return dx, dy
