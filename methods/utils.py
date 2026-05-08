from dataclasses import dataclass

@dataclass
class Particle():
    """
    Represents a particle in the swarm. It contains attributes useful for both PSO paradigms.

    Attributes:
        aisles_items (list[int]): Number of items available in selected aisles.
        number_items (int): Total number of items in the selected orders.
        number_aisles (int): Number of aisles selected.
        objective (float): Value of the objective function for the particle.
    """
    
    aisles_items: list[int]
    number_items: int
    number_aisles: int
    objective: float