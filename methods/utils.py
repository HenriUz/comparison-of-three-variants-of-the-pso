from abc import ABC, abstractmethod
from dataclasses import dataclass
from process.dataset import Problem

@dataclass
class Particle(ABC):
    """
    Represents a particle in the swarm. It contains attributes useful for both PSO paradigms.

    Attributes:
        aisles_items (dict[int, int]): Number of items available in selected aisles.
        selected_aisles (list[int]): A binary list indicating whether or not an aisle is present.
        orders (list[int]): Indexes of the orders included in the solution.
        number_items (int): Total number of items in the selected orders.
        number_aisles (int): Number of aisles selected.
        objective (float): Value of the objective function for the particle.
    """
    
    aisles_items: dict[int, int]
    selected_aisles: list[int]
    orders: list[int]
    number_items: int
    number_aisles: int
    objective: float

    @abstractmethod
    def clone(self) -> Particle:
        pass

    def add_orders(self, problem: Problem) -> None:
        """
        Add orders to the solution in a greedy manner, selecting those with the largest number of items (`problem` is expected to already provide the sorted list), without violating upper bound and supply constraints.

        The strategy of this function is to always assume that no orders have been selected, so every time it is called, the orders are populated from scratch.

        Args:
            problem (Problem): An instance of the Problem class containing the dataset information.
        """
        
        # Resetting order information.
        self.orders = []
        self.number_items = 0

        # Adding orders that do not violate the upper bound and supply constraints.
        for o in problem.sorted_orders:
            valid = True
            for item, quantity in problem.orders[o[0]].items():
                if quantity > self.aisles_items[item]:
                    valid = False
                    break
            if valid and self.number_items + o[1] <= problem.ub:
                self.number_items += o[1]
                self.orders.append(o[0])
                for item, quantity in problem.orders[o[0]].items():
                    self.aisles_items[item] -= quantity