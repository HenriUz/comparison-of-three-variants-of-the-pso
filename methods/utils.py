import process

from dataclasses import dataclass
from random import sample
from typing import Dict, List

@dataclass
class Particle:
    available_items: Dict[int, int]
    aisles: List[int]
    aisles_items: Dict[int, int]
    selected_aisles: List[int]
    orders: List[int]
    orders_items: Dict[int, int]
    selected_orders: List[int]
    number_items: int
    number_aisles: int
    objective: float

    def clone(self) -> Particle:
        return Particle(
            self.available_items.copy(),
            self.aisles[:],
            self.aisles_items.copy(),
            self.selected_aisles[:],
            self.orders[:],
            self.orders_items.copy(),
            self.selected_orders[:],
            self.number_items,
            self.number_aisles,
            self.objective,
        )

    def objective_function(self, problem: process.Problem) -> None:
        if self.number_items < problem.lb or self.number_items > problem.ub:
            self.objective = 0.0
            return

        for o in self.orders_items:
            if self.orders_items[o] > self.aisles_items[o]:
                self.objective = 0.0
                return

        self.objective = self.number_items / self.number_aisles

    def add_orders(self, problem: process.Problem) -> None:
        self.available_items = self.aisles_items.copy()
        self.orders = []
        self.orders_items = dict.fromkeys(range(problem.i), 0)
        self.selected_orders = [0 for _ in range(problem.o)]
        self.number_items = 0

        valid_orders = []
        for o in range(problem.o):
            if not self.selected_orders[o]:
                valid = True
                total = 0
                for item, quantity in problem.orders[o].items():
                    total += quantity
                    if quantity > self.available_items[item]:
                        valid = False
                        break
                if valid:
                    valid_orders.append([o, total])

        valid_orders.sort(key = lambda i: i[1], reverse = True)
        for o in valid_orders:
            valid = True
            for item, quantity in problem.orders[o[0]].items():
                if quantity > self.available_items[item]:
                    valid = False
                    break
            if valid and self.number_items + o[1] <= problem.ub:
                self.number_items += o[1]
                self.selected_orders[o[0]] = 1
                self.orders.append(o[0])
                for item, quantity in problem.orders[o[0]].items():
                    self.available_items[item] -= quantity
                    self.orders_items[item] += quantity
    
    def add_aisle(self, problem: process.Problem, aisle: int) -> None:
        if aisle >= 0 and aisle <= problem.a and not self.selected_aisles[aisle]:
            self.aisles.append(aisle)
            self.selected_aisles[aisle] = 1
            self.number_aisles += 1

            for item, quantity in problem.aisles[aisle].items():
                self.aisles_items[item] += quantity

    def remove_aisle(self, problem: process.Problem, aisle: int) -> None:
        if aisle >= 0 and aisle <= problem.a and self.selected_aisles[aisle]:
            self.aisles.remove(aisle)
            self.selected_aisles[aisle] = 0
            self.number_aisles -= 1

            for item, quantity in problem.aisles[aisle].items():
                self.aisles_items[item] -= quantity

def generate_particle(problem: process.Problem, number_aisles: int, aisles: list) -> Particle:
        particle = Particle(
            dict.fromkeys(range(problem.i), 0),
            sample(aisles, k = number_aisles),
            dict.fromkeys(range(problem.i), 0),
            [0 for _ in range(problem.a)],
            [],
            dict.fromkeys(range(problem.i), 0),
            [0 for _ in range(problem.o)],
            0,
            number_aisles,
            0.0,
        )

        for a in particle.aisles:
            particle.selected_aisles[a] = 1
            for item, quantity in problem.aisles[a].items():
                particle.aisles_items[item] += quantity
                particle.available_items[item] += quantity
        
        particle.add_orders(problem)
        particle.objective_function(problem)
        return particle