import math

from dataclasses import dataclass
from methods.utils import Particle
from process.dataset import Problem
from random import uniform
from time import perf_counter

@dataclass
class BinaryParticle(Particle):
    pbest: list[int]
    transfer: list[float]
    velocity: list[float]

    def clone(self) -> BinaryParticle:
        return BinaryParticle(
            self.aisles_items.copy(),
            self.selected_aisles[:],
            self.orders[:],
            self.number_items,
            self.number_aisles,
            self.objective,
            self.pbest[:],
            self.transfer[:],
            self.velocity[:]
        )

def generate_particle(problem: Problem) -> BinaryParticle:
    """
    Generates a particle with random elements. Each element in the universe has a 50% chance of being selected.
    
    Args:
        problem (Problem): An instance of the Problem class containing the dataset information.
        
    Returns:
        particle (BinaryParticle): The particle that was created.
    """

    particle = BinaryParticle(
        aisles_items = dict.fromkeys(range(problem.i), 0),
        selected_aisles = [1 if uniform(0, 1) <= 0.5 else 0 for _ in range(problem.a)],
        orders = [],
        number_items = 0,
        number_aisles = 0,
        objective = 0.0,
        pbest = [0 for _ in range(problem.a)],
        transfer = [0 for _ in range(problem.a)],
        velocity = []
    )
    
    # Updating the attributes associated with the aisles.
    for a in range(problem.a):
        if particle.selected_aisles[a]:
            particle.number_aisles += 1
            particle.pbest[a] = 1
            for item, quantity in problem.aisles[a].items():
                particle.aisles_items[item] += quantity
    
    particle.add_orders(problem)
    particle.objective = problem.objective_function(particle.number_items, particle.number_aisles)
    return particle

def generate_initial_swarm(problem: Problem, size: int, v_min: float, v_max: float, swarm: list) -> tuple[float, int, list[int], list[int]]:
    """
    Generate the swarm particles.

    The function returns a tuple containing only the essential information about the best particle in the swarm, namely: the objective function value, the number of aisles, the list of selected aisles, and the list of orders in the solution.

    The inital velocity of each component of the particle's velocity is generated randomly within the range of the maximum and minumum velocities. Performance is slightly degraded by the addition of another loop to calculate the velocities; this could be avoided by calculating the velocity during particle generation, but to ensure that the particles have the same initial position across all methods (including SBPSO), this step is necessary.

    Args:
        problem (Problem): An instance of the Problem class containing the dataset information.
        size (int): Number of particles in the swarm.
        v_min (float): Minimum velocity.
        v_max (float): Maximum velocity.
        swarm (list): An empty list where the particles will be stored.
    
    Returns:
        best_particle (tuple[float, int, list[int], list[int]]): Information from the best particle.
    """
    
    best_objective = 0.0
    best_number_aisles = 0
    best_aisles = []
    best_orders = []

    for _ in range(size):
        particle = generate_particle(problem)
        swarm.append(particle)

        # print(particle.selected_aisles)

        if particle.objective > best_objective or (particle.objective == best_objective and particle.number_aisles < best_number_aisles):
            best_objective = particle.objective
            best_number_aisles = particle.number_aisles
            best_aisles = particle.selected_aisles
            best_orders = particle.orders
    
    for i in range(size):
        swarm[i].velocity = [((v_max - v_min) * uniform(0, 1) + v_min) for _ in range(problem.a)]
    
    return (best_objective, best_number_aisles, best_aisles[:], best_orders[:])

def YMBPSO(problem: Problem, size: int, max_generation: int, w_max: float, w_min:float , c1: float, c2: float, v_min: float, v_max: float, r_mu: float) -> None:
    """
    Implementation of Modified Binary PSO for the order-batching approach. The algorithm operates solely within the set of aisles, and orders are filled greedily based on the selected aisles.

    In this implementation, the following transfer function is used:
    - If x <= 0: 1 - 2 / (1 + exp(-x))
    - If x > 0: 2 / (1 + exp(-x)) - 1
    
    Where x is the particle's new velocity.

    The solution will be saved directly in `problem`.

    Args:
        problem (Problem): An instance of the Problem class containing the dataset information.
        size (int): Number of particles in the swarm.
        max_generation (int): Stop condition.
        w_max (float): Initial value of inertia.
        w_min (float): Final value of inertia.
        c1 (float): Cognitive component.
        c2 (float): Social component.
        v_min (float): Minimum velocity.
        v_max (float): Maximum velocity.
        r_mu (float): Probability of mutation.
    """

    start = perf_counter()
    
    # Starting the swarm.
    swarm = []
    best_objective, best_number_aisles, best_aisles, best_orders = generate_initial_swarm(problem, size, v_min, v_max, swarm)
    
    old_number_aisles = [0 for _ in range(size)]

    generation = 0
    while generation < max_generation:
        # Applying a linear reduction to the inertia component.
        w = (w_max - w_min) * ((max_generation - generation) / max_generation) + w_min

        # Calculating velocity and updating positions.
        for i in range(size):
            swarm[i].aisles_items = dict.fromkeys(range(problem.i), 0)
            old_number_aisles[i] = swarm[i].number_aisles
            swarm[i].number_aisles = 0
            for a in range(problem.a):
                swarm[i].velocity[a] = w * swarm[i].velocity[a]
                swarm[i].velocity[a] += c1 * uniform(0, 1) * (swarm[i].pbest[a] - swarm[i].selected_aisles[a])
                swarm[i].velocity[a] += c2 * uniform(0, 1) * (best_aisles[a] - swarm[i].selected_aisles[a])
        
                swarm[i].velocity[a] = max(v_min, swarm[i].velocity[a])
                swarm[i].velocity[a] = min(v_max, swarm[i].velocity[a])

                # Calculating the value of the transfer function and applying it to the paricle's position.
                if swarm[i].velocity[a] <= 0:
                    swarm[i].transfer[a] = 1 - 2 / (1 + math.exp(-swarm[i].velocity[a]))
                else:
                    swarm[i].transfer[a] = 2 / (1 + math.exp(-swarm[i].velocity[a])) - 1
        
                if uniform(0, 1) <= swarm[i].transfer[a]:
                    if swarm[i].velocity[a] <= 0:
                        swarm[i].selected_aisles[a] = 0
                    else:
                        swarm[i].selected_aisles[a] = 1

                # Checking whether a mutation will occur.
                if uniform(0, 1) < r_mu:
                    swarm[i].selected_aisles[a] = 1 - swarm[i].selected_aisles[a]

                # Updating available items.
                if swarm[i].selected_aisles[a]:
                    swarm[i].number_aisles += 1
                    for item, quantity in problem.aisles[a].items():
                        swarm[i].aisles_items[item] += quantity

        # Updating personal and global values.
        for i in range(size):
            old_objective = swarm[i].objective

            swarm[i].add_orders(problem)
            swarm[i].objective = problem.objective_function(swarm[i].number_items, swarm[i].number_aisles)
            
            if swarm[i].objective > old_objective or (swarm[i].objective == old_objective and swarm[i].number_aisles < old_number_aisles[i]):
                swarm[i].pbest = swarm[i].selected_aisles[:]
            
            if swarm[i].objective > best_objective or (swarm[i].objective == best_objective and swarm[i].number_aisles < best_number_aisles):
                best_objective = swarm[i].objective
                best_number_aisles = swarm[i].number_aisles
                best_aisles = swarm[i].selected_aisles[:]
                best_orders = swarm[i].orders[:]
        
        generation += 1

    end = perf_counter()

    # Converting the binary list into a list of aisles to save in the solution.
    aisles = []
    for a in range(problem.a):
        if best_aisles[a]:
            aisles.append(a)

    problem.result["orders"] = best_orders
    problem.result["aisles"] = aisles
    problem.result["objective"] = best_objective
    problem.result["time"] = end - start

def ZMBPSO(problem: Problem, size: int, max_generation: int, w_max: float, w_min:float , c1: float, c2: float, v_min: float, v_max: float, r_mu: float, k: float) -> None:
    """
    Implementation of Modified Binary PSO for the order-batching approach. The algorithm operates solely within the set of aisles, and orders are filled greedily based on the selected aisles.

    In this implementation, the following transfer function is used: 1 - exp(-`k` * |x|)
    
    Where x is the particle's new velocity.

    The solution will be saved directly in `problem`.

    Args:
        problem (Problem): An instance of the Problem class containing the dataset information.
        size (int): Number of particles in the swarm.
        max_generation (int): Stop condition.
        w_max (float): Initial value of inertia.
        w_min (float): Final value of inertia.
        c1 (float): Cognitive component.
        c2 (float): Social component.
        v_min (float): Minimum velocity.
        v_max (float): Maximum velocity.
        r_mu (float): Probability of mutation.
        k (float): Value that controls the format of the transfer function.
    """

    start = perf_counter()
    
    # Starting the swarm.
    swarm = []
    best_objective, best_number_aisles, best_aisles, best_orders = generate_initial_swarm(problem, size, v_min, v_max, swarm)

    old_number_aisles = [0 for _ in range(size)]

    generation = 0
    while generation < max_generation:
        # Applying a linear reduction to the inertia component.
        w = (w_max - w_min) * ((max_generation - generation) / max_generation) + w_min
        
        # Calculating velocity and updating positions.
        for i in range(size):
            swarm[i].aisles_items = dict.fromkeys(range(problem.i), 0)
            old_number_aisles[i] = swarm[i].number_aisles
            swarm[i].number_aisles = 0
            for a in range(problem.a):
                swarm[i].velocity[a] = w * swarm[i].velocity[a]
                swarm[i].velocity[a] += c1 * uniform(0, 1) * (swarm[i].pbest[a] - swarm[i].selected_aisles[a])
                swarm[i].velocity[a] += c2 * uniform(0, 1) * (best_aisles[a] - swarm[i].selected_aisles[a])
        
                swarm[i].velocity[a] = max(v_min, swarm[i].velocity[a])
                swarm[i].velocity[a] = min(v_max, swarm[i].velocity[a])

                # Calculating the value of the transfer function and applying it to the paricle's position.
                swarm[i].transfer[a] = 1 - math.exp(-k * abs(swarm[i].velocity[a]))
                if uniform(0, 1) <= swarm[i].transfer[a]:
                    if swarm[i].velocity[a] <= 0:
                        swarm[i].selected_aisles[a] = 0
                    else:
                        swarm[i].selected_aisles[a] = 1

                # Checking whether a mutation will occur.
                if uniform(0, 1) < r_mu:
                    swarm[i].selected_aisles[a] = 1 - swarm[i].selected_aisles[a]

                # Updating available items.
                if swarm[i].selected_aisles[a]:
                    swarm[i].number_aisles += 1
                    for item, quantity in problem.aisles[a].items():
                        swarm[i].aisles_items[item] += quantity

        # Updating personal and global values.
        for i in range(size):
            old_objective = swarm[i].objective

            swarm[i].add_orders(problem)
            swarm[i].objective = problem.objective_function(swarm[i].number_items, swarm[i].number_aisles)
            
            if swarm[i].objective > old_objective or (swarm[i].objective == old_objective and swarm[i].number_aisles < old_number_aisles[i]):
                swarm[i].pbest = swarm[i].selected_aisles[:]
            
            if swarm[i].objective > best_objective or (swarm[i].objective == best_objective and swarm[i].number_aisles < best_number_aisles):
                best_objective = swarm[i].objective
                best_number_aisles = swarm[i].number_aisles
                best_aisles = swarm[i].selected_aisles[:]
                best_orders = swarm[i].orders[:]
        
        generation += 1

    end = perf_counter()

    # Converting the binary list into a list of aisles to save in the solution.
    aisles = []
    for a in range(problem.a):
        if best_aisles[a]:
            aisles.append(a)

    problem.result["orders"] = best_orders
    problem.result["aisles"] = aisles
    problem.result["objective"] = best_objective
    problem.result["time"] = end - start