import math
import methods
import process

from random import randint, sample, uniform
from time import perf_counter

def generate_initial_swarm(problem: process.Problem, size: int, swarm: list, objectives: list) -> methods.Particle:
    best_particle = methods.Particle({}, [], {}, [], [], {}, [], 0, 0, 0.0)
    
    aisles = [a for a in range(problem.a)]
    for _ in range(size):
        quantity = randint(1, problem.a)
        particle = methods.generate_particle(problem, quantity, aisles)

        objectives.append(particle.objective)
        swarm.append({
            "particle": particle,
            "X": set(particle.aisles),
            "P": set(particle.aisles),
            "V": set()
        })

        if particle.objective > best_particle.objective:
            best_particle = particle
    
    return best_particle.clone()

def scalar_multiplication(n: float, V: set[tuple[str, int]]) -> set[tuple[str, int]]:
    length = math.floor(n * len(V))
    return set(sample(sorted(V), k = length))

def difference_in_positions(X1: set[int], X2: set[int]) -> set[tuple[str, int]]:
    velocity = set()

    add = X1 - X2
    while add:
        velocity.add(("+", add.pop()))
    
    remove = X2 - X1
    while remove:
        velocity.add(("-", remove.pop()))

    return velocity

def number_of_elements(B: float, P: set[int]) -> int:
    number = math.floor(B)
    if uniform(0, 1) < B - number:
        number += 1
    
    return min(len(P), number)

def k_tournament_selection(problem: process.Problem, particle: methods.Particle, k: float, A: set[int], N: int) -> set[tuple[str, int]]:
    velocity = set()
    sorted_a = sorted(A)
    sorted_len = len(sorted_a)
    
    for _ in range(N):
        best_element = 0
        best_value = 0
        
        elements = sample(sorted_a, min(k, sorted_len))
        for e in elements:
            particle_clone = particle.clone()
            particle_clone.add_aisle(problem, e)
            particle_clone.add_orders(problem)
            particle_clone.objective_function(problem)

            if particle_clone.objective > best_value:
                best_element = e
                best_value = particle_clone.objective

        velocity.add(("+", best_element))
    return velocity

def removal_of_elements(S: set[int], N: int) -> set[tuple[str, int]]:
    number = math.floor(N)
    elements = sample(sorted(S), k = number)

    velocity = set()
    for e in elements:
        velocity.add(("-", e))
    
    return velocity

def SBPSO(problem: process.Problem, size: int, max_generation: int, c1: float, c2: float, c3: float, c4: float, k: float) -> None:
    if c1 < 0 or c1 > 1:
        return
    if c2 < 0 or c2 > 1:
        return
    
    start = perf_counter()

    objectives = []
    swarm = []

    best_particle = generate_initial_swarm(problem, size, swarm, objectives)
    G = set(best_particle.aisles)
    U = set([a for a in range(problem.a)])

    for i in swarm:
        print(i["particle"].aisles)

    generation = 0
    while generation < max_generation:
        print(best_particle.objective)
        print(best_particle.aisles)

        for i in range(size):
            pbest_diff = difference_in_positions(swarm[i]["P"], swarm[i]["X"])
            pbest_velocity = scalar_multiplication(c1 * uniform(0, 1), pbest_diff)

            gbest_diff = difference_in_positions(G, swarm[i]["X"])
            gbest_velocity = scalar_multiplication(c2 * uniform(0, 1), gbest_diff)

            A = U - (swarm[i]["X"] | swarm[i]["P"] | G)
            random_addition = k_tournament_selection(problem, swarm[i]["particle"], k, A, number_of_elements(c3 * uniform(0, 1), A))

            S = swarm[i]["X"] & swarm[i]["P"] & G
            random_removal = removal_of_elements(S, number_of_elements(c4 * uniform(0, 1), S))

            swarm[i]["V"] = pbest_velocity | gbest_velocity | random_addition | random_removal

        for i in range(size):
            while swarm[i]["V"]:
                operation, aisle = swarm[i]["V"].pop()
                if operation == "+":
                    swarm[i]["particle"].add_aisle(problem, aisle)
                else:
                    swarm[i]["particle"].remove_aisle(problem, aisle)

            old_objective = swarm[i]["particle"].objective
            swarm[i]["particle"].add_orders(problem)
            swarm[i]["particle"].objective_function(problem)

            if swarm[i]["particle"].objective >= old_objective:
                swarm[i]["P"] = set(swarm[i]["particle"].aisles)

            if swarm[i]["particle"].objective >= best_particle.objective:
                best_particle = swarm[i]["particle"].clone()
                G = set(best_particle.aisles)

        generation += 1

    end = perf_counter()
    problem.result["orders"] = best_particle.orders
    problem.result["aisles"] = best_particle.aisles
    problem.result["objective"] = best_particle.objective
    problem.result["time"] = end - start