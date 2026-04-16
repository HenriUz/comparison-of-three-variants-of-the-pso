import math

from dataclasses import dataclass
from methods.utils import add_orders, Particle
from process.dataset import Problem
from random import sample, uniform
from time import perf_counter

@dataclass
class SetParticle(Particle):
    x: set[int]
    pbest: set[int]

def generate_particle(problem: Problem) -> SetParticle:
    """
    Generates a particle with random elements. Each element in the universe has a 50% chance of being selected.
    
    Args:
        problem (Problem): An instance of the Problem class containing the dataset information.
    
    Returns:
        particle (SetParticle): The particle that was created.
    """
    
    particle = SetParticle(
        aisles_items = dict.fromkeys(range(problem.i), 0),
        selected_aisles = [1 if uniform(0, 1) <= 0.5 else 0 for _ in range(problem.a)],
        orders = [],
        number_items = 0,
        number_aisles = 0,
        objective = 0.0,
        x = set(),
        pbest = set()
    )

    # Updating the attributes associated with the aisles.
    for a in range(problem.a):
        if particle.selected_aisles[a]:
            particle.number_aisles += 1
            particle.x.add(a)
            particle.pbest.add(a)
            for item, qty in problem.aisles[a].items():
                particle.aisles_items[item] += qty

    particle.number_items, particle.orders = add_orders(problem, particle.aisles_items)
    particle.objective = problem.objective_function(particle.number_items, particle.number_aisles)
    return particle

def generate_initial_swarm(problem: Problem, size: int, swarm: list) -> tuple[float, int, set[int], list[int]]:
    """
    Generate the swarm particles.

    The function returns a tuple containing only the essential information about the best particle in the swarm, namely: the objective function value, the number of aisles, the set of aisles, and the list of orders in the solution.

    Args:
        problem (Problem): An instance of the Problem class containing the dataset information.
        size (int): Number of particles in the swarm.
        swarm (list): An empty list where the particles will be stored.
    
    Returns:
        best_particle (tuple[float, int, set[int], list[int]]): Information from the best particle.
    """
    
    best_obj = 0.0
    best_n_aisles = 0
    best_orders = []
    G = set()

    for _ in range(size):
        particle = generate_particle(problem)
        swarm.append(particle)

        if particle.objective > best_obj or (particle.objective == best_obj and particle.number_aisles < best_n_aisles):
            best_obj = particle.objective
            best_n_aisles = particle.number_aisles
            best_orders = particle.orders
            G = particle.pbest
    
    return (best_obj, best_n_aisles, G.copy(), best_orders[:])

def scalar_multiplication(n: float, V: set[tuple[str, int]]) -> set[tuple[str, int]]:
    """
    Multiplies a velocity (`V`) by a scalar value (`n`). The result is a new velocity with `x` random elements of `V`. The number of random elements is defined by `floor(n * |V|)`.

    Note that the scalar value must be within the closed interval between 0 and 1, since sets cannot have a negative number of elements or repeated elements. Thus, if `n` is 1, the result is the velocity itself, and if `n` is 0, the result is empty.

    Args:
        n (float): Scalar value in the closed interval between 0 and 1.
        V (set[tuple[str, int]]): Set of operations to be performed on the particle.
    
    Returns:
        V' (set[tuple[str, int]]): A set of randomly selected operations.
    """

    if n < 0 or n > 1:
        return set()
    
    length = math.floor(n * len(V))
    return set(sample(sorted(V), k = length))

def difference_in_positions(X1: set[int], X2: set[int]) -> set[tuple[str, int]]:
    """
    Calculates the difference between two sets of positions (`X1` and `X2`), resulting in the velocity needed to transform `X2` into `X1`, by adding all elements that are only in `X1` (`X1 - X2`) and removing all elements that are only in `X2` (`X2 - X1`).

    Args:
        X1 (set[int]): Target positions.
        X2 (set[int]): The set that will be transformed.

    Returns:
        V (set[tuple[str, int]]): Velocity required to transform `X2` into `X1`. 
    """
    
    velocity = set()

    add = X1 - X2
    while add:
        velocity.add(("+", add.pop()))
    
    remove = X2 - X1
    while remove:
        velocity.add(("-", remove.pop()))

    return velocity

def number_of_elements(B: float, P: set[int]) -> int:
    """
    Calculates the number of elements to be selected from a set (`P`). This number is limited by the size of `P`, and is calculated as `floor(B) + 1` or `floor(B) + 0`, depending on whether the condition `uniform(0, 1) < B - floor(B)` is satisfied.

    Args:
        B (float): Number used to determine the quantity of elements.
        P (set[int]): Set of positions.
    
    Returns:
        N (int): Number of elements.
    """
    
    number = math.floor(B)
    if uniform(0, 1) < B - number:
        number += 1
    
    return min(len(P), number)

def k_tournament_selection(problem: Problem, particle: SetParticle, A: set[int], N: int, k: int) -> set[tuple[str, int]]:
    """
    Select `N` elements from the universe that are not in the union of the sets of the particle's position, its best position and the swarm's best position (`A`), to add to the current position.

    To avoid adding elements completely at random, for each `N`, `k` elements are selected at random, and only the best one is chosen.

    Args:
        problem (Problem): An instance of the Problem class containing the dataset information.
        particle (SetParticle): The instance of the current particle.
        A (set[int]): Elements of the universe that are not members of the union of the sets.
        N (int): Number of items to be added.
        k (int): Number of particles sampled before selecting the best one.
    
    Returns:
        V (set[tuple[str, int]]): Elements that will be added to the set.
    """
    
    velocity = set()
    remaining = list(A)
    base_items = particle.aisles_items.copy()
    
    for _ in range(N):
        # Selecting the k elements (if the size of the set is less than k, the elements will be the set itself).    
        k_actual = min(k, len(remaining))
        elements = sample(remaining, k_actual)
        
        # Selecting the best.
        best_element = -1
        best_value = -1
        best_n_aisles = problem.a
        
        for e in elements:
            # Adding the temporary element and updating available items.
            temp_items = base_items.copy()
            for item, qty in problem.aisles[e].items():
                temp_items[item] += qty
            
            # Calculating the value of the objective function and identifying the optimal element.
            number_items, _ = add_orders(problem, temp_items)
            obj = problem.objective_function(number_items, particle.number_aisles + 1)

            if obj > best_value or (obj == best_value and particle.number_aisles + 1 < best_n_aisles):
                best_element = e
                best_value = obj
                best_n_aisles = particle.number_aisles + 1

        velocity.add(("+", best_element))
        remaining.remove(best_element)
        
        # Updating items to reflect that the best option has already been chosen.
        for item, qty in problem.aisles[best_element].items():
            base_items[item] += qty

    return velocity

def removal_of_elements(S: set[int], N: int) -> set[tuple[str, int]]:
    """
    Given the set of points where the particle's position, its best position, and the swarm's best position (`S`) intersect, select `N` random elements to remove.

    Note that `N` must be in the closed interval from 0 to the size of the set.
    
    Args:
        S (set[int]): Intersection of sets.
        N (int): Number of items to select at random.
    
    Returns:
        V (set[tuple[str, int]]): Elements that will be removed from the set.
    """
    
    elements = sample(sorted(S), k = N)
    
    velocity = set()
    for e in elements:
        velocity.add(("-", e))
    
    return velocity

def SBPSO(problem: Problem, size: int, max_generation: int, c1: float, c2: float, c3: float, c4: float, k: int) -> None:
    """
    Implementation of Set-Based PSO for the order-batching approach. The algorithm operates solely within the set of aisles, and orders are filled greedily based on the selected aisles.

    In this algorithm, `c1` and `c2` must be in the closed interval between 0 and 1, and `c3` and `c4` in the closed interval between 0 and the number of aisles. `c3` and `c4` are not checked, since the functions that use them limit the size to the size of the universe if they are larger.

    If the value of the objective function is the same when determining the best individual and global position, the postition with the fewest aisles will be chosen.

    The solution will be saved directly in `problem`.

    Args:
        problem (Problem): An instance of the Problem class containing the dataset information.
        size (int): Number of particles in the swarm.
        max_generation (int): Stop condition.
        c1 (float): Cognitive component.
        c2 (float): Social component.
        c3 (float): Addition component.
        c4 (float): Removal component.
        k (int): Number for tournament selection.
    """
    
    if c1 < 0 or c1 > 1:
        return
    if c2 < 0 or c2 > 1:
        return
    
    start = perf_counter()

    # Starting the swarm.
    swarm = []
    best_obj, best_n_aisles, G, best_orders = generate_initial_swarm(problem, size, swarm)
    U = set([a for a in range(problem.a)])

    old_n_aisles = [0 for _ in range(size)]
    
    generation = 0
    while generation < max_generation:
        # Calculating velocity and updating positions.
        for i in range(size):
            pbest_diff = difference_in_positions(swarm[i].pbest, swarm[i].x)
            pbest_velocity = scalar_multiplication(c1 * uniform(0, 1), pbest_diff)

            gbest_diff = difference_in_positions(G, swarm[i].x)
            gbest_velocity = scalar_multiplication(c2 * uniform(0, 1), gbest_diff)

            A = U - (swarm[i].x | swarm[i].pbest | G)
            random_addition = k_tournament_selection(problem, swarm[i], A, number_of_elements(c3 * uniform(0, 1), A), k)

            S = swarm[i].x & swarm[i].pbest & G
            random_removal = removal_of_elements(S, number_of_elements(c4 * uniform(0, 1), S))

            velocity = pbest_velocity | gbest_velocity | random_addition | random_removal

            old_n_aisles[i] = swarm[i].number_aisles
            while velocity:
                op, aisle = velocity.pop()
                if op == "+":
                    swarm[i].selected_aisles[aisle] = 1
                    swarm[i].number_aisles += 1
                    swarm[i].x.add(aisle)

                    for item, qty in problem.aisles[aisle].items():
                        swarm[i].aisles_items[item] += qty
                else:
                    swarm[i].selected_aisles[aisle] = 0
                    swarm[i].number_aisles -= 1
                    swarm[i].x.remove(aisle)

                    for item, qty in problem.aisles[aisle].items():
                        swarm[i].aisles_items[item] -= qty
        
        # Updating personal and global values.
        for i in range(size):
            old_obj = swarm[i].objective

            swarm[i].number_items, swarm[i].orders = add_orders(problem, swarm[i].aisles_items)
            swarm[i].objective = problem.objective_function(swarm[i].number_items, swarm[i].number_aisles)

            if swarm[i].objective > old_obj or (swarm[i].objective == old_obj and swarm[i].number_aisles < old_n_aisles[i]):
                swarm[i].pbest = swarm[i].x.copy()

            if swarm[i].objective > best_obj or (swarm[i].objective == best_obj and swarm[i].number_aisles < best_n_aisles):
                best_obj = swarm[i].objective
                best_n_aisles = swarm[i].number_aisles
                best_orders = swarm[i].orders[:]
                G = swarm[i].pbest.copy()

        generation += 1

    end = perf_counter()
    problem.result["orders"] = best_orders
    problem.result["aisles"] = list(G)
    problem.result["objective"] = best_obj
    problem.result["time"] = end - start