import pandas as pd
import random
import math
import sys
import os
import csv
import time
import matplotlib.pyplot as plt
from typing import List, Tuple

'''Simulated Annealing algorithm'''


# Distance calculation function for Simulated Annealing
def calculate_distance(solution: List[Tuple[str, float, float]]) -> float:
    t_distance = 0
    # last_distance = 0
    for i in range(1, len(solution)):
        x1, y1, z1 = solution[i - 1]
        x2, y2, z2 = solution[i]
        t_distance += math.sqrt((y2 - y1) ** 2 + (z2 - z1) ** 2)
    return t_distance


# Distance calculation function for Genetic Algorithm
def calculate_distance2(individual: List[Tuple[str, int]], coordinates: List[Tuple[str, float, float]]) -> float:
    t_distance = 0
    coord_dict = {coord[0]: coord[1:] for coord in coordinates}
    path_dist = [coord_dict[i[0]] for i in individual if
                 i[0] in coord_dict]  # Use the integer part of the tuple as the index
    for i in range(1, len(path_dist)):
        x1, y1 = path_dist[i - 1]
        x2, y2 = path_dist[i]
        t_distance += math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return t_distance


# Simulated Annealing main code
def simulatedAnnealing(temperature: int, cooling_schedule: float, coordinates: List[Tuple[str, float, float]],
                       filename: str):
    # Execution time start
    start_time = time.time()
    iterations = 0

    # Defining current solution
    cur_solution = coordinates
    cur_solution.append(cur_solution[0])
    initState = cur_solution[0][0]
    initSolution = cur_solution[:]
    initTemperature = temperature
    init_path_cost = calculate_distance(cur_solution)
    while temperature > 0:
        # Generating new solution by swapping 2 edges in the current solution in every loop
        new_solution = cur_solution[:]
        edge1, edge2 = random.sample(range(1, len(new_solution)), 2)
        new_solution[edge1:edge2] = reversed(new_solution[edge1:edge2])
        distance_difference = calculate_distance(cur_solution) - calculate_distance(new_solution)

        # Exception handling to avoid the OverflowError
        try:
            acceptance_probability = math.exp(-distance_difference / temperature)
        except OverflowError:
            acceptance_probability = 0.0
        # If the distance difference is positive, we allow the new solution
        if distance_difference > 0:
            cur_solution = new_solution[:]

        # If the distance difference is negative, then we allow only with certain probability
        elif acceptance_probability > random.random():
            cur_solution = new_solution[:]

        # Temperature cools down at the rate of cooling schedule/cooling rate
        temperature *= cooling_schedule
        iterations += 1

    # PLotting the graph
    x_val = [coord[1] for coord in cur_solution]
    y_val = [coord[2] for coord in cur_solution]
    plt.plot(x_val, y_val, marker='o')
    path_cost = calculate_distance(cur_solution)
    formatted_cost = f"{path_cost:.2f}"
    end_time = time.time()
    plt.title("Simulated Annealing Graph\n"
              "(Close this window to display the final results in terminal)")
    plt.show()

    # Calculating total execution time
    total_time = end_time - start_time
    print(f'''Somanagoudar, Saikiran, A20542890 solution:
        Initial state: {initState}

        Simulated Annealing:
        Command Line Parameters: {filename} 1 {initTemperature} {cooling_schedule}
        Initial path cost: {init_path_cost:.2f}
        Initial solution: {[i[0] for i in initSolution]}
        Final solution: {[f[0] for f in cur_solution]}
        Number of iterations: {iterations}
        Execution time: {total_time:.2f} seconds
        Complete path cost: {path_cost:.2f}''')
    base_filename = os.path.splitext(filename)[0]
    output_filename = f"{base_filename}_SOLUTION_SA.csv"
    with open(output_filename, mode='w', newline='') as file:
        writer = csv.writer(file, delimiter='\n')
        writer.writerow([formatted_cost])
        writer.writerow([i[0] for i in cur_solution])


'''Genetic Algorithm'''


# Fitness function
def fitness(individual):
    # Assuming individual is a list of tuples (e.g., [('A', 10), ('B', 20), ...])
    # print("individual", individual)
    # x = sum(value for _, value in individual if isinstance(value, int))
    print("individual", individual)
    return sum(value for _, value in individual) ** 2


# Selection mechanism - Roulette Wheel
def roulette_wheel(population: List[List[int]], fitnesses: List[float]) -> List[int]:
    total_fitness = sum(fitnesses)
    select = random.uniform(0, total_fitness)
    current = 0
    for individual, fitness in zip(population, fitnesses):
        current += fitness
        if current > select:
            return individual


# 2-point crossover mechanism
def two_point_crossover(parent1, parent2):
    size = len(parent1)
    # Ensure the crossover points exclude the first and last city
    c_point1, c_point2 = sorted(random.sample(range(1, size - 1), 2))

    # Initialize children with None to facilitate unique city inclusion
    child1, child2 = [None] * size, [None] * size

    # Include the start and end city
    child1[0], child1[-1] = parent1[0], parent1[0]
    child2[0], child2[-1] = parent2[0], parent2[0]

    # Copy the segments between c_point1 and c_point2 from each parent to the corresponding child
    child1[c_point1:c_point2] = parent1[c_point1:c_point2]
    child2[c_point1:c_point2] = parent2[c_point1:c_point2]

    # Filling in the remaining cities from the other parent, ensuring no duplicates
    def fill_child(child, parent, end):
        fill_pos = end
        for city in parent[end:] + parent[1:end]:
            if city not in child:
                while child[fill_pos % size] is not None:
                    fill_pos += 1
                child[fill_pos % size] = city

    fill_child(child1, parent2, c_point2)
    fill_child(child2, parent1, c_point2)

    # Ensuring the start and end city are the same to complete the Hamiltonian cycle
    child1[-1], child2[-1] = child1[0], child2[0]

    return child1, child2


# Mutation function
def mutation(member, pm):
    # for i in range(len(member)):
    #     if random.random() < pm:
    #         city, value = member[i]
    #         member[i] = (city, random.randint(0, 40)
    # return member

    # for i in range(len(member)):
    #     if random.random() < pm:
    #         city, value = member[i]
    #         member[i] = (city, random.randint(0, 40))
    # return member

    # for i in range(1, len(member) - 1):  # Exclude the first and last element
    #     if random.random() < pm:
    #         city, value = member[i]
    #         member[i] = (city, random.randint(0, 40))
    # return member

    # for i in range(1, len(member)-1):  # Exclude the first and last element for mutation
    #     if random.random() < pm:
    #         swap_with = random.randint(1, len(member)-2)  # Ensure we don't swap the first/last city
    #         member[i], member[swap_with] = member[swap_with], member[i]
    # return member

    for _ in range(len(member)):
        if random.random() < pm:
            idx1, idx2 = random.sample(range(1, len(member) - 1), 2)  # Exclude start/end city
            member[idx1], member[idx2] = member[idx2], member[idx1]
    return member


# def shuffle_list(some_list):
#     randomized_list = some_list[:]
#     n = len(randomized_list)
#     shuffled = False
#     while not shuffled:
#         random.shuffle(randomized_list)
#         shuffled = all(a != b for a, b in zip(some_list, randomized_list))
#     return randomized_list
#
#
# def shuffle_tuples_in_list(some_list):
#     return [shuffle_list([t for t in sublist]) for sublist in some_list]


def generate_population(city_names, n):
    # # Making start and end city same forming Hamiltonian cycle
    # start_end_city = city_names[0]
    # shuffled_cities = shuffle_list(city_names[1:-1])
    # population = [[start_end_city] + shuffled_cities + [start_end_city] for _ in range(n)]
    # shuffled_population = shuffle_tuples_in_list(population)
    # return shuffled_population
    population = []
    for _ in range(n):
        shuffled_cities = city_names[1:-1]  # Exclude the start/end city for shuffling
        random.shuffle(shuffled_cities)
        individual = [city_names[0]] + shuffled_cities + [city_names[0]]  # Re-add the start/end city
        population.append(individual)
    return population


# Genetic Algorithm main code
def geneticAlgorithm(n: int, pc: float, num_iterations: int, pm: float, coordinates: List[Tuple[str, float, float]],
                     filename: str):
    start_time = time.time()
    coordinates.append(coordinates[0])
    initState = coordinates[0][0]
    print('initial state:', initState)
    init_path_cost = calculate_distance(coordinates)
    print('initial path cost:', init_path_cost)

    # Create initial population
    city_names = [(city[0], i) for i, city in enumerate(coordinates)]

    population = generate_population(city_names, n)
    print("population", population)
    for i in range(num_iterations):
        # Calculating fitness for each individual in the population
        fitnesses = [fitness(individual) for individual in population]
        parent1 = roulette_wheel(population, fitnesses)
        parent2 = roulette_wheel(population, fitnesses)

        # Perform 2-point crossover
        if random.random() < pc:
            child1, child2 = two_point_crossover(parent1, parent2)
        else:
            child1, child2 = parent1, parent2
        # Perform mutation
        child1 = mutation(child1, pm)
        child2 = mutation(child2, pm)

        # least_fit_indices = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i])[:2]
        # population[least_fit_indices[0]] = child1
        # population[least_fit_indices[1]] = child2
        # # Replace parents with new children
        population.remove(parent1)
        # Removing parent2 only if it is distinct
        if parent1 != parent2:
            population.remove(parent2)
        population.append(child1)
        population.append(child2)
    best_individual = max(population, key=fitness)
    print("best individual", best_individual)
    path_cost = calculate_distance2(best_individual, coordinates)
    formatted_cost = f"{path_cost:.2f}"
    coord_dict = {coord[0]: coord[1:] for coord in coordinates}
    final_population = best_individual[:]
    end_time = time.time()
    val = [coord_dict[i[0]] for i in best_individual if i[0] in coord_dict]
    x_val = [v[0] for v in val]
    y_val = [v[1] for v in val]
    plt.plot(x_val, y_val, marker='o')
    plt.title("Genetic Algorithm Graph\n"
              "(Close this window to display the final results in terminal)")
    plt.show()
    total_time = end_time - start_time
    print(f'''Somanagoudar, Saikiran, A20542890 solution:
            Initial state: {initState}

            Genetic Algorithm:
            Command Line Parameters: {filename} 2 {num_iterations} {pm}
            Initial path cost: {init_path_cost:.2f}
            Initial solution: {[i[0] for i in coordinates]}
            Final solution: {[f[0] for f in final_population]}
            Number of iterations: {num_iterations}
            Execution time: {total_time:.2f} seconds
            Complete path cost: {path_cost:.2f}''')
    base_filename = os.path.splitext(filename)[0]
    output_filename = f"{base_filename}_SOLUTION_GA.csv"
    with open(output_filename, mode='w', newline='') as file:
        writer = csv.writer(file, delimiter='\n')
        writer.writerow([formatted_cost])
        writer.writerow([i[0] for i in final_population])


# Checking input arguments
if len(sys.argv) != 5:
    print("ERROR: Not enough or too many input arguments.")
    exit()
else:
    filename = sys.argv[1]
    data = pd.read_csv(filename, header=None)
    coordinates = data.iloc[:, 0:3].values.tolist()
    p1 = int(sys.argv[3])
    p2 = float(sys.argv[4])

if sys.argv[2] == '1':
    simulatedAnnealing(p1, p2, coordinates, filename)
elif sys.argv[2] == '2':
    geneticAlgorithm(8, 1, p1, p2, coordinates, filename)
