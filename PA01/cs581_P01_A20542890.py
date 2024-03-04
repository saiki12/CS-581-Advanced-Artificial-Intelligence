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
def fitness(individual: List[Tuple[str, int]]) -> float:
    # print(f"individual: {individual}, type: {type(individual)}")
    x = sum(i[1] for i in individual if isinstance(i[1], int))
    return pow(x, 2)


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
def two_point_crossover(parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
    c_point1 = random.randint(0, len(parent1))
    c_point2 = random.randint(0, len(parent1))
    c_point1, c_point2 = min(c_point1, c_point2), max(c_point1, c_point2)
    child1 = parent1[:c_point1] + parent2[c_point1:c_point2] + parent1[c_point2:]
    child2 = parent2[:c_point1] + parent1[c_point1:c_point2] + parent2[c_point2:]
    return child1, child2


# Mutation function
def mutation(member: List[int], pm: float) -> List[int]:
    for i in range(len(member)):
        if random.random() < pm:
            member[i] = (member[i][0], random.randint(0, 40))
    return member


def shuffle_list(some_list):
    randomized_list = some_list[1:]
    while True:
        random.shuffle(randomized_list)
        for a, b in zip(some_list, randomized_list):
            if a == b:
                break
        else:
            return randomized_list


def generate_population(city_names, n):
    # Making start and end city same forming Hamiltonian cycle
    start_end_city = city_names[0]
    shuffled_cities = shuffle_list(city_names[1:])
    population = [[start_end_city] + shuffled_cities + [start_end_city] for _ in range(n)]
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
    elitism_size = 5
    for i in range(num_iterations):
        # Calculating fitness for each individual in the population
        fitnesses = [fitness(individual) for individual in population]
        elites = sorted(population, key=fitness, reverse=True)[:elitism_size]
        new_population = []

        while len(new_population) < n - elitism_size:
            # Select parents
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
            new_population.extend([child1, child2])
            # Replace parents with new children
            # population.remove(parent1)
        population = elites + new_population
        # Removing parents only if they are distinct
        # if parent1 != parent2:
        #     population.remove(parent2)
        # # if parent1 and parent2 in population:
        # population.append(child1)
        # population.append(child2)
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
