import pandas as pd
import random
import math
import sys
import os
import csv
import time
import matplotlib.pyplot as plt
from typing import List, Tuple, Union

'''Simulated Annealing algorithm'''


# Calculating distance function
def calculate_distance(solution: List[Tuple[str, float, float]]) -> float:
    t_distance = 0
    # last_distance = 0
    for i in range(1, len(solution)):
        x1, y1, z1 = solution[i - 1]
        x2, y2, z2 = solution[i]
        t_distance += math.sqrt((y2 - y1) ** 2 + (z2 - z1) ** 2)
    # x1, y1 = solution[0]
    # x2, y2 = solution[-1]
    # last_distance += math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    # solution.append(solution[0])
    # t_distance += last_distance
    return t_distance


# Simulated Annealing main code
def simulatedAnnealing(temperature: int, cooling_schedule: float, coordinates: List[Tuple[str, float, float]], filename: str):
    start_time = time.time()
    iterations = 0
    # Defining initial solution in random with cur_solution variable
    # cur_solution = random.sample(coordinates, len(coordinates))
    cur_solution = coordinates
    cur_solution.append(cur_solution[0])
    init_path_cost = calculate_distance(cur_solution)
    initState = cur_solution[0][0]
    initSolution = cur_solution[:]
    initTemperature = temperature
    # print("initial coordinates: ", cur_solution[0])
    # print("initial solution:", cur_solution)
    # print("Prev distance:", calculate_distance(cur_solution))
    # plt.ion()
    while temperature > 0:
        # Generating solution by swapping 2 edges in the current solution
        new_solution = cur_solution[:]
        edge1, edge2 = random.sample(range(1, len(new_solution)), 2)
        new_solution[edge1:edge2] = reversed(new_solution[edge1:edge2])
        # print("2 edge swap:", new_solution[edge1:edge2])
        distance_difference = calculate_distance(cur_solution) - calculate_distance(new_solution)
        # if distance_difference < 0 or random.random() < math.exp(-distance_difference / temperature):
        #     cur_solution = new_solution[:]
        try:
            acceptance_probability = math.exp(-distance_difference / temperature)
        except OverflowError:
            acceptance_probability = 0.0
        if distance_difference > 0:
            cur_solution = new_solution[:]
        elif acceptance_probability > random.random():
            cur_solution = new_solution[:]
        temperature *= cooling_schedule
        iterations += 1
        # plt.clf()  # Clear the current figure
        # Add labels
        # plt.xlabel('X values')
        # plt.ylabel('Y values')
        # plt.draw()  # Redraw the current figure
        # plt.pause(0.01)
    x_val = [coord[1] for coord in cur_solution]
    y_val = [coord[2] for coord in cur_solution]
    plt.plot(x_val, y_val, marker='o')
    # print("Final solution:", cur_solution)
    # print("Final state spaces", [i[0] for i in cur_solution])
    # print("Final distance:", calculate_distance(cur_solution))
    # plt.ioff()
    path_cost = calculate_distance(cur_solution)
    formatted_cost = f"{path_cost:.2f}"
    end_time = time.time()
    plt.title("Simulated Annealing graph\n"
              "(Close this window to display the final results in terminal)")
    plt.show()
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
def fitness(individual: List[Union[int, float]]) -> float:
    # fitness logic
    if all(element in [0, 1] for element in individual):
        p = 0
        for i in reversed(individual):
            x = x + individual[i] * pow(2, p)
            p += 1
    elif type(individual[0]) == float or type(individual[0]) == int:
        x = sum(individual)
    return pow(x, 2)



# Selection mechanism - Roulette Wheel
def roulette_wheel(population: List[List[Union[int, float]]], fitnesses: List[float]) -> List[Union[int, float]]:
    total_fitness = sum(fitnesses)
    select = random.uniform(0, total_fitness)
    current = 0
    for individual, fitness in zip(population, fitnesses):
        current += fitness
        if current > select:
            return individual


# 2-point crossover mechanism
def two_point_crossover(parent1: List[Union[int, float]], parent2: List[Union[int, float]]) -> Tuple[List[Union[int, float]], List[Union[int, float]]]:
    c_point1 = random.randint(0, len(parent1))
    c_point2 = random.randint(0, len(parent1))
    c_point1, c_point2 = min(c_point1, c_point2), max(c_point1, c_point2)
    child1 = parent1[:c_point1] + parent2[c_point1:c_point2] + parent1[c_point2:]
    child2 = parent2[:c_point1] + parent1[c_point1:c_point2] + parent2[c_point2:]
    return child1, child2

# Mutation function
def mutation(individual: List[Union[int, float]], Pm: float) -> List[Union[int, float]]:
    if all(element in [0, 1] for element in individual):
        for i in range(len(individual)):
            if random.random() < Pm:
                individual[i] = abs(individual[i] - 1)
    elif all(isinstance(element, int) for element in individual):
        for i in range(len(individual)):
            if random.random() < Pm:
                individual[i] = random.randint(-100, 100)
    elif all(isinstance(element, float) for element in individual):
        for i in range(len(individual)):
            if random.random() < Pm:
                individual[i] += random.gauss(0, 1)
    return individual


# Genetic Algorithm main code
def geneticAlgorithm(N: int, Pc: float, Pm: float, num_iterations: int, filename: str) -> List[Union[int, float]]:
    population = # population initialization logic
    for _ in range(num_iterations):
        # Calculating fitness for each individual in the population
        fitnesses = [fitness(individual) for individual in population]

        # Select parents
        parent1 = roulette_wheel(population, fitnesses)
        parent2 = roulette_wheel(population, fitnesses)

        # Perform crossover
        if random.random() < Pc
            child1, child2 = two_point_crossover(parent1, parent2)
        else:
            child1, child2 = parent1, parent2

        # Perform mutation
        child1 = mutation(child1, Pm)
        child2 = mutation(child2, Pm)

        #Replace parents with new children
        population.remove(parent1)
        population.remove(parent2)
        population.append(child1)
        population.append(child2)
    return max(population, key=fitness)

# Checking input arguments
if len(sys.argv) != 5:
    print("ERROR: Not enough or too many input arguments.")
    exit()

if sys.argv[2] == '1':
    # Retrieving initial temperature from the command line argument P1
    temperature = int(sys.argv[3])
    # Retrieving alpha parameter from the command line argument for temperature cooling schedule P2
    cooling_schedule = float(sys.argv[4])
    filename = sys.argv[1]
    data = pd.read_csv(filename, header=None)
    # Storing the data in a list
    coordinates = data.iloc[:, 0:3].values.tolist()
    simulatedAnnealing(temperature, cooling_schedule, coordinates, filename)
elif sys.argv[2] == '2':
    # Retrieving number of iterations from the command line argument P1
    iterations = int(sys.argv[3])
    # Retrieving probability of mutation value
    p_mutation = float(sys.argv[4])
    filename = sys.argv[1]
    data = pd.read_csv(filename, header=None)
    # Storing the data in a list
    coordinates = data.iloc[:, 0:3].values.tolist()
    geneticAlgorithm(N, 1, p_mutation, iterations, coordinates, filename)

