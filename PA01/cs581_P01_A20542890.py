import pandas as pd
import random
import math
import sys
import os
import csv
import time
import matplotlib.pyplot as plt
from typing import List, Tuple

'''Simulated Annealing Algorithm'''


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


# Rescaling fitness function values
def rescale_values(values):
    # Shifting values to ensure the minimum is 0
    min_value = min(values)
    shifted_values = [value - min_value for value in values]

    # Finding the new maximum value in the shifted dataset
    max_shifted = max(shifted_values)

    # Avoiding division by zero in case all values are the same
    if max_shifted == 0:
        return [0 for _ in values]  # or [100] if you prefer all max

    # Rescaling values to the range 0 to 100
    rescaled_values = [(value / max_shifted) * 100 for value in shifted_values]

    return rescaled_values


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
    fit_values = []
    while temperature > 0:
        iterations += 1
        # Generating new solution by swapping 2 edges in the current solution in every loop
        new_solution = cur_solution[:]
        edge1, edge2 = random.sample(range(1, len(new_solution)), 2)
        new_solution[edge1:edge2] = reversed(new_solution[edge1:edge2])

        # Fitness function calculation - difference between current solution and the new solution
        distance_difference = calculate_distance(cur_solution) - calculate_distance(new_solution)
        fit_values.append(distance_difference)
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

        # Temperature cools down exponentially at the rate of cooling schedule/cooling rate
        temperature *= math.exp(-iterations * cooling_schedule)

    # Rescaling fitness values
    rescaled = rescale_values(fit_values)

    # PLotting the graph
    i = range(1, iterations + 1)
    plt.plot(i, rescaled, label='Fitness', color='blue', marker='o')

    path_cost = calculate_distance(cur_solution)
    formatted_cost = f"{path_cost:.2f}"

    # Execution time end
    end_time = time.time()

    plt.title("Fitness Graph of Simulated Annealing\n"
              "(Close this window to display the final results in terminal)")
    plt.legend()
    plt.grid(True)
    plt.xlabel('Iterations')
    plt.ylabel('Fitness values')
    plt.show()

    # Calculating total execution time
    total_time = end_time - start_time

    # Displaying results in the terminal
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

    # Writing the results to a csv file
    base_filename = os.path.splitext(filename)[0]
    output_filename = f"{base_filename}_SOLUTION_SA.csv"
    with open(output_filename, mode='w', newline='') as file:
        writer = csv.writer(file, delimiter='\n')
        writer.writerow([formatted_cost])
        writer.writerow([i[0] for i in cur_solution])


'''Genetic Algorithm'''


# Calculating fitness function
def fitness(individual, coordnates):
    return calculate_distance2(individual, coordnates) ** 2


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
    # Ensuring the crossover points exclude the first and last city
    c_point1, c_point2 = sorted(random.sample(range(1, size - 1), 2))

    # Initializing children with None to facilitate unique city inclusion
    child1, child2 = [None] * size, [None] * size

    # Including the start and end city
    child1[0], child1[-1] = parent1[0], parent1[0]
    child2[0], child2[-1] = parent2[0], parent2[0]

    # Copying the segments between c_point1 and c_point2 from each parent to the corresponding child
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
    for _ in range(len(member)):
        if random.random() < pm:
            idx1, idx2 = random.sample(range(1, len(member) - 1), 2)  # Exclude start/end city
            member[idx1], member[idx2] = member[idx2], member[idx1]
    return member


# Generating population from the input of city names and number of population
def generate_population(city_names, n):
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
    # Execution time start
    start_time = time.time()
    coordinates.append(coordinates[0])
    initState = coordinates[0][0]
    init_path_cost = calculate_distance(coordinates)

    # Creating initial population
    city_names = [(city[0], i) for i, city in enumerate(coordinates)]

    population = generate_population(city_names, n)
    min_fit_values, max_fit_values, avg_fit_values = [], [], []
    for i in range(num_iterations):
        # Calculating fitness for each individual in the population
        fitnesses = [fitness(individual, coordinates) for individual in population]

        min_fit_values.append(min(fitnesses))
        max_fit_values.append(max(fitnesses))
        avg_fit_values.append(sum(fitnesses) / len(fitnesses))

        parent1 = roulette_wheel(population, fitnesses)
        parent2 = roulette_wheel(population, fitnesses)

        # Performing 2-point crossover if the probability of crossover is greater than some random number between 0 and 1
        if random.random() < pc:
            child1, child2 = two_point_crossover(parent1, parent2)
        else:
            child1, child2 = parent1, parent2

        # Performing mutation
        child1 = mutation(child1, pm)
        child2 = mutation(child2, pm)

        # Implementing elitism
        least_fit_indices = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i])[:2]
        population[least_fit_indices[0]] = child1
        population[least_fit_indices[1]] = child2

    # Assigning the best individual data to the best_individual variable
    best_individual = max(population, key=lambda x: fitness(x, coordinates))

    # Calculating final path cost
    path_cost = calculate_distance2(best_individual, coordinates)
    formatted_cost = f"{path_cost:.2f}"

    # Capturing final solution
    final_solution = best_individual[:]

    # Plotting the fitness graph
    iterations = range(1, num_iterations + 1)
    plt.figure(figsize=(12, 8))
    plt.plot(iterations, min_fit_values, label='Min Fitness', color='red', marker='o')
    plt.plot(iterations, max_fit_values, label='Max Fitness', color='green', marker='s')
    plt.plot(iterations, avg_fit_values, label='Average Fitness', color='blue', marker='x')
    plt.xlabel('Iterations')
    plt.ylabel('Fitness values')

    plt.title('Fitness Graph of Genetic Algorithm\n'
              '(Close this window to display the final results in terminal)')
    plt.legend()
    plt.grid(True)
    # Execution time end
    end_time = time.time()
    plt.show()

    # Calculating total execution time
    total_time = end_time - start_time

    # Display the results in terminal
    print(f'''Somanagoudar, Saikiran, A20542890 solution:
            Initial state: {initState}

            Genetic Algorithm:
            Command Line Parameters: {filename} 2 {num_iterations} {pm}
            Initial path cost: {init_path_cost:.2f}
            Initial solution: {[i[0] for i in coordinates]}
            Final solution: {[f[0] for f in final_solution]}
            Number of iterations: {num_iterations}
            Execution time: {total_time:.2f} seconds
            Complete path cost: {path_cost:.2f}''')
    base_filename = os.path.splitext(filename)[0]
    output_filename = f"{base_filename}_SOLUTION_GA.csv"
    with open(output_filename, mode='w', newline='') as file:
        writer = csv.writer(file, delimiter='\n')
        writer.writerow([formatted_cost])
        writer.writerow([i[0] for i in final_solution])


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
    geneticAlgorithm(50, 1, p1, p2, coordinates, filename)
