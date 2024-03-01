import pandas as pd
import random
import math
import sys
# import os
# import csv
import time
import matplotlib.pyplot as plt


# Calculating distance function
def calculate_distance(solution):
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


# Simulated Annealing code
def simulatedAnnealing(temperature, cooling_schedule, filename, coordinates):
    start_time = time.time()
    iterations = 0
    # Defining initial solution in random with cur_solution variable
    cur_solution = random.sample(coordinates, len(coordinates))
    cur_solution.append(cur_solution[0])
    initState = cur_solution[0][0]
    initSolution = cur_solution[:]
    initTemperature = temperature
    # print("initial coordinates: ", cur_solution[0])
    # print("initial solution:", cur_solution)
    # print("Prev distance:", calculate_distance(cur_solution))
    plt.ion()
    while temperature > 0:
        # Generating solution by swapping 2 edges in the current solution
        new_solution = cur_solution[:]
        edge1, edge2 = random.sample(range(1, len(new_solution)-1), 2)
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
        plt.clf()  # Clear the current figure
        x_val = [coord[1] for coord in cur_solution]
        y_val = [coord[2] for coord in cur_solution]
        plt.plot(x_val, y_val, marker='o')
        # Add labels
        plt.xlabel('X values')
        plt.ylabel('Y values')
        plt.draw()  # Redraw the current figure
        plt.pause(0.01)
    # print("Final solution:", cur_solution)
    # print("Final state spaces", [i[0] for i in cur_solution])
    # print("Final distance:", calculate_distance(cur_solution))
    plt.ioff()
    plt.show()
    path_cost = calculate_distance(cur_solution)
    end_time = time.time()
    total_time = end_time - start_time
    print(f'''Somanagoudar, Saikiran, A20542890 solution:
        Initial state: {initState}

        Simulated Annealing:
        Command Line Parameters: {filename} 1 {initTemperature} {cooling_schedule}
        Initial solution: {[i[0] for i in initSolution]}
        Final solution: {[f[0] for f in cur_solution]}
        Number of iterations: {iterations}
        Execution time: {total_time:.2f} seconds
        Complete path cost: {path_cost:.2f}''')
    # base_filename = os.path.splitext(filename)[0]
    # output_filename = f"{base_filename}_SOLUTION_SA.csv"
    # with open(output_filename, mode='w', newline='') as file:
    #     writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)
    #     writer.writerow([i[0] for i in cur_solution])


# Checking input arguments
if len(sys.argv) != 5:
    print("ERROR: Not enough or too many input arguments.")
    exit()

if sys.argv[3] == '1':
    # Retrieving initial temperature from the command line argument P1
    temperature = int(sys.argv[2])
    # Retrieving alpha parameter from the command line argument for temperature cooling schedule P2
    cooling_schedule = float(sys.argv[4])
    filename = sys.argv[1]
    data = pd.read_csv(filename, header=None)
    # Storing the data in 3 different lists
    coordinates = data.iloc[:, 0:3].values.tolist()
    simulatedAnnealing(temperature, cooling_schedule, filename, coordinates)
