import pandas as pd
import random
import math
import sys
import matplotlib.pyplot as plt
# import os
#
# os.rename('main.py', 'cs581_P01_A20542890.py')

# Checking input arguments
if len(sys.argv) != 5:
    print("ERROR: Not enough or too many input arguments.")
    exit()

file = sys.argv[1]
data = pd.read_csv(file, header=None)

# Storing the data in 3 different lists - state_space, x_cord, y_cord
state_space = data.iloc[:, 0].values
coordinates = data.iloc[:, 1:3].values.tolist()


# Calculating distance function
def calculate_distance(solution):
    t_distance = 0
    last_distance = 0
    for i in range(1, len(solution)):
        x1, y1 = solution[i - 1]
        x2, y2 = solution[i]
        t_distance += math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    # x1, y1 = solution[0]
    # x2, y2 = solution[-1]
    # last_distance += math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    # solution.append(solution[0])
    # t_distance += last_distance
    return t_distance


# Simulated Annealing code
def simulatedAnnealing(temperature, cooling_schedule):
    # Defining initial solution in random with cur_solution variable
    cur_solution = random.sample(coordinates, len(coordinates))
    cur_solution.append(cur_solution[0])
    print("initial coordinates: ", cur_solution[0])
    print("initial solution:", cur_solution)
    print("Prev distance:", calculate_distance(cur_solution))
    plt.ion()
    while temperature > 0:
        # Generating solution by swapping 2 edges in the current solution
        new_solution = cur_solution[:]
        edge1, edge2 = random.sample(range(1, len(new_solution)-1), 2)
        new_solution[edge1:edge2] = reversed(new_solution[edge1:edge2])
        print("2 edge swap:", new_solution[edge1:edge2])
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
        plt.clf()  # Clear the current figure
        x_val = [coord[0] for coord in cur_solution]
        y_val = [coord[1] for coord in cur_solution]
        plt.plot(x_val, y_val, marker='o')
        # Add labels
        plt.xlabel('X values')
        plt.ylabel('Y values')
        plt.draw()  # Redraw the current figure
        plt.pause(0.01)
    print("Final solution:", cur_solution)
    print("Final distance:", calculate_distance(cur_solution))
    plt.ioff()
    plt.show()


if sys.argv[3] == '1':
    # Retrieving initial temperature from the command line argument P1
    temperature = float(sys.argv[2])
    # Retrieving alpha parameter from the command line argument for temperature cooling schedule P2
    cooling_schedule = float(sys.argv[4])
    simulatedAnnealing(temperature, cooling_schedule)
