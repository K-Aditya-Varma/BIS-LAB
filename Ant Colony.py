import numpy as np
import random

# Function to calculate the total cost of a given solution
def calculate_cost(solution, dist_matrix):
    num_cities = len(solution)
    cost = sum(dist_matrix[solution[i], solution[i+1]] for i in range(num_cities-1))
    cost += dist_matrix[solution[-1], solution[0]]  # Return to the start
    return cost

# Function to generate a solution (route) for an ant
def gen_solution(dist_matrix, pheromone, alpha, beta):
    num_cities = len(dist_matrix)
    solution = [random.randint(0, num_cities-1)]  # Start at a random city
    visited = [False] * num_cities
    visited[solution[0]] = True

    while len(solution) < num_cities:
        current = solution[-1]
        probs = [
            pheromone[current, i] ** alpha * (1 / dist_matrix[current, i]) ** beta if not visited[i] else 0
            for i in range(num_cities)
        ]
        total_prob = sum(probs)
        probs = [p / total_prob for p in probs]  # Normalize probabilities
        next_city = np.random.choice(range(num_cities), p=probs)
        solution.append(next_city)
        visited[next_city] = True
    return solution

# Function to update the pheromone matrix based on the solutions found
def update_pheromone(pheromone, solutions, dist_matrix, q, decay):
    pheromone *= (1 - decay)  # Evaporate pheromones

    for solution in solutions:
        cost = calculate_cost(solution, dist_matrix)
        deposit = q / cost  # Pheromone deposit
        for i in range(len(solution)-1):
            pheromone[solution[i], solution[i+1]] += deposit
        pheromone[solution[-1], solution[0]] += deposit  # Return to start

# Main ACO algorithm
def simple_aco(dist_matrix, num_ants, num_iterations, alpha=1, beta=2, decay=0.1, q=100):
    num_cities = len(dist_matrix)
    pheromone = np.ones((num_cities, num_cities))  # Initialize pheromone matrix

    best_solution = None
    best_cost = float('inf')

    # Main ACO loop
    for _ in range(num_iterations):
        solutions = [gen_solution(dist_matrix, pheromone, alpha, beta) for _ in range(num_ants)]  # Generate solutions
        update_pheromone(pheromone, solutions, dist_matrix, q, decay)  # Update pheromone

        for solution in solutions:
            cost = calculate_cost(solution, dist_matrix)
            if cost < best_cost:
                best_cost = cost
                best_solution = solution

    return best_solution, best_cost


# Example usage
dist_matrix = np.array([
    [0, 10, 15, 20],
    [10, 0, 35, 25],
    [15, 35, 0, 30],
    [20, 25, 30, 0]
])

best_solution, best_cost = simple_aco(dist_matrix, num_ants=5, num_iterations=100)
print("Best solution:", best_solution)
print("Best solution cost:", best_cost)
