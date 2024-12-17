import random

POPULATION_SIZE = 100
MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.8
NUM_GENERATIONS = 50
RANGE_MIN = -10
RANGE_MAX = 10

def fitness_function(x):
    return x**2

def create_initial_population():
    return [random.uniform(RANGE_MIN, RANGE_MAX) for _ in range(POPULATION_SIZE)]

def evaluate_population(population):
    return [fitness_function(individual) for individual in population]

def select_parents(population, fitness_scores):
    total_fitness = sum(fitness_scores)
    if total_fitness == 0:
        return random.choice(population), random.choice(population)

    selection_probs = [fitness / total_fitness for fitness in fitness_scores]
    parent1 = random.choices(population, weights=selection_probs, k=1)[0]
    parent2 = random.choices(population, weights=selection_probs, k=1)[0]
    return parent1, parent2

def crossover(parent1, parent2):
    if random.random() < CROSSOVER_RATE:
        alpha = random.random()
        offspring1 = alpha * parent1 + (1 - alpha) * parent2
        offspring2 = alpha * parent2 + (1 - alpha) * parent1
        return offspring1, offspring2
    return parent1, parent2

def mutate(individual):
    if random.random() < MUTATION_RATE:
        return random.uniform(RANGE_MIN, RANGE_MAX)
    return individual

def genetic_algorithm():
    population = create_initial_population()
    best_solution = None
    best_fitness = float('-inf')

    for generation in range(NUM_GENERATIONS):
        fitness_scores = evaluate_population(population)

        for i in range(POPULATION_SIZE):
            if fitness_scores[i] > best_fitness:
                best_fitness = fitness_scores[i]
                best_solution = population[i]

        new_population = []

        while len(new_population) < POPULATION_SIZE:
            parent1, parent2 = select_parents(population, fitness_scores)
            offspring1, offspring2 = crossover(parent1, parent2)
            offspring1 = mutate(offspring1)
            offspring2 = mutate(offspring2)
            new_population.extend([offspring1, offspring2])

        population = new_population[:POPULATION_SIZE]

        print(f"Generation {generation + 1}: Best Fitness = {best_fitness}, Best Solution = {best_solution}")

    print("\nBest Solution Found:")
    print(f"x = {best_solution}, f(x) = {best_fitness}")

genetic_algorithm()
