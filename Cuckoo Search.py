import numpy as np

def objective_function(x):
    return sum(x**2)

def random_step():
    return np.random.uniform(-1, 1, size=2)

def cuckoo_search(num_nests, num_iterations, pa):
    nests = np.random.rand(num_nests, 2) * 10 - 5
    fitness = np.array([objective_function(nest) for nest in nests])

    best_nest = nests[np.argmin(fitness)]
    best_fitness = min(fitness)

    for _ in range(num_iterations):
        for i in range(num_nests):
            new_nest = nests[i] + random_step()
            new_fitness = objective_function(new_nest)

            if new_fitness < fitness[i]:
                nests[i] = new_nest
                fitness[i] = new_fitness

        num_abandon = int(pa * num_nests)
        worst_indices = np.argsort(fitness)[-num_abandon:]

        for j in worst_indices:
            nests[j] = np.random.rand(2) * 10 - 5
            fitness[j] = objective_function(nests[j])

        current_best_index = np.argmin(fitness)
        if fitness[current_best_index] < best_fitness:
            best_fitness = fitness[current_best_index]
            best_nest = nests[current_best_index]

    return best_nest, best_fitness

num_nests = 25
num_iterations = 100
pa = 0.25

best_nest, best_fitness = cuckoo_search(num_nests, num_iterations, pa)

print("Best Nest:", best_nest)
print("Best Fitness:", best_fitness)
