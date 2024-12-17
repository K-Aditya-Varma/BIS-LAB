import numpy as np

class Particle:
    def __init__(self, dim, minx, maxx):
        self.position = np.random.uniform(minx, maxx, dim)
        self.velocity = np.random.uniform(-1, 1, dim)
        self.bestPos = self.position.copy()
        self.bestFitness = float('inf')

def objective_function(x):
    # Example: Sphere function (sum of squares)
    return np.sum(x**2)

def pso(d, minx, maxx, N, max_iter, w, c1, c2):
    # Initialize swarm
    swarm = [Particle(d, minx, maxx) for _ in range(N)]
    best_pos_swarm = None
    best_fitness_swarm = float('inf')

    # Lists to store the best positions and fitness values at each iteration
    best_positions = []
    best_fitnesses = []

    for iteration in range(max_iter):
        for particle in swarm:
            # Calculate fitness
            particle.fitness = objective_function(particle.position)

            # Update personal best
            if particle.fitness < particle.bestFitness:
                particle.bestFitness = particle.fitness
                particle.bestPos = particle.position.copy()

            # Update global best
            if particle.fitness < best_fitness_swarm:
                best_fitness_swarm = particle.fitness
                best_pos_swarm = particle.position.copy()

        # Store the best position and fitness for the current iteration
        best_positions.append(best_pos_swarm)
        best_fitnesses.append(best_fitness_swarm)

        # Update velocity and position for each particle
        for particle in swarm:
            r1, r2 = np.random.rand(2)
            particle.velocity = (w * particle.velocity +
                                 r1 * c1 * (particle.bestPos - particle.position) +
                                 r2 * c2 * (best_pos_swarm - particle.position))

            # Update position
            particle.position += particle.velocity

            # Clip position within bounds
            particle.position = np.clip(particle.position, minx, maxx)

    return best_positions, best_fitnesses, best_pos_swarm, best_fitness_swarm

# Parameters
d = 2  # Number of dimensions
minx = -10  # Lower bound
maxx = 10   # Upper bound
N = 30      # Number of particles
max_iter = 10  # Maximum number of iterations
w = 0.5     # Inertia weight
c1 = 1.5    # Cognitive coefficient
c2 = 1.5    # Social coefficient
best_positions, best_fitnesses, final_best_position, final_best_fitness = pso(d, minx, maxx, N, max_iter, w, c1, c2)

# Display results
for i in range(max_iter):
    print(f"Iteration {i + 1}: Best Position = {best_positions[i]}, Best Fitness = {best_fitnesses[i]}")

print(f"\nFinal Best Position: {final_best_position}")
print(f"Final Best Fitness: {final_best_fitness}")
