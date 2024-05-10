import numpy as np

# For using in google collab below two import needs to be commented
from travelling_salesman import TravelingSalesman
from es_strategy import ES_STRATEGY


# Create cities with random coordinates
cities = np.random.rand(36, 2)
ts = TravelingSalesman(cities)
distances = ts.distances
ts.print_cities()
ts.print_distances()
ts.plot_cities()


def fitness(x):
    """
    Define the fitness function.

    The fitness function calculates the total distance of a tour.
    The tour is the permutation of cities specified by x.
    The total distance is the sum of the distances between consecutive cities in the tour.
    Since we want to minimize the total distance, the fitness is the negative total.
    """
    tour = np.argsort(x)
    total_distance = np.sum(distances[tour[:-1], tour[1:]])
    return -total_distance


initial_tour = list(range(len(cities)))
print("\nTravelling Without optimization")

# Initial tour = [1,2,3,4,5....n]
# At the start the sales man moves city1->city2->city3->city4->city5...cityn
print("\ninitial tour", initial_tour)

total_distance = ts.calculate_total_distance(initial_tour)
print(f"\nTotal distance after optimization: {total_distance:.2f}km\n")
ts.travel_cities_plot(initial_tour, f"Without Optimization: {total_distance:.2f}km")

# Define the ES optimization parameters
# The dimension is the number of cities
DIMENSION = len(cities)

# The number of parents in the population. Each parent represents a possible cities combination.
POPULATION_SIZE_MU = 50

# The number of offspring to generate. Each offspring represents a possible cities combination.
OFFSPRING_SIZE_LAMBDA = 250

# The standard deviation of the Gaussian noise added to generate offspring.
# This controls the amount of variation in the offspring.
SIGMA = 0.1

# The maximum number of generations to run the optimization.
# The optimization stops after this many generations, even if a better solution has not been found.
MAX_GENERATIONS = 200

STRATEGY = [
    "mu_lambda",
    "mu_plus_lambda",
    "one_plus_one",
    "one_plus_one_1_fifth",
    "mu_over_mu_lambda",
]
current_strategy = STRATEGY[0]

# Run ES optimization
print(f"\nOptimizing paths using {current_strategy} ES")

optimizer = ES_STRATEGY(
    current_strategy,
    fitness,
    DIMENSION,
    POPULATION_SIZE_MU,
    OFFSPRING_SIZE_LAMBDA,
    SIGMA,
    MAX_GENERATIONS,
)
best_solution, best_fitness, best_fitnesses = optimizer.run()

print(f"\nFor {current_strategy} ES:")
print(f"Best fitness: {-best_fitness} km")


print("\nTravelling after optimization")
final_tour = list(np.argsort(best_solution))

# Final tour example: if final tour is [3,5,1,4,2]
# Sales man moves city3->city5->city1->city4->city2
print("final tour", final_tour)

total_distance = ts.calculate_total_distance(final_tour)
print(f"\nTotal distance after optimization: {total_distance:.2f}km\n")

ts.travel_cities_plot(
    final_tour, f"With Optimization {current_strategy}: {total_distance:.2f}km"
)
