import numpy as np


class ES_STRATEGY:
    def __init__(
        self,
        strategy,
        fitness_func,
        dimension,
        population_size_mu,
        offspring_size_lambda,
        sigma,
        max_generations,
    ):
        self.strategy = strategy
        self.fitness_func = fitness_func
        self.dimension = dimension
        self.population_size_mu = population_size_mu
        self.offspring_size_lambda = offspring_size_lambda
        self.sigma = sigma
        self.max_generations = max_generations
        self.best_fitnesses = []
        self.parents = np.random.uniform(size=(population_size_mu, dimension))

    def generate_offspring(self):
        # Initialize an empty list to store the offspring
        offspring = []

        # If the strategy is (μ, λ)-ES or (μ+λ)-ES
        if self.strategy in ["mu_lambda", "mu_plus_lambda"]:
            # Generate λ offspring
            for _ in range(self.offspring_size_lambda):
                # Select a random parent
                parent_idx = np.random.randint(self.population_size_mu)
                # Generate an offspring by adding Gaussian noise to the parent

                offspring_candidate = self.parents[parent_idx] + np.random.normal(
                    0, self.sigma, self.dimension
                )
                # Add the offspring to the list
                # Add the offspring to the list
                offspring.append(offspring_candidate)

        # If the strategy is (1+1)-ES
        elif self.strategy in ["one_plus_one", "one_plus_one_1_fifth"]:
            # Generate an offspring by adding Gaussian noise to the first parent
            offspring_candidate = self.parents[0] + np.random.normal(
                0, self.sigma, self.dimension
            )
            # Add the offspring to the list
            offspring.append(offspring_candidate)

        # If the strategy is (μ/μ, λ)-ES
        elif self.strategy == "mu_over_mu_lambda":
            # Generate λ offspring
            for _ in range(self.offspring_size_lambda):
                # Select two random parents
                parents_indices = np.random.choice(
                    self.population_size_mu, 2, replace=False
                )
                # Generate an offspring by averaging the parents
                offspring_candidate = np.mean(self.parents[parents_indices], axis=0)
                # Add the offspring to the list
                offspring.append(offspring_candidate)

        # Return the list of offspring
        return offspring

    def evaluate_fitness(self, offspring):
        return [self.fitness_func(ind) for ind in offspring]

    def run(self):
        print("\nUsing strategy", self.strategy)
        success_count = 0
        for gen in range(self.max_generations):
            offspring = self.generate_offspring()
            offspring_fitnesses = self.evaluate_fitness(offspring)

            # In the (μ, λ)-ES strategy, only the offspring are considered for the next generation.
            if self.strategy == "mu_lambda":
                # Get the indices that would sort the offspring fitnesses in descending order.
                sorted_indices = np.argsort(offspring_fitnesses)[::-1]
                # Select the best μ individuals from the offspring to become the new parents.
                self.parents = np.array(offspring)[
                    sorted_indices[: self.population_size_mu]
                ]

            # In the (μ+λ)-ES strategy, the parents and the offspring are combined,
            # and the best μ individuals are selected as the new parents.
            elif self.strategy == "mu_plus_lambda":
                combined_population = np.concatenate((self.parents, offspring))
                combined_fitnesses = np.concatenate(
                    (self.evaluate_fitness(self.parents), offspring_fitnesses)
                )
                # Get the indices that would sort the combined fitnesses in descending order.
                sorted_indices = np.argsort(combined_fitnesses)[::-1]
                # Select the best μ individuals from the combined population to become the new parents.
                self.parents = combined_population[
                    sorted_indices[: self.population_size_mu]
                ]

            best_fitness = self.evaluate_fitness(self.parents)[0]
            self.best_fitnesses.append(best_fitness)

            # Apply the 1/5th success rule only for the (1+1)-ES strategy
            if self.strategy == "one_plus_one_1_fifth":
                # Check if the best offspring is better than the best parent
                if gen > 0 and best_fitness > self.best_fitnesses[-2]:
                    success_count += 1

                # Adjust sigma according to the 1/5th success rule
                if gen % self.offspring_size_lambda == 0:
                    if success_count > self.offspring_size_lambda / 5:
                        self.sigma *= 1.2
                    else:
                        self.sigma /= 1.2
                    success_count = 0
            if gen % 10 == 0:
                print(f"Generation {gen}, best fitness: {best_fitness}")

        best_solution = self.parents[0]
        return best_solution, best_fitness, self.best_fitnesses
