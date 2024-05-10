import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt


# Define the TSP problem
class TravelingSalesman:
    def __init__(self, cities):
        self.cities = np.array(cities)
        self.distances = np.round(squareform(pdist(self.cities * 10))).astype(int)

    def print_cities(self):
        print("\nCities:")
        print(self.cities)

    def print_distances(self):
        df = pd.DataFrame(self.distances)
        print("\n")
        print("\nDistance Among Cities:")
        print(df)

    def calculate_total_distance(self, tour):
        total_distance = 0
        print("\n")
        for i in range(len(tour) - 1):
            distance = self.distances[tour[i], tour[i + 1]]
            total_distance += distance
            # print(
            #     f"Travelling from city {tour[i]} to city {tour[i + 1]} = {distance:.2f}km"
            # )
        return total_distance

    def plot_cities(self):
        """
        Create a scatter plot of the cities.
        """
        plt.scatter(self.cities[:, 0], self.cities[:, 1])

        # Annotate the plot with the city numbers
        for i in range(len(self.cities)):
            plt.text(self.cities[i, 0], self.cities[i, 1], f"City {i+1}")
        plt.show()

    def travel_cities_plot(self, tour, title):
        """
        Visualize a tour of the cities.

        The function creates a scatter plot of the cities and draws lines between them according to the order specified by the tour.
        It also annotates the plot with the city numbers and the distances between consecutive cities.
        """
        # Convert cities to a NumPy array for easier indexing
        cities_array = np.array(self.cities)

        # Create a scatter plot of the cities
        plt.scatter(cities_array[:, 0], cities_array[:, 1])

        # Annotate the plot with the city numbers
        for i in range(len(cities_array)):
            plt.text(cities_array[i, 0], cities_array[i, 1], f"City {i}")

        # Draw lines between the cities and annotate them with the distances
        for i in range(len(tour) - 1):
            plt.plot(
                [cities_array[tour[i], 0], cities_array[tour[i + 1], 0]],
                [cities_array[tour[i], 1], cities_array[tour[i + 1], 1]],
                "k-",
            )
            mid_point = (cities_array[tour[i]] + cities_array[tour[i + 1]]) / 2
            plt.text(
                mid_point[0],
                mid_point[1],
                f"{self.distances[tour[i], tour[i+1]]:d}km",
                ha="center",
            )

        # Draw line between the last and the first city
        plt.plot(
            [cities_array[tour[-1], 0], cities_array[tour[0], 0]],
            [cities_array[tour[-1], 1], cities_array[tour[0], 1]],
            "k-",
        )
        mid_point = (cities_array[tour[-1]] + cities_array[tour[0]]) / 2
        plt.text(
            mid_point[0],
            mid_point[1],
            f"{self.distances[tour[-1], tour[0]]:d}km",
            ha="center",
        )
        plt.title(title)
        plt.show()
