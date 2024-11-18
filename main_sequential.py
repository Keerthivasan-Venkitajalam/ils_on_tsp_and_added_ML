import tsplib95
import random
import time
import pathlib
import sys


class SequentialExecution:

    def __init__(self, instance, max_iterations):
        self.instance = instance
        self.iteration_max = max_iterations

        self.problem = tsplib95.load(f'{pathlib.Path(__file__).parent.resolve()}/instances/{instance}.tsp')
        self.problem.get_graph()
        self.problem_description = self.problem.as_keyword_dict()

        self.cities = [i - 1 for i in list(self.problem.get_nodes())]
        self.n_cities = len(self.cities)
        self.coord_x = []
        self.coord_y = []
        self.load_coordinates()

        self.best_solution = None
        self.best_cost = None

    def load_coordinates(self):
        if self.problem_description['EDGE_WEIGHT_TYPE'] == 'EUC_2D':
            for i in range(1, self.n_cities + 1):
                x, y = self.problem_description['NODE_COORD_SECTION'][i]
                self.coord_x.append(x)
                self.coord_y.append(y)

    def distance_cities(self, city_a, city_b):
        coord = city_a + 1, city_b + 1
        return self.problem.get_weight(*coord)

    def route_cost(self, cities):
        cost = 0
        i = 0
        while i < self.n_cities - 1:
            cost += self.distance_cities(cities[i], cities[i + 1])
            i += 1
        cost += self.distance_cities(cities[-1], cities[0])
        return cost

    def nearest_neighbour(self, initial_city):
        solution_cities = [initial_city]
        current_city = initial_city

        visited = [False] * self.n_cities
        visited[current_city] = True

        while len(solution_cities) < self.n_cities:
            min_distance = 9999999
            next_city = None

            for candidate_city in range(self.n_cities):
                if not visited[candidate_city] and candidate_city != current_city:
                    cost = self.distance_cities(current_city, candidate_city)
                    if cost < min_distance:
                        min_distance = cost
                        next_city = candidate_city

            solution_cities.append(next_city)
            visited[next_city] = True
            current_city = next_city

        return solution_cities

    def two_opt_local_search(self, cities):
        best_route = cities[:]
        best_cost = self.route_cost(best_route)
        improved = False
        for i in range(1, self.n_cities - 2):
            for j in range(i + 1, self.n_cities):
                if j - i == 1:
                    continue
                current_route = cities[:]
                current_route[i:j] = current_route[j - 1:i - 1:-1]
                new_cost = self.route_cost(current_route)

                if new_cost < best_cost:
                    best_cost = new_cost
                    best_route = current_route
                    improved = True

        if improved:
            return best_route[:]
        else:
            return cities

    def perturbation(self, cities):
        i = random.randint(0, self.n_cities - 1)
        j = 0
        if i != self.n_cities - 1 and i != j:
            j = i + 1

        temp = cities[i]
        cities[i] = cities[j]
        cities[j] = temp
        return cities

    def ils(self):
        start_time = time.time()

        # Initial Solution
        initial_city = random.randint(0, self.n_cities)
        # initial_city = 1
        # print(f"Initial City: {initial_city}")

        solution = self.nearest_neighbour(initial_city)
        # cost = self.route_cost(solution)
        # print(f"Initial Route: {solution} - Initial Route Cost: {cost}")

        # Local Search
        solution = self.two_opt_local_search(solution)
        cost = self.route_cost(solution)

        self.best_solution = solution
        self.best_cost = cost

        # Iterations
        count_no_improvement = 0
        for iteration in range(self.iteration_max):
            print(f"Iteration:{iteration + 1} - Best Cost:{self.best_cost} - No Improvement: {count_no_improvement}")

            inital_cost = self.best_cost

            # Perturbation
            solution = self.perturbation(solution)
            pertubation_cost = self.route_cost(solution)

            if pertubation_cost < self.best_cost:
                count_no_improvement = 0
                self.best_cost = pertubation_cost
                self.best_solution = solution[:]

            # Local Search
            solution = self.two_opt_local_search(solution)
            local_search_cost = self.route_cost(solution)

            if local_search_cost < self.best_cost:
                count_no_improvement = 0
                self.best_cost = local_search_cost
                self.best_solution = solution[:]

            if self.best_cost == inital_cost:
                count_no_improvement = count_no_improvement + 1

            if abs(self.best_cost - local_search_cost) / self.best_cost > 0.01:
                self.best_solution = solution[:]

        end_time = time.time()
        total_time = end_time - start_time

        print("---------------------------------------------")
        print("Route Cost  : %d" % self.best_cost)
        print("Processing Time : %f" % total_time)
        print(f"Route Solution {self.best_solution}")
        f = open(f"{pathlib.Path(__file__).parent.resolve()}/results/results_sequential_{self.instance}.xlsx", "a")
        f.write(f"{initial_city},{self.best_cost},{total_time}\n")
        f.close()

    def run(self):
        self.ils()


def main():
    instance = sys.argv[1]
    max_iterations = int(sys.argv[2])
    print("Graph {} - {} Iterations\n".format(instance, max_iterations))
    ex1 = SequentialExecution(instance, max_iterations)
    ex1.run()


if __name__ == "__main__":
    main()
