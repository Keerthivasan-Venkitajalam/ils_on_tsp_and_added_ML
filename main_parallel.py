import random
import time
import threading
import pathlib
import sys
from main_sequential import SequentialExecution


class ParallelExecution(SequentialExecution):
    def __init__(self, instance, max_iterations,  n_threads):
        super().__init__(instance, max_iterations)
        self.n_threads = n_threads

        self.global_best_solution = []
        self.global_best_cost = []
        self.sem = threading.Semaphore()

    def perturbation(self, cities):
        i = random.randint(0, self.n_cities - 1)
        j = random.randint(0, self.n_cities - 1)
        if i != self.n_cities - 1 and i != j:
            j = i + 1

        temp = cities[i]
        cities[i] = cities[j]
        cities[j] = temp
        return cities

    def iterations(self, thread, solution, best_solution, best_cost):
        local = threading.local()
        local.solution = solution[:]
        local.best_solution = best_solution[:]
        local.best_cost = best_cost

        count_no_improvement = 0
        for iteration in range(self.iteration_max):

            print(f"Thread:{int(thread)+1} - Iteration:{iteration+1} - Best Cost:{local.best_cost} "
                  f"- No Improvement: {count_no_improvement}")
            inital_cost = local.best_cost

            # Perturbations
            local.solution = self.perturbation(local.solution)
            pertubation_cost = self.route_cost(local.solution)

            if pertubation_cost < local.best_cost:
                count_no_improvement = 0
                local.best_cost = pertubation_cost
                local.best_solution = local.solution[:]

            # Local Search
            local.solution = self.two_opt_local_search(local.solution)
            local_search_cost = self.route_cost(local.solution)

            if local_search_cost < local.best_cost:
                count_no_improvement = 0
                local.best_cost = local_search_cost
                local.best_solution = local.solution[:]

            if local.best_cost == inital_cost:
                count_no_improvement = count_no_improvement + 1

            if abs(local.best_cost - local_search_cost) / local.best_cost > 0.01:
                local.solution = local.best_solution[:]

        # Global Critiria
        self.sem.acquire()
        if local.best_cost < self.global_best_cost[-1]:
            self.global_best_solution.append(local.best_solution[:])
            self.global_best_cost.append(local.best_cost)
        self.sem.release()

    def run_parallel_iterations(self, n_threads, initial_solution, initial_best_solution, initial_best_cost):
        threads = []
        for thread in range(n_threads):
            t = threading.Thread(target=self.iterations,
                                 args=(str(thread), initial_solution, initial_best_solution, initial_best_cost))
            t.start()
            threads.append(t)

        for thread in threads:
            thread.join()

    def ils(self):
        start_time = time.time()

        # Initial Solution
        initial_city = random.randint(0, self.n_cities)
        # initial_city = 1

        initial_solution = self.nearest_neighbour(initial_city)

        # Local Search
        initial_solution = self.two_opt_local_search(initial_solution)
        cost = self.route_cost(initial_solution)

        initial_best_solution = initial_solution[:]
        initial_best_cost = cost
        self.global_best_cost.append(initial_best_solution)
        self.global_best_cost.append(cost)

        # Iterations
        self.run_parallel_iterations(self.n_threads, initial_solution, initial_best_solution, initial_best_cost)

        end_time = time.time()
        total_time = end_time - start_time

        print("---------------------------------------------")
        print("Route Cost  : %d" % self.global_best_cost[-1])
        print("Processing Time : %f" % total_time)
        print(f"Route Solution {self.global_best_solution[-1]}")
        f = open(f"{pathlib.Path(__file__).parent.resolve()}/results/"
                 f"results_modified_{self.instance}_{self.n_threads}.xlsx", "a")
        f.write(f"{initial_city},{self.global_best_cost[-1]},{total_time}\n")
        f.close()

    def run(self):
        self.ils()


def main():
    instance = sys.argv[1]
    threads = int(sys.argv[2])
    max_iterations = int(sys.argv[3])
    print("Graph {} - {} Threads  - {} Iterations\n".format(instance, threads, max_iterations))
    ex1 = ParallelExecution(instance, max_iterations, threads)
    ex1.run()


if __name__ == "__main__":
    main()
