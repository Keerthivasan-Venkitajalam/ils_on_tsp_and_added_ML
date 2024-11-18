## Iterated Local Search for Traveling Salesman Problem (TSP) with Parallel Execution

This repository contains a Python implementation of the **Iterated Local Search (ILS)** algorithm to solve the **Traveling Salesman Problem (TSP)**. The project aims to optimize the TSP route by combining local search techniques with perturbation strategies, while leveraging parallel execution to significantly improve the efficiency of the solution process. This implementation is particularly useful for solving large TSP instances within a reasonable amount of time.

### Problem Overview: 
The **Traveling Salesman Problem (TSP)** is a well-known combinatorial optimization problem where the objective is to find the shortest possible route that visits each city exactly once and returns to the starting point. It is NP-hard, making it computationally expensive to solve for large numbers of cities. Iterated Local Search (ILS) is a metaheuristic approach that improves the solution by iteratively perturbing and refining an initial solution, making it a suitable choice for TSP.

### Key Features:

#### 1. **Iterated Local Search (ILS) Algorithm**:
   - **Perturbation**: The solution is modified randomly by swapping two cities. This disrupts the current solution, enabling the search to explore new regions of the solution space.
   - **Local Search (2-opt)**: After perturbation, a local search is applied using the 2-opt algorithm. This involves iteratively swapping pairs of edges to reduce the overall route cost.
   - **Global Improvement**: The best solution found during each iteration is tracked and stored. The process continues for a pre-determined number of iterations to avoid local minima.

#### 2. **Parallel Execution**:
   - **Threading**: The ILS algorithm is parallelized using Python’s `threading` library. Multiple threads run concurrently to explore different parts of the solution space in parallel, improving overall runtime.
   - **Semaphore Synchronization**: A semaphore is used to control access to the shared global best solution across threads, ensuring thread safety while updating the global best solution.

#### 3. **Customizable TSP Instances**:
   - The project can handle TSP problem instances in TSPLIB format, which is a common standard for representing TSP problems. The algorithm supports **EUC_2D** edge weight type, which assumes that cities are represented in 2D Euclidean space.
   - Users can load their own TSP instances by specifying the filename (without the extension) when running the script.

#### 4. **Results Storage**:
   - After execution, the best solution and the corresponding route cost are saved into an **Excel file**. This allows users to easily track and compare the performance of the algorithm for different problem instances or parameter configurations.
   - The results file is named `results_parallel_{instance}.xlsx`, where `{instance}` is the name of the TSP problem instance being solved.

### How It Works:

1. **Problem Initialization**:
   - The TSP problem instance is loaded using the `tsplib95` library, which provides functionality for working with TSPLIB files.
   - The coordinates of the cities are extracted from the problem data. The cities are then indexed, and their coordinates are stored in two separate lists (`coord_x` and `coord_y`).

2. **Initial Solution**:
   - An initial solution is generated using the **nearest neighbor heuristic**, which starts from a random city and at each step chooses the closest unvisited city until all cities have been visited.
   - This initial solution is then improved by applying a **2-opt local search** to remove potential "crossings" in the route, thereby shortening the overall travel distance.

3. **Iterative Improvement**:
   - For each iteration, a perturbation step is applied to the current solution by swapping two cities. After perturbation, the 2-opt local search is applied again to refine the solution.
   - This process is repeated for a specified number of iterations (`iteration_max`), allowing the algorithm to explore and improve the solution progressively.

4. **Parallelization**:
   - The solution process is parallelized using multiple threads. Each thread runs a separate instance of the ILS algorithm, working independently to find the best solution.
   - The best solution from each thread is tracked and updated in a global list, ensuring that the solution with the lowest cost is preserved.

5. **Results Output**:
   - Once the parallel execution is complete, the best solution is printed, including the total route cost and the processing time.
   - The results, including the initial city, best cost, and total processing time, are saved to an Excel file (`results_parallel_{instance}.xlsx`), which can be found in the results directory.

### Requirements:
To run this project, you need the following Python packages:
- **`tsplib95`**: A library for parsing and working with TSPLIB files (used to load TSP problem instances).
- **`openpyxl`**: A library to read and write Excel files, used to save the results.
- **`pathlib`**: A module to handle filesystem paths (used for file management).
- **`threading`**: A module to run the algorithm in parallel using threads.

Install the dependencies using `pip`:

```bash
pip install tsplib95 openpyxl
```

### Usage:
1. Clone the repository.
2. Install dependencies using `pip install tsplib95 openpyxl`.
3. Run the relevant code segment in the .ipynb notebook file.
4. Enter the number of iterations and threads when prompted.

### Example:
```bash
Enter the instance filename (without extension, e.g., 'berlin52'): berlin52
Enter the maximum number of iterations: 1000
Enter the number of threads for parallel execution: 4
```


### Results File (Excel):

The results file will contain the following columns:
- **Initial City**: The city from which the algorithm started.
- **Best Cost**: The best route cost found by the algorithm.
- **Total Time**: The total processing time for the algorithm.

