# 🧠 Intelligent TSP Solver

## Hybrid AI Framework for the Traveling Salesman Problem

```{=html}
<p align="center">
```
![AI](https://img.shields.io/badge/AI-Artificial%20Intelligence-blue)
![Python](https://img.shields.io/badge/Python-3.x-yellow)
![Algorithms](https://img.shields.io/badge/Solvers-ACO%20%7C%20GA%20%7C%20Held--Karp-green)
![ML](https://img.shields.io/badge/Machine%20Learning-Decision%20Tree-orange)
![GUI](https://img.shields.io/badge/Interface-PyQt6-purple)
![Status](https://img.shields.io/badge/Project-Academic-lightgrey)

```{=html}
</p>
```

------------------------------------------------------------------------

## 📌 Overview

This project is an intelligent framework for solving the **Traveling
Salesman Problem (TSP)** using a combination of classical Artificial
Intelligence, optimization algorithms, and Machine Learning.

The system integrates:

-   Exact optimization
-   Metaheuristic search
-   Intelligent algorithm selection

to analyze different solving strategies and choose appropriate
approaches based on problem characteristics.

The goal is not only to find a solution, but also to study the behavior,
efficiency, and scalability of different AI methods.

------------------------------------------------------------------------

# ✨ Features

-   ✅ Multiple TSP solving strategies
-   ✅ Exact and approximate optimization comparison
-   ✅ Graph-based geographic modeling
-   ✅ Interactive PyQt6 interface
-   ✅ Real-time visualization
-   ✅ Machine learning advisor
-   ✅ Modular software architecture
-   ✅ Performance benchmarking

------------------------------------------------------------------------

# 🏗️ Architecture

                        User Interface
                             |
                           PyQt6
                             |
                             v

                  +---------------------+
                  |  Data Processing    |
                  | Haversine Distance  |
                  | Distance Matrix     |
                  +----------+----------+

                             |
            +----------------+----------------+
            |                |                |
            v                v                v

          ACO Solver       GA Solver     Prolog Solver
       (Swarm AI)      (Evolutionary)    (Exact DP)

            \                |                /
             \               |               /

                  Decision Tree Advisor
                  Intelligent Selection

------------------------------------------------------------------------

# 🧩 Implemented Algorithms

## 🐜 Ant Colony Optimization (ACO)

Inspired by natural ant colonies.

Main mechanisms:

-   Pheromone matrix
-   Probabilistic exploration
-   Evaporation
-   Reinforcement of good routes

ACO provides effective solutions for large search spaces.

------------------------------------------------------------------------

## 🧬 Genetic Algorithm (GA + ERX)

An evolutionary approach designed for permutation problems.

Includes:

-   Population generation
-   Selection
-   Mutation
-   Edge Recombination Crossover (ERX)

ERX preserves useful connections between cities and improves route
quality.

------------------------------------------------------------------------

## 🔮 Exact Prolog Solver (Held-Karp)

A logic programming based exact solver.

Characteristics:

-   Dynamic programming approach
-   Optimal solution guarantee
-   Memoization optimization

Used as a baseline for comparing heuristic algorithms.

------------------------------------------------------------------------

## 🌳 AI Algorithm Advisor

A Decision Tree model predicts which solver is more appropriate.

The model considers:

-   Number of cities
-   Graph characteristics
-   Distance distribution
-   Problem complexity

and recommends a suitable optimization method.

------------------------------------------------------------------------

# 🌍 Data Layer

Cities are represented as nodes of a weighted graph.

The system:

1.  Receives geographic coordinates
2.  Calculates distances using Haversine formula
3.  Creates an N×N distance matrix
4.  Sends the matrix to solvers

------------------------------------------------------------------------

# 📂 Repository Structure

    TSP-AI-Solver/

    ├── main.py
    │   └── Application entry point

    ├── gui.py
    │   └── PyQt6 graphical interface

    ├── core.py
    │   └── Data processing and distance calculation

    ├── solvers.py
    │   └── ACO and Genetic Algorithm

    ├── tsp_solver.pl
    │   └── Held-Karp Prolog solver

    ├── prolog_bridge.py
    │   └── Python-Prolog interface

    ├── train_model.py
    │   └── Decision Tree training

    └── models/
        └── Trained ML model files

------------------------------------------------------------------------

# ⚙️ Installation

Clone the repository:

``` bash
git clone <repository-url>

cd TSP-AI-Solver
```

Install dependencies:

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

# ▶️ Running

Start the application:

``` bash
python main.py
```

Train the advisor model:

``` bash
python train_model.py
```

------------------------------------------------------------------------

# 📊 Algorithm Comparison

  Method              Category        Optimal   Suitable Size
  ------------------- --------------- --------- ---------------
  Held-Karp           Exact           Yes       Small
  ACO                 Metaheuristic   No        Medium/Large
  Genetic Algorithm   Evolutionary    No        Medium/Large
  Decision Tree       ML Advisor      N/A       Selection

------------------------------------------------------------------------

# 🧪 Benchmarking

The framework can compare:

-   Execution time
-   Route quality
-   Convergence speed
-   Scalability

As the number of cities increases:

-   Exact approaches become computationally expensive
-   Metaheuristics become more practical

------------------------------------------------------------------------

# 🖥️ Demo

Add your screenshots here:

    docs/
     ├── interface.png
     ├── benchmark.png
     └── architecture.png

Example:

![Application Screenshot](docs/interface.png)

------------------------------------------------------------------------

# 🔬 Academic Background

This project demonstrates the integration of:

-   Search algorithms
-   Optimization techniques
-   Logic programming
-   Evolutionary computation
-   Machine learning

for solving a classical NP-hard optimization problem.

------------------------------------------------------------------------

# 👨‍💻 Author

**Moein Hassanpour**

Artificial Intelligence Course Project

------------------------------------------------------------------------

⭐ If you find this project useful, consider giving it a star.
