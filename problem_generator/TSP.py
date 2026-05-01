import numpy as np
import networkx as nx
from docplex.mp.model import Model
import itertools
import random
try:
    from openqaoa.problems import FromDocplex2IsingModel
except ModuleNotFoundError as e:
    from openqaoa.problems import FromDocplex2IsingModel


def TSP(G: nx.Graph) -> Model:
    """
    Traveling salesman problem (TSP) docplex model from a graph. https://en.wikipedia.org/wiki/Travelling_salesman_problem
    
    Parameters
    ----------
    G : nx.Graph()
        Networx graph of the TSP.

    Returns
    -------
    Model
        Docplex model of the TSP.

    """
    mdl = Model(name="TSP")
    cities = G.number_of_nodes()
    x = {
        (i, j): mdl.binary_var(name=f"x_{i}_{j}")
        for i in range(cities)
        for j in range(cities)
        if i != j
    }
    mdl.minimize(
        mdl.sum(
            G.edges[(i, j, 0)]["weight"] * x[(i, j)]
            for i in range(cities)
            for j in range(cities)
            if i != j and (i, j, 0) in G.edges
        )
    )
    # Only 1 edge goes out from each node
    for i in range(cities):
        mdl.add_constraint(mdl.sum(x[i, j] for j in range(cities) if i != j) == 1)
    # Only 1 edge comes into each node
    for j in range(cities):
        mdl.add_constraint(mdl.sum(x[i, j] for i in range(cities) if i != j) == 1)

    # To eliminate sub-tours
    cities_list = list(range(1, cities))
    possible_subtours = []
    for i in range(2, len(cities_list) + 1):
        for comb in itertools.combinations(cities_list, i):
            possible_subtours.append(list(comb))
    for subtour in possible_subtours:
        mdl.add_constraint(
            mdl.sum(x[(i, j)] for i in subtour for j in subtour if i != j)
            <= len(subtour) - 1
        )
    return mdl



def validate_and_evaluate_tsp_solution(bit_string: str, G: nx.DiGraph, mdl):

    n = G.number_of_nodes()
    # expected_length = G.number_of_edges()*2
    expected_length = G.number_of_nodes()**2 - G.number_of_nodes()
    if len(bit_string) != expected_length:
        print(f"Error: Expected bit string of length {expected_length} but got length {len(bit_string)}")
        return False, None

    # Map the bits to the directed edges in the given order.
    # Order: for i in range(n): for j in range(n) if i != j.
    matrix = np.zeros((n, n))

    edge_order = [(i, j) for i in range(n) for j in range(n) if i != j]
    for idx, (i, j) in enumerate(edge_order):
        matrix[i][j] = int(bit_string[idx])


    # Only 1 edge goes out from each node
    for i in range(n):
        if np.sum([matrix[i, j] for j in range(n) if i != j]) != 1:
            return False, None
    # Only 1 edge comes into each node
    for j in range(n):
        if np.sum([matrix[i, j] for i in range(n) if i != j]) != 1:
            return False, None

    # To eliminate sub-tours
    cities_list = list(range(1, n))
    possible_subtours = []
    for i in range(2, len(cities_list) + 1):
        for comb in itertools.combinations(cities_list, i):
            possible_subtours.append(list(comb))
    for subtour in possible_subtours:
        if np.sum([matrix[(i, j)] for i in subtour for j in subtour if i != j]) > len(subtour) - 1:
            return False, None

    # Evaluate the objective function: sum(weight[i,j] * x[i,j])
    objective_value = np.sum([G.edges[(i, j, 0)]["weight"] * matrix[i][j] for i in range(n) for j in range(n) if i != j and (i, j, 0) in G.edges])

    return True, objective_value


def nodes_from_binary_vars(n):
    # Solve k(k-1)/2 = n --> k^2 - k - 2n = 0
    # k = (1 + sqrt(1 + 8n)) / 2
    k = (1 + (1 + 4 * n)**0.5) / 2
    print(k)
    if k.is_integer() and k >= 3:
        return int(k)
    else:
        raise ValueError("n must be of the form k*(k-1)/2 for integer k >= 3 to form a complete undirected graph.")
    

def TSP_problem_generation(problem_size):
        n = problem_size  # Valid: 6 = 4*3/2, so 4 nodes

        try:
            num_nodes = nodes_from_binary_vars(n)
            print(f"Number of nodes in the TSP graph: {num_nodes}")
        except ValueError as e:
            print(e)
            raise

        G = nx.MultiDiGraph()
        G.add_nodes_from(range(num_nodes))
        # elist = [(0, 1, 1.0), (0, 2, 1.0), (0, 3, 1.0), (1, 2, 1.0), (2, 3, 1.0)]
        print("done1")
        elist = [(i,j,random.uniform(1, 10)) for i in range(num_nodes) for j in range(num_nodes) if i != j]
        # tuple is (i,j,weight) where (i,j) is the edge
        G.add_weighted_edges_from(elist)
        print("done2")
        mdl = TSP(G)
        print("done3")


        lambda0 = 9*num_nodes + 10 #Preoptimized values (see ref.1)
        lambda1 = (9*num_nodes + 10)/10
        lambda2 = (9*num_nodes + 10)/10
        ising_hamiltonian = FromDocplex2IsingModel(mdl, multipliers=lambda0, unbalanced_const=True,
                                                strength_ineq=[lambda1, lambda2]).ising_model
        print("done4")
        non_ising_hamiltonian = FromDocplex2IsingModel(mdl, multipliers=lambda0, unbalanced_const=True,
                                                strength_ineq=[lambda1, lambda2]).qubo_docplex
        print("done5")
        tspnew = FromDocplex2IsingModel(mdl, multipliers=lambda0, unbalanced_const=True,
                                                strength_ineq=[lambda1, lambda2])
        qubo_dict = tspnew.qubo_dict
        return qubo_dict