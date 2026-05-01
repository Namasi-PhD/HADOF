try:
    from openqaoa.problems import FromDocplex2IsingModel
except ModuleNotFoundError as e:
    from openqaoa.problems import FromDocplex2IsingModel

import random
from docplex.mp.model import Model


# Docplex model, we need to convert our problem in this format to use the unbalanced penalization approach
def generate_knapsack_problem(problem_size):
    item_labels = [str(i+1) for i in range(problem_size)]
    items_values = {item: random.randint(10, 50) for item in item_labels}
    items_weight = {item: random.randint(10, 50) for item in item_labels}
    values_list = list(items_values.values())
    weights_list = list(items_weight.values())

    print("Items values:", items_values)
    print("Items weights:", items_weight)

    total_weight = sum(weights_list)
    maximum_weight = total_weight // 3
    print(maximum_weight)
    n_items = len(values_list)
    mdl = Model()
    x = mdl.binary_var_list(range(n_items), name="x")
    cost = -mdl.sum(x[i] * values_list[i] for i in range(n_items))
    mdl.minimize(cost)
    mdl.add_constraint(mdl.sum(x[i] * weights_list[i] for i in range(n_items)) <= maximum_weight)

    lambda_1, lambda_2 = (
        1,
        0.03,
    )  # Parameters of the unbalanced penalization function (They are in the main paper)
    ising_hamiltonian = FromDocplex2IsingModel(
        mdl,
        unbalanced_const=True,
        strength_ineq=[lambda_1, lambda_2],  # https://arxiv.org/abs/2211.13914
    ).ising_model

    non_ising_hamiltonian = FromDocplex2IsingModel(
        mdl,
        unbalanced_const=True,
        strength_ineq=[lambda_1, lambda_2],  # https://arxiv.org/abs/2211.13914
    ).qubo_docplex

    tspnew = FromDocplex2IsingModel(mdl, unbalanced_const=True,
                                                strength_ineq=[lambda_1, lambda_2])
    qubo_dict = tspnew.qubo_dict

    return qubo_dict
