import pandas as pd
import numpy as np
import math
from ortools.linear_solver import pywraplp
"""Solve a simple assignment problem."""
from ortools.sat.python import cp_model

from MCJITransformer import MCJITransformer


from pyitlib.discrete_random_variable import information_mutual_conditional, information_mutual_normalised


def main(costs, n_final, type="minimize", one_to_one=False):
    # Data

    num_features = len(costs)
    n = -1 * (n_final - num_features)
    print("Number of features: {}".format(n))

    # Solver
    # Create the mip solver with the SCIP backend.
    # solver = pywraplp.Solver.CreateSolver('SCIP')
    # Model
    solver = cp_model.CpModel()

    # Variables
    # x[i, j] is an array of 0-1 variables, which will be 1
    # if worker i is assigned to task j.
    x = {}
    for i in range(num_features):
        for j in range(num_features):
            # x[i, j] = solver.IntVar(0, 1, '')
            # x[i, j] = solver.BoolVar(f'x[{i},{j}]')
            x[i, j] = solver.NewBoolVar(f'x[{i},{j}]')

    # Declare our intermediate boolean variable.
    rows = {}
    for i in range(num_features):
        rows[i] = solver.NewBoolVar(f'rows[{i}]')
        # Implement rows[i] == (sum(x[i, j...]) >= 1).
        solver.Add(sum([x[i, j] for j in range(num_features)]) >= 1).OnlyEnforceIf(rows[i])
        solver.Add(sum([x[i, j] for j in range(num_features)]) < 1).OnlyEnforceIf(rows[i].Not())

        # Create our half-reified constraints.
        for j in range(num_features):
            if i != j:
                solver.Add(x[j, i] == 0).OnlyEnforceIf(rows[i])


    # solver.SetHint([x[0, j] for j in range(1, n+1)], [1.0 for j in range(1, n+1)])
    for j in range(1, n + 1):
        solver.AddHint(x[0, j], 1)

    # Exclude diagonals.
    # solver.Add(solver.Sum([x[i, i] for i in range(num_features)]) == 0)
    # solver.Add(sum([x[i, i] for i in range(num_features)]) == 0)
    for i in range(num_features):
        solver.Add(x[i, i] == 0)

    # only n assignments.
    # solver.Add(solver.Sum([x[i, j] for i in range(num_features) for j in range(num_features)]) == n)
    solver.Add(sum([x[i, j] for i in range(num_features) for j in range(num_features)]) == n)

    # Each task is assigned to at most 1 worker.
    for j in range(num_features):
        # solver.Add(solver.Sum([x[i, j] for i in range(num_features)]) <= 1)
        solver.Add(sum([x[i, j] for i in range(num_features)]) <= 1)

    # Objective
    objective_terms = []
    for i in range(num_features):
        for j in range(num_features):
            objective_terms.append(costs[i][j] * x[i, j])

    if type == "minimize":
        solver.Minimize(sum(objective_terms))
    else:
        solver.Maximize(sum(objective_terms))

    # Solve
    # status = solver.Solve()
    outs = np.zeros((len(costs), len(costs)))

    # Solve
    solverCpSolver = cp_model.CpSolver()
    status = solverCpSolver.Solve(solver)

    # Print solution.
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print(f'Total cost = {solverCpSolver.ObjectiveValue()}\n')
        for worker in range(num_features):
            for task in range(num_features):
                if solverCpSolver.BooleanValue(x[worker, task]):
                    print(f'Worker {worker} assigned to task {task}.' +
                          f' Cost = {costs[worker][task]}')
                    outs[i, j] = 1
                else:
                    outs[i, j] = 0
    else:
        print('No solution found.')
        print(status)

    # Print solution.
    # if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
    #     print(f'Total cost = {solver.Objective().Value()}\n')
    #     for i in range(num_features):
    #         for j in range(num_features):
    #             # Test if x[i,j] is 1 (with tolerance for floating point arithmetic).
    #             if x[i, j].solution_value() > 0.5:
    #                 print(f'Feature {i} is associeted with feature {j}.' +
    #                       f' Cost: {costs[i][j]}')
    #                 outs[i, j] = 1
    #             else:
    #                 outs[i, j] = 0
    # else:
    #     print('No solution found.')

    return outs


# def read_data(path):
#     df = pd.read_csv(path)
#     array = df.values
#     # target = df['TARGET']
#     X = array[:, 0:8]
#     target = array[:, 8]
#     return df, X, target

def read_data(path):
    df = pd.read_csv(path)
    array = df.values
    X = array[:, 0:-1]
    target = array[:, -1]
    return df, X, target


def calculate_matrix_IMC(matrix, target, precision=100):
    costs = np.zeros((matrix.shape[1], matrix.shape[1]))
    for i, Z in enumerate(matrix.T):
        for j, X in enumerate(matrix.T):
            # I(X;Y|Z)=H(X|Z)−H(X|Y,Z)
            costs[i, j] = math.trunc(information_mutual_conditional(X=X, Y=target, Z=Z) * precision)
    return costs.astype(int)


def calculate_matrix_COV(matrix, precision=100):
    corr = np.cov(matrix)
    costs = np.abs(corr) * precision
    return costs

def generate_synthetic():
    # two features
    X1 = np.random.randn(1000, 1)
    X1 = (X1 - np.mean(X1)) / np.std(X1)
    # digitize examples
    X2 = np.random.randn(1000, 1)
    X2 = (X2 - np.mean(X2)) / np.std(X2)
    Y = 2.3 * X1 - 0.8 * X2 + 0.3 * np.random.rand(1000, 1)
    X = np.concatenate([X1, X2, 15*X1, 18*X2, 17*X1, -30*X1, 0.8*X2], axis=1)
    return X, Y.reshape(-1)

if __name__ == '__main__':
    path = "/home/weslleylc/PycharmProjects/Chagas/data/chagas_processed_wes.csv"
    # path = "/home/weslleylc/PycharmProjects/pythonProject1/data/Diabetes/train.csv"
    df, X, target = read_data(path)

    value = information_mutual_conditional(X=X[:, 2], Y=X[:, 2], Z=target)
    valueA = information_mutual_conditional(X=X[:, 2], Y=target, Z=X[:, 1])
    valueB = information_mutual_conditional(X=X[:, 1], Y=target, Z=X[:, 2])



    mcji = MCJITransformer(2, verbose=True, max_time_in_seconds=10*60)
    mcji2 = MCJITransformer(2, verbose=True, max_time_in_seconds=10*60, func_type="maximize")

    outs = mcji.fit_transform(X, target)
    outs2 = mcji2.fit_transform(X, target)


#
# datasets_list = ['Handwriting', 'Credit approval', 'Gas sensor', 'Libra movement', 'Parkinson',
#             'Breast', 'Sonar', 'Musk', 'Handwriting', 'Colon', 'Leukemia', 'Lymphoma']
#
# def check(name, data_list):
#     for data in data_list:
#         if name.lower() in data.lower():
#             return True
#     return False
#
# datasets['title'].apply(lambda x: check(x, datasets_list))
#
#
#
# import py_uci