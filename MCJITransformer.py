import numpy as np
import math
from os.path import exists
import pickle
from ortools.sat.python import cp_model
from pyitlib.discrete_random_variable import information_mutual_conditional
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.base import BaseEstimator, TransformerMixin

from knncmi import cmi


class MCJITransformer(BaseEstimator, TransformerMixin):
    # the constructor
    ''''''

    def __init__(self, k=2, precision=1000000000000, func_type="minimize", verbose=False, max_time_in_seconds=-1,
                 path_matrix="./costs.data"):
        self.k = k
        self.precision = precision
        self.func_type = func_type
        self.verbose = verbose
        self.max_time_in_seconds = max_time_in_seconds
        self.path_matrix = path_matrix
        self.costs = None
        self.outs = None
        self.cols = None
        self.init_solution = None

    # estimator method
    def fit(self, X, y):

        if self.k >= X.shape[1] or self.k < 1:
            self.cols = list(range(X.shape[1]))
            return self

        self.costs = self.calculate_matrix_IMC(matrix=X, target=y, precision=self.precision)

        self.init_solution = None
        self.outs = self.calculate_features(self.costs,
                                            n_final=self.k,
                                            type=self.func_type)
        self.cols = np.argwhere(np.sum(self.outs, axis=0) == 0).reshape(-1)
        return self

    # transformation
    def transform(self, X, y=None):
        return X[:, self.cols]

    def calculate_matrix_IMC(self, matrix, target, precision=100):
        size = matrix.shape[1]
        costs = np.zeros((size, size))
        matrix = matrix.T
        for i, X in enumerate(matrix):
            # I(X;Y|Z)=H(X|Z)−H(X|Y,Z)
            if self.func_type == "minimize":
                for j, Y in enumerate(matrix):
                    if i == j:
                        costs[i, j] = 0
                    else:
                        costs[i, j] = math.trunc(information_mutual_conditional(X=Y, Y=target, Z=X) * precision)
            else:
                for j, Y in enumerate(matrix[i:, :]):

                    if i == j:
                        costs[i, j] = precision
                    else:
                        costs[i, j] = math.trunc(information_mutual_conditional(X=X, Y=Y, Z=target) * precision)
                        costs[j, i] = costs[i, j]

        return costs.astype(int)

    def calculate_features(self, costs, n_final, type="minimize"):
        # Data

        num_features = len(costs)
        n = num_features - n_final

        # Model
        model = cp_model.CpModel()

        # Variables
        # x[i, j] is an array of 0-1 variables, which will be 1
        # if worker i is assigned to task j.
        x = {}
        for i in range(num_features):
            for j in range(num_features):
                x[i, j] = model.NewBoolVar(f'x[{i},{j}]')

        # Declare our intermediate boolean variable.
        rows = {}
        for i in range(num_features):
            rows[i] = model.NewBoolVar(f'rows[{i}]')
            # Implement rows[i] == (sum(x[i, j...]) >= 1).
            model.Add(sum([x[i, j] for j in range(num_features)]) >= 1).OnlyEnforceIf(rows[i])
            model.Add(sum([x[i, j] for j in range(num_features)]) < 1).OnlyEnforceIf(rows[i].Not())

            # Create our half-reified constraints.
            for j in range(num_features):
                if i != j:
                    model.Add(x[j, i] == 0).OnlyEnforceIf(rows[i])

        # Constraints

        # Exclude diagonals.
        for i in range(num_features):
            model.Add(x[i, i] == 0)

        # only n assignments.
        model.Add(sum([x[i, j] for i in range(num_features) for j in range(num_features)]) == n)

        # Each task is assigned to at most 1 worker.
        for j in range(num_features):
            model.Add(sum([x[i, j] for i in range(num_features)]) <= 1)

        # Objective
        objective_terms = []
        for i in range(num_features):
            for j in range(num_features):
                objective_terms.append(costs[i][j] * x[i, j])

        if type == "minimize":
            model.Minimize(sum(objective_terms))
        else:
            model.Maximize(sum(objective_terms))

        # Solve
        outs = np.zeros((len(costs), len(costs)))
        # set initial feasible solution
        for j in range(1, n + 1):
            model.AddHint(x[0, j], 1)

        # Solve
        solver = cp_model.CpSolver()
        if self.max_time_in_seconds != -1:
            solver.parameters.max_time_in_seconds = self.max_time_in_seconds

        status = solver.Solve(model)

        # Print solution.
        if self.verbose:
            if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
                print(f'Total cost = {solver.ObjectiveValue()}\n')
                for i in range(num_features):
                    for j in range(num_features):
                        if solver.BooleanValue(x[i, j]):
                            print(f'Worker {i} assigned to task {j}.' +
                                  f' Cost = {costs[i][j]}')
                            outs[i, j] = 1
                        else:
                            outs[i, j] = 0
            else:
                print('No solution found.')
        else:
            for i in range(num_features):
                for j in range(num_features):
                    if solver.BooleanValue(x[i, j]):
                        outs[i, j] = 1
                    else:
                        outs[i, j] = 0
        return outs

