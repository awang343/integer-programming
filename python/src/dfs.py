from dataclasses import dataclass
import numpy as np
from docplex.mp.model import Model
import math


# {{{ Class Definition
@dataclass(frozen=True)
class IPConfig:
    numTests: int  # number of tests
    numDiseases: int  # number of diseases
    costOfTest: np.ndarray  # [numTests] the cost of each test
    A: (
        np.ndarray
    )  # [numTests][numDiseases] 0/1 matrix if test is positive for disease}}}


# {{{ Data parser
#  * File Format
#  * #Tests (i.e., n)
#  * #Diseases (i.e., m)
#  * Cost_1 Cost_2 . . . Cost_n
#  * A(1,1) A(1,2) . . . A(1, m)
#  * A(2,1) A(2,2) . . . A(2, m)
#  * . . . . . . . . . . . . . .
#  * A(n,1) A(n,2) . . . A(n, m)


def data_parse(filename: str):
    try:
        with open(filename, "r") as fl:
            numTests = int(fl.readline().strip())  # n
            numDiseases = int(fl.readline().strip())  # m

            costOfTest = np.array([float(i) for i in fl.readline().strip().split()])

            A = np.zeros((numTests, numDiseases))
            for i in range(0, numTests):
                A[i, :] = np.array([int(i) for i in fl.readline().strip().split()])
            return numTests, numDiseases, costOfTest, A
    except Exception as e:
        print(f"Error reading instance file. File format may be incorrect.{e}")
        exit(1)


# }}}


class dfsInstance:
    def __init__(self, filename: str) -> None:
        # print("USING DFS")
        numT, numD, cst, A = data_parse(filename)
        self.numTests = numT
        self.numDiseases = numD
        self.A = A
        self.costOfTest = cst

        # Make sure the initial cost is strictly worse than any solution
        self.incumbent_cost = sum(self.costOfTest) + 1
        self.unassigned = set(l for l in range(self.numTests))

        self.model = Model()  # CPLEX solver
        self.init_model()

    def toString(self):
        out = ""
        out = f"Number of test: {self.numTests}\n"
        out += f"Number of diseases: {self.numDiseases}\n"
        cst_str = " ".join([str(i) for i in self.costOfTest])
        out += f"Cost of tests: {cst_str}\n"
        A_str = "\n".join(
            [" ".join([str(j) for j in self.A[i]]) for i in range(0, self.A.shape[0])]
        )
        out += f"A:\n{A_str}"
        return out

    def init_model(self):
        # Compute 3d matrix
        int_A = self.A.astype(int)
        xor_matrix = np.bitwise_xor(int_A[:, :, np.newaxis], int_A[:, np.newaxis, :])
        xor_A = np.transpose(xor_matrix, (1, 2, 0))

        self.tests = self.model.continuous_var_list(self.numTests, 0, 1)

        # Constraints
        for d1 in range(self.numDiseases):
            for d2 in range(d1 + 1, self.numDiseases):
                self.model.add_constraint(
                    self.model.scal_prod(terms=self.tests, coefs=xor_A[d1][d2]) >= 1
                )

        # Objective function
        self.model.minimize(
            self.model.scal_prod(terms=self.tests, coefs=self.costOfTest)
        )

    def brancher(self, best_idx):
        to_append = []

        self.model.add_constraint(self.tests[best_idx] == 0, ctname="explorer")
        sol0 = self.model.solve()
        if sol0 is not None:
            is_int = all(
                [x.solution_value == 0 or x.solution_value == 1 for x in self.tests]
            )
            obj0 = sol0.objective_value
            if obj0 < self.incumbent_cost:
                if is_int:
                    self.incumbent_cost = obj0
                else:
                    next_dist = 0.5
                    next_idx = None
                    for idx in self.unassigned:
                        dist = abs(self.tests[idx].solution_value - 0.5)
                        if dist < next_dist:
                            next_idx = idx
                            next_dist = dist

                    to_append += [
                        (best_idx, -1),
                        (best_idx, 0, next_idx),
                    ]
        self.model.remove_constraint("explorer")

        self.model.add_constraint(self.tests[best_idx] == 1, ctname="explorer")
        sol1 = self.model.solve()
        if sol1 is not None:
            is_int = all(
                [x.solution_value == 0 or x.solution_value == 1 for x in self.tests]
            )
            obj1 = sol1.objective_value
            if obj1 < self.incumbent_cost:
                if is_int:
                    self.incumbent_cost = obj1
                else:
                    next_dist = 0.5
                    next_idx = None
                    for idx in self.unassigned:
                        dist = abs(self.tests[idx].solution_value - 0.5)
                        if dist < next_dist:
                            next_idx = idx
                            next_dist = dist

                    if obj1 < obj0:
                        to_append = to_append + [
                            (best_idx, -1),
                            (best_idx, 1, next_idx),
                        ]
                    else:
                        to_append = [
                            (best_idx, -1),
                            (best_idx, 1, next_idx),
                        ] + to_append
        self.model.remove_constraint("explorer")

        return to_append

    def solve(self):
        # Start by setting rows that are all the same to 0
        sol = self.model.solve()
        best_dist = 0.5
        best_idx = None
        for idx in self.unassigned:
            dist = abs(self.tests[idx].solution_value - 0.5)
            if dist < best_dist:
                best_idx = idx
                best_dist = dist

        # Contains tuples of assignments
        # Assignment to -1 is an instruction to reset
        dfs_stack = self.brancher(best_idx)

        # Main DFS loop
        while dfs_stack:
            # Solve the current LP
            action = dfs_stack.pop(-1)

            # If current action is to remove, we can just remove and continue
            if action[1] == -1:
                self.model.remove_constraint(str(action[0]))
                self.unassigned.add(action[0])
                continue
            else:
                self.model.add_constraint(
                    self.tests[action[0]] == action[1], ctname=str(action[0])
                )
                self.unassigned.remove(action[0])

            best_idx = action[2]
            to_append = self.brancher(best_idx)
            dfs_stack.extend(to_append)

        return int(self.incumbent_cost)
