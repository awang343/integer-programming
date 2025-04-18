from dataclasses import dataclass
import numpy as np
from docplex.mp.model import Model
import math
import heapq as hq


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


class bfsfarInstance:
    def __init__(self, filename: str) -> None:
        # print("Using BFS Far")
        numT, numD, cst, A = data_parse(filename)
        self.numTests = numT
        self.numDiseases = numD
        self.A = A
        self.costOfTest = cst

        # Make sure the initial cost is strictly worse than any solution
        self.incumbent_cost = sum(self.costOfTest) + 1
        self.heap = []
        hq.heapify(self.heap)

        self.model = Model()  # CPLEX solver
        self.presolve()

    def presolve(self):
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

        sol = self.model.solve()
        hq.heappush(
            self.heap, (sol.objective_value, [i.solution_value for i in self.tests], {})
        )

    def analyze_node(self, popped_node):
        # print(popped_node)
        obj_val, solution, assignments = popped_node

        # 1. Check if popped node can be pruned
        if obj_val >= self.incumbent_cost:
            return

        # 2. If not, find the variable closest to 0.5
        best_dist = 2
        best_idx = None
        for idx, val in enumerate(solution):
            dist = min(abs(val), abs(val-1))
            if dist <= best_dist and idx not in assignments:
                best_idx = idx
                best_dist = dist
        # 3. Check if the two nodes added are integer solutions and update incumbent
        self.model.add_constraints(
            [self.tests[var] == val for var, val in assignments.items()],
            names=[str(var) for var in assignments],
        )

        for val in [0, 1]:
            self.model.add_constraint(self.tests[best_idx] == val, ctname="branch_var")
            sol = self.model.solve()
            self.model.remove_constraint("branch_var")

            if sol is None:
                continue

            if all([v.solution_value in [0, 1] for v in self.tests]):
                self.incumbent_cost = min(sol.objective_value, self.incumbent_cost)
                continue

            new = assignments.copy()
            new[best_idx] = val
            hq.heappush(
                self.heap,
                (
                    sol.objective_value,
                    [x.solution_value for x in self.tests],
                    new,
                ),
            )

        self.model.remove_constraints([str(var) for var in assignments])

    def solve(self):
        while self.heap:
            # 1. While heap is not empty, pop nodes
            popped = hq.heappop(self.heap)
            self.analyze_node(popped)

        return int(self.incumbent_cost)

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
