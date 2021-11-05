import numpy as np
import os
import sys

problem = "dropwave"

if problem == "dropwave":
    budget = 40.0
    n_trials = 100
    algos = ["EI-PUC-CC", "B-MS-EI_111_1"]

#
script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
problem_results_dir = script_dir + "/results/" + problem + "/"

print(problem)
for a, algo in enumerate(algos):
    print(algo)
    if algo == "EI-PUC-CC" or "B-MS-EI" in algo:
        algo_id = algo + "_" + str(int(budget))
    else:
        algo_id = algo

    algo_results_dir = problem_results_dir + algo_id + "/"
    for trial in range(1, n_trials + 1):
        #print(trial)
        try:
            X = np.loadtxt(algo_results_dir + "X/X_" + str(trial) + ".txt")
            if X.ndim == 1:
                X = np.expand_dims(X, axis=-1)
            input_dim = X.shape[1]
            n_init_evals = 2 * (input_dim + 1)
            cost_X = np.loadtxt(algo_results_dir + "cost_X/cost_X_" + str(trial) + ".txt")
            init_cost = sum(cost_X[:n_init_evals])
            total_cost = sum(cost_X)
            if total_cost - init_cost < budget:
                print("Trial {} is not complete yet.".format(trial))
                print("Current cumulative cost is: {}".format(total_cost - init_cost))
        except:
            print("Trial {} was not found.".format(trial))