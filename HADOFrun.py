# import sys
# import builtins
# import os
# sys.stdout = open("stdout.txt", "w", buffering=1)
# def print(text):
#     builtins.print(text)
#     os.fsync(sys.stdout)

# print("This is immediately written to stdout.txt")



problem_generation = "external" #QUBOdict, Knapsack, TSP, load, external
problem_size = 248 # Number of binary variables 
problem_device = "CPU" # CPU or fake_backend or real_backend
num_qubit = 3 # Number of qubits in the circuit
num_models = int(problem_size/num_qubit) # Number of models 
shots1 = 500
shots2 = 5000
steps = 5 #Number of layers of QAOA if using layer, optimisation steps if using full
step_layers = 5 # Number of layers in the circuit for each model when using in full
step_multiplier = 3 # Multiplier for the number of optimser.step() rounds per step in qaoac full/falqon layer
convergence_window = 3
parallel_jobs = 10


hadof_optimiser = "QAOAt-qiskit" # QAOAt, QAOAc(only in partial_opt=full mode now), QA, SA, FALQON, QAOAt-qiskit
selection = "ordered" # random_ordered, ordered, window, random_with_repetition
partial_opt = "manual" # full, layer, manual (set-up gammas and betas in the sequentialHADOF file)
parallel = "sequential" # parallel, sequential
last_step_para = "seq" # para, seq
convergence = "non-convergent" # convergent, non_convergent, acceptance
threading = "False" # True, False
qml_device = "lightning.qubit" # default.qubit, lightning.qubit, lightning.gpu
final_distribution_generation = "sampling" # sampling, postprocessing, sampling with sequential hamiltonian completion
marginal_sample_generation = "circuit" # qubit, circuit
# batching(?)



import numpy as np
import matplotlib.pyplot as plt
import random
import string
from docplex.mp.model import Model
from collections import defaultdict
from sympy import *
import re
import pennylane as qml
from joblib import Parallel, delayed
import networkx as nx
import itertools
import dill as pickle
from copy import copy
import time
from dwave.samplers import SimulatedAnnealingSampler
import multiprocessing as mp
import openjij as oj
if problem_generation == "Knapsack" or problem_generation == "TSP":
    try:
        from openqaoa.problems import FromDocplex2IsingModel
    except ModuleNotFoundError as e:
        from openqaoa.problems import FromDocplex2IsingModel




# if problem_device == "real_backend":
#     your_token="fSx9NV5_U7brUFcHfoulGcNlZfexdHWmrUnwH90gwSud"
#     your_instance="crn:v1:bluemix:public:quantum-computing:us-east:a/d3b8eb50562f47cdb40a5c2a0b1aa923:2ba75e72-fb85-45ae-ac9e-bec4c07810dd::"
# if problem_device == "real_backend":
#     from qiskit_ibm_runtime import QiskitRuntimeService

    # QiskitRuntimeService.save_account(
    #     token=your_token,
    #     instance=your_instance,
    #     overwrite=True)
if problem_device == "fake_backend":
    from qiskit_ibm_runtime.fake_provider import FakeTorino
    backend = FakeTorino()
    session = None
elif problem_device == "real_backend":
    from qiskit_ibm_runtime import QiskitRuntimeService, Session
    service = QiskitRuntimeService()
    # backend = service.least_busy(operational=True, simulator=False, min_num_qubits=num_qubit)
    backend = service.backend("ibm_kingston")
    # session = Session(backend=backed)
    session = None
else: 
    backend = None
    session = None

    


def from_Q_to_Ising(Qubodict, n_qubits):
    """Convert the matrix Q of Eq.3 into Eq.13 elements J and h"""
    # Create default dictionaries to store h and pairwise interactions J
    h = defaultdict(int)
    J = defaultdict(int)

    # Loop over each qubit (variable) in the QUBO matrix
    for i in range(n_qubits):
        # Update the magnetic field for qubit i based on its diagonal element in Q
        h[(i,)] -= Qubodict[(i,)] / 2
        # Loop over other qubits (variables) to calculate pairwise interactions
        for j in range(i + 1, n_qubits):
            # Update the pairwise interaction strength (J) between qubits i and j
            J[(i, j)] += Qubodict[(i, j)] / 4
            # Update the magnetic fields for qubits i and j based on their interactions in Q
            h[(i,)] -= Qubodict[(i, j)] / 4
            h[(j,)] -= Qubodict[(i, j)] / 4
    # Return the magnetic fields, pairwise interactions
    return h, J



def evaluate_penalty(qubo_dict, *args):
    score = 0
    for i in range(problem_size):
        for j in range(i, problem_size):
            if i==j and args[0][i] == 1:
                score+=qubo_dict[(i,)]
            elif j>i and args[0][i] == 1 and args[0][j] == 1:
                score+=qubo_dict[(i, j)]
    return score




def main(Q=None):

    if problem_generation == "load":
        Q = np.load("/Users/namasi/code/HADOFv2/qubo_dict.npy")
        qubo_dict = defaultdict(int)
        for i in range(problem_size):
            for j in range(i, problem_size):
                if i==j:
                    qubo_dict[(i,)] = Q[i][j]
                else:
                    qubo_dict[(i, j)] = Q[i][j]
        print(len(Q))
        # with open("data.pkl", "rb") as f:
        #     qubo_dict = pickle.load(f)

    if problem_generation == "external":
        if Q is not None:
            qubo_dict = defaultdict(int)

            ############################################

            problem_size = len(Q)
            for num in range(0, num_qubit):
                if (problem_size+num)%num_qubit == 0:
                    problem_size = problem_size+num
            num_models = int(problem_size/num_qubit)

            def evaluate_penalty(qubo_dict, *args):
                score = 0
                for i in range(problem_size):
                    for j in range(i, problem_size):
                        if i==j and args[0][i] == 1:
                            score+=qubo_dict[(i,)]
                        elif j>i and args[0][i] == 1 and args[0][j] == 1:
                            score+=qubo_dict[(i, j)]
                return score

            ###########################################

            for i in range(problem_size):
                for j in range(i, problem_size):
                    if i<len(Q) and j<len(Q):
                        if i==j:
                            qubo_dict[(i,)] = Q[i][j]
                        else:
                            # qubo_dict[(i, j)] = 2*Q[i][j]
                            qubo_dict[(i, j)] = Q[i][j]
                    else:
                        if i==j:
                            qubo_dict[(i,)] = 0
                        else:
                            # qubo_dict[(i, j)] = 2*Q[i][j]
                            qubo_dict[(i, j)] = 0
    
        else:
            raise ValueError("For external problem generation, please provide a qubo_dict as input to the main function.")

    if problem_generation == "QUBOdict":
        from problem_generator import QUBOdict
        qubo_dict = QUBOdict.generate_random_qubo_dict(problem_size)
        with open("data.pkl", "wb") as f:
            pickle.dump(qubo_dict, f)

    elif problem_generation == "TSP":
        from problem_generator import TSP
        # Try an example
        qubo_dict = TSP.TSP_problem_generation(problem_size)

    elif problem_generation == "Knapsack":
        from problem_generator import Knapsack
        # Try an example
        qubo_dict = Knapsack.generate_knapsack_problem(problem_size)



    h, J = from_Q_to_Ising(qubo_dict, problem_size)
        
    part_sol = [0.5]*problem_size

    if parallel == "parallel":
        part_sol_iter = part_sol

    if convergence == "convergent":
        part_sol_storage = []

    start_time = time.time()

    final_probs = []
    sampleprobs = []
    finalsamps = []
    for i in range(steps):

        if parallel == "parallel":
            part_sol_iter = part_sol
        
        if convergence == "convergent":
            if len(part_sol_storage) < convergence_window:
                part_sol_storage.append(part_sol)
            else:
                part_sol_storage[i%convergence_window] = part_sol
            part_sol = list(np.mean(part_sol_storage, axis=0))
            
            if parallel == "parallel":
                part_sol_iter = part_sol
            
        if selection == "random_ordered":
            arr = np.arange(problem_size)
            if i != steps-1:  
                np.random.shuffle(arr)

        if parallel == "parallel" and threading == "True" and (i != steps-1 or last_step_para=="para"):
            from HADOF import parallelHADOF
            if selection == "random_ordered":
                # pool = mp.Pool(processes=10)
                # joint = [pool.apply(parallelHADOF.loop, args=(j, arr, part_sol_iter, part_sol, qubo_dict, i, parallel, selection, partial_opt, steps, step_layers, step_multiplier, hadof_optimiser, problem_device, qml_device, shots1, shots2, num_qubit, problem_size, num_models, finalsamps, backend, session)) for j in range(num_models)]
                if i!=steps-1:
                    joint = Parallel(n_jobs=parallel_jobs)(delayed(parallelHADOF.loop)(j, arr, part_sol_iter, part_sol, qubo_dict, i, parallel, selection, partial_opt, steps, step_layers, step_multiplier, hadof_optimiser, problem_device, qml_device, shots1, shots2, num_qubit, problem_size, num_models, finalsamps, backend, session) for j in range(num_models))
                    for klm in range(num_models):
                        for ind in range(num_qubit):
                            part_sol[arr[klm*num_qubit+ind]] = joint[klm][ind]
                else:
                    newjoint = Parallel(n_jobs=parallel_jobs)(delayed(parallelHADOF.loop)(j, arr, part_sol_iter, part_sol, qubo_dict, i, parallel, selection, partial_opt, steps, step_layers, step_multiplier, hadof_optimiser, problem_device, qml_device, shots1, shots2, num_qubit, problem_size, num_models, finalsamps, backend, session) for j in range(num_models))
                    for xx in range(num_models):
                        finalsamps.append(newjoint[xx][1])
                        final_probs.append(newjoint[xx][2])
                        sampleprobs.append(newjoint[xx][3])
            elif selection == "ordered":
                # pool = mp.Pool(processes=10)
                # joint = [pool.apply(parallelHADOF.loop, args=(j, np.arange(problem_size), part_sol_iter, part_sol, qubo_dict, i, parallel, selection, partial_opt, steps, step_layers, step_multiplier, hadof_optimiser, problem_device, qml_device, shots1, shots2, num_qubit, problem_size, num_models, finalsamps, backend, session)) for j in range(num_models)]
                if i!=steps-1:
                    joint = Parallel(n_jobs=parallel_jobs)(delayed(parallelHADOF.loop)(j, np.arange(problem_size), part_sol_iter, part_sol, qubo_dict, i, parallel, selection, partial_opt, steps, step_layers, step_multiplier, hadof_optimiser, problem_device, qml_device, shots1, shots2, num_qubit, problem_size, num_models, finalsamps, backend, session) for j in range(num_models))
                    for klm in range(num_models):
                        part_sol[klm*num_qubit:(klm+1)*num_qubit] = joint[klm]                    
                else:
                    newjoint = Parallel(n_jobs=parallel_jobs)(delayed(parallelHADOF.loop)(j, np.arange(problem_size), part_sol_iter, part_sol, qubo_dict, i, parallel, selection, partial_opt, steps, step_layers, step_multiplier, hadof_optimiser, problem_device, qml_device, shots1, shots2, num_qubit, problem_size, num_models, finalsamps, backend, session) for j in range(num_models))
                    # print(newjoint)
                    for xx in range(num_models):
                        finalsamps.append(newjoint[xx][1])
                        final_probs.append(newjoint[xx][2])
                        sampleprobs.append(newjoint[xx][3])

        else:
            from HADOF import sequentialHADOF
            if selection == "random_ordered":
                finalsamps, final_probs, sampleprobs = sequentialHADOF.seqloop(num_models, num_qubit, qubo_dict, part_sol, selection, arr, partial_opt, steps, step_layers, step_multiplier, hadof_optimiser, problem_device, qml_device, shots1, shots2, problem_size, i, finalsamps, final_probs, sampleprobs, backend, session)
            elif selection == "ordered":
                finalsamps, final_probs, sampleprobs = sequentialHADOF.seqloop(num_models, num_qubit, qubo_dict, part_sol, selection, np.arange(problem_size), partial_opt, steps, step_layers, step_multiplier, hadof_optimiser, problem_device, qml_device, shots1, shots2, problem_size, i, finalsamps, final_probs, sampleprobs, backend, session)




    if session is not None:
        session.close()

    end_time = time.time()
    print(f"Total time: {end_time - start_time}")

    most_probable_solution = []
    for i in range(num_models):
        for k in max(final_probs[i], key=final_probs[i].get):
            most_probable_solution.append(int(k))


    if hadof_optimiser == 'SA':
        average_solutions = []
        for i in range(shots2):
            solv = []
            for j in range(num_models):
                for k in range(num_qubit):
                    solv.append(finalsamps[j][k][i])
            average_solutions.append(solv)
    else:
        average_solutions = []
        for i in range(shots2):
            solv = []
            for j in range(num_models):
                for k in range(num_qubit):
                    solv.append(finalsamps[j][i][k])
            average_solutions.append(solv)





    # avg_sol = [evaluate_penalty(qubo_dict, average_solutions[i]) for i in range(shots2)]
    # average = np.mean(avg_sol)
    # best = np.min(avg_sol)

    print(evaluate_penalty(qubo_dict, most_probable_solution))
    print(np.mean([evaluate_penalty(qubo_dict, average_solutions[i]) for i in range(50)]))


####################################################################################################################################################
    # solver = SimulatedAnnealingSampler()
    # start_time2 = time.time()
    # # sampleset = solver.sample_ising(list(h.values()), J, num_reads = 5000, beta_range = [0,1], beta_schedule_type = 'linear', num_sweeps=50)
    # sampleset = solver.sample_ising(list(h.values()), J, num_reads = 100, anneal_time = 20)
    # # print(sampleset)
    # end_time2 = time.time()
    # sampleset = sampleset.to_pandas_dataframe().sort_values("energy").reset_index(drop=True)
    # sampleset = sampleset.replace(1, 0).replace(-1, 1)
    # samples_unbalanced = defaultdict(int)
    # values = {}
    # for i, row in sampleset.iterrows():
    #     # Postprocessing the information
    #     sample_i = "".join(str(round(row[q])) for q in range(len(h)))
    #     samples_unbalanced[sample_i] += row["num_occurrences"]

    # print(f"Total time: {end_time2 - start_time2}")

    # most_probable_solution2 = []
    # for k in  max(samples_unbalanced, key=samples_unbalanced.get):
    #     most_probable_solution2.append(int(k))
    # # average_solutions2 = []
    # # for i in range(shots2):
    # #     solv = []
    # #     for j in range(problem_size):
    # #         solv.append(sampleset[j][i])
    # #     average_solutions2.append(solv)
    # # avg_sol2 = [evaluate_penalty(qubo_dict, average_solutions2[i]) for i in range(shots2)]
    # # average2 = np.mean(avg_sol2)
    # # best2 = np.min(avg_sol2)

    # print(evaluate_penalty(qubo_dict, most_probable_solution2))

    # new_qubo = {}
    # for key, v in qubo_dict.items():
    #     if len(key) == 1:
    #         i = key[0]
    #         new_qubo[(i, i)] = v
    #     else:
    #         new_qubo[key] = v
    # # print("done")
    # start_time2 = time.time()
    # sampleset = solver.sample_qubo(new_qubo, num_reads=100, anneal_time=20)
    # end_time2 = time.time()
    # sampleset = sampleset.to_pandas_dataframe().sort_values("energy").reset_index(drop=True)
    # # print(sampleset)
    # samples_unbalanced = defaultdict(int)
    # values = {}
    # for i, row in sampleset.iterrows():
    #     # Postprocessing the information
    #     sample_i = "".join(str(round(row[q])) for q in range(len(h)))
    #     samples_unbalanced[sample_i] += row["num_occurrences"]
    # most_probable_solution2 = []
    # for k in  max(samples_unbalanced, key=samples_unbalanced.get):
    #     most_probable_solution2.append(int(k))
    # print(f"Total time: {end_time2 - start_time2}")
    # print(evaluate_penalty(qubo_dict, most_probable_solution2))



    # new_h = {k[0]: v for k, v in h.items()}
    # solver = oj.SASampler()
    # start_time2 = time.time()
    # sampleset = solver.sample_ising(new_h, J, num_reads=5000)
    # end_time2 = time.time()
    # sampleset = sampleset.to_pandas_dataframe().sort_values("energy").reset_index(drop=True)
    # sampleset = sampleset.replace(1, 0).replace(-1, 1)
    # samples_unbalanced = defaultdict(int)
    # values = {}
    # for i, row in sampleset.iterrows():
    #     # Postprocessing the information
    #     sample_i = "".join(str(round(row[q])) for q in range(len(h)))
    #     samples_unbalanced[sample_i] += row["num_occurrences"]

    # print(f"Total time: {end_time2 - start_time2}")

    # most_probable_solution2 = []
    # for k in  max(samples_unbalanced, key=samples_unbalanced.get):
    #     most_probable_solution2.append(int(k))
    # # average_solutions2 = []
    # # for i in range(shots2):
    # #     solv = []
    # #     for j in range(problem_size):
    # #         solv.append(sampleset[j][i])
    # #     average_solutions2.append(solv)
    # # avg_sol2 = [evaluate_penalty(qubo_dict, average_solutions2[i]) for i in range(shots2)]
    # # average2 = np.mean(avg_sol2)
    # # best2 = np.min(avg_sol2)

    # print(evaluate_penalty(qubo_dict, most_probable_solution2))




    # if problem_size <= 80:
    #     mdl = Model(name="qubo")
    #     x = mdl.binary_var_list(problem_size, name='x')
    #     objective = mdl.sum([qubo_dict[(i,)] * x[i] * x[j] if i==j else qubo_dict[(i, j)] * x[i] * x[j] for j in range(problem_size) for i in range(problem_size)])
    #     mdl.minimize(objective)
    #     start_time3 = time.time()
    #     docplex_sol = mdl.solve()  
    #     end_time3 = time.time() 
    #     solution = []
    #     for i in range(problem_size): 
    #         solution.append(int(docplex_sol.get_value(f"x_{i}")))
    #     print("Docplex solution penalty:", evaluate_penalty(qubo_dict, solution))
    #     print(f"Total time: {end_time3 - start_time3}")

####################################################################################################################################################

    # if problem_size<=80:
    #     print(best, evaluate_penalty(qubo_dict, most_probable_solution), average, end_time - start_time, best2, evaluate_penalty(qubo_dict, most_probable_solution2), average2, end_time2 - start_time2, evaluate_penalty(qubo_dict, solution), end_time3 - start_time3)
    # else:
    #     print(best, evaluate_penalty(qubo_dict, most_probable_solution), average, end_time - start_time, best2, evaluate_penalty(qubo_dict, most_probable_solution2), average2, end_time2 - start_time2)
    
    
    
    # with open("results-newnoisy.pkl", "wb") as f:
    #     pickle.dump([most_probable_solution, qubo_dict, average_solutions], f)


    return most_probable_solution, qubo_dict, average_solutions

if __name__ == '__main__':
    # mp.freeze_support()  # optional unless you're freezing the script
    main()