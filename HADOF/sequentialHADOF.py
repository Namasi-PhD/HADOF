import numpy as np
from collections import defaultdict
import openjij as oj
import pennylane as qml
from dwave.samplers import SimulatedAnnealingSampler

def newmodel(qubodict, part_solution):
    vars = []
    rename = defaultdict(int)
    count = 0
    for i in range(len(part_solution)):
        if part_solution[i] == 'a':
            vars.append(i)
            rename[i] = count
            count += 1  
    newdict = defaultdict(int)

    for i in range(len(vars)):
        for j in range(i, len(vars)):
            if i == j:
                newdict[(i,)] += qubodict[(vars[i],)]
                for k in range(len(part_solution)):
                    if part_solution[k] != 'a':
                        newdict[(i,)] += qubodict[(vars[i], k)]*part_solution[k] 
                        newdict[(i,)] += qubodict[(k, vars[i])]*part_solution[k]
            else:
                newdict[(i, j)] += qubodict[(vars[i], vars[j])]
    return newdict



def samples_dict(samples, n_items):
    """Just sorting the outputs in a dictionary"""
    results = defaultdict(int)
    for sample in samples:
        results["".join(str(i) for i in sample)[:n_items]] += 1
    return results

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



def seqloop(num_models, num_qubit, qubo_dict, part_sol, selection, arr, partial_opt, steps, step_layers, step_multiplier, hadof_optimiser, problem_device, qml_device, shots1, shots2, problem_size, i, finalsamps, final_probs, sampleprobs, backend, session):    
    for j in range(num_models):
        # Docplex model, we need to convert our problem in this format to use the unbalanced penalization approach

        local_sol = part_sol.copy()
            
        if selection == "ordered":
            local_sol[((j)%num_models)*num_qubit:(((j)%num_models)+1)*num_qubit] = ['a']*num_qubit
        elif selection == "random_ordered":
            for ind in range(num_qubit):
                local_sol[arr[((j)%num_models)*num_qubit+ind]] = 'a'


        qubodict = newmodel(qubo_dict, local_sol)
        h_new, J_new = from_Q_to_Ising(qubodict, num_qubit)  # Convert the QUBO to Ising model


        if hadof_optimiser == "QAOAt":
            from problem_solver import QAOAt

            if partial_opt == "layer":
                betas = np.linspace(0, (i+1)/steps, i+1)[::-1]  # Parameters for the mixer Hamiltonian
                gammas = np.linspace(0, (i+1)/steps, i+1)
            elif partial_opt == "full":
                betas = np.linspace(0, 1, step_layers+1)[::-1]  # Parameters for the mixer Hamiltonian
                gammas = np.linspace(0, 1, step_layers+1)
            elif partial_opt == "manual":
                betas = np.linspace(0, (i+1)/steps, step_layers+1)[::-1]  # Parameters for the mixer Hamiltonian
                gammas = np.linspace(0, (i+1)/steps, step_layers+1)                

            qaoa_circuit1, qaoa_circuit2 = QAOAt.create_qaoa_circuit(problem_device, qml_device, shots1, shots2, num_qubit, backend)
            res = qaoa_circuit1(gammas, betas, h_new, J_new, num_qubits=num_qubit)
        elif hadof_optimiser == "SA":
            solver = SimulatedAnnealingSampler()
            # sampleset = solver.sample_ising(list(h_new.values()), J_new, num_reads = shots1, beta_range = [0,1], beta_schedule_type = 'linear', num_sweeps=50)
            sampleset = solver.sample_ising(list(h_new.values()), J_new, num_reads = shots1, anneal_time = 20)
            # h_new2 = {k[0]: v for k, v in h_new.items()}
            # solver = oj.SASampler()
            # sampleset = solver.sample_ising(h_new2, J_new, num_reads=shots1)
            sampleset = sampleset.to_pandas_dataframe().sort_values("energy").reset_index(drop=True)
            res = sampleset.replace(1, 0).replace(-1, 1)
        elif hadof_optimiser == "QAOAc":
            from pennylane import numpy as pnp
            from problem_solver import QAOAc
            qaoa_circuit1, qaoa_circuit2, cost_function = QAOAc.create_qaoac_circuit(problem_device, qml_device, shots1, shots2, num_qubit, step_layers, h_new, J_new, backend)
            optimizer = qml.AdamOptimizer()

            # params = pnp.array([[0.5]*step_layers, [0.5]*step_layers], requires_grad=True)     #param0 is cost, param1 is mixer

            param0 = np.linspace(0, 1, step_layers)
            param1 = np.linspace(0, 1, step_layers)[::-1]  
            params = pnp.array([param0, param1], requires_grad=True)

            for k in range((i+1)*step_multiplier):
                params = optimizer.step(cost_function, params)
            
            res = qaoa_circuit1(params[0], params[1])
        elif hadof_optimiser == "FALQON":
            from problem_solver import falqon
            falqon_circuit1, falqon_circuit2, cost_function = falqon.create_falqon_circuit(problem_device, qml_device, shots1, shots2, num_qubit, h_new, J_new, backend)
            if partial_opt == "layer":
                beta = [0.0]
                for _ in range((i+1)*step_multiplier):
                    beta.append(-1 * cost_function(beta))  
            elif partial_opt == "full":
                beta = [0.0]
                for _ in range((steps)):
                    beta.append(-1 * cost_function(beta))  
            res = falqon_circuit1(beta)
        elif hadof_optimiser == "QAOAt-qiskit":
            from problem_solver import QAOAt_qiskit

            if partial_opt == "layer":
                betas = np.linspace(0, (i+1)/steps, i+1)[::-1]  # Parameters for the mixer Hamiltonian
                gammas = np.linspace(0, (i+1)/steps, i+1)
            elif partial_opt == "full":
                betas = np.linspace(0, 1, step_layers+1)[::-1]  # Parameters for the mixer Hamiltonian
                gammas = np.linspace(0, 1, step_layers+1)
            elif partial_opt == "manual":
                betas = np.linspace(0, (i+1)/steps, step_layers+1)[::-1]  # Parameters for the mixer Hamiltonian
                gammas = np.linspace(0, (i+1)/steps, step_layers+1)  

            qaoa_circuit1, pm, sampler = QAOAt_qiskit.create_qaoa_circuit(problem_device, num_qubit, backend, session)
            qc = qaoa_circuit1(gammas, betas, h_new, J_new, num_qubits=num_qubit)



            qc.measure_all()
            candidate_circuit = pm.run(qc)
            sampler.options.default_shots = shots2


            # Set simple error suppression/mitigation options
            if problem_device == "real_backend":
                sampler.options.dynamical_decoupling.enable = True
                sampler.options.dynamical_decoupling.sequence_type = "XY4"
                # sampler.options.twirling.enable_gates = True
                # sampler.options.twirling.num_randomizations = "auto"



            pub = (candidate_circuit,)
            job = sampler.run([pub], shots=shots2)
            bits = job.result()[0].data.meas.get_bitstrings()
            samps = np.array([list(map(int, s))[::-1] for s in bits], dtype=int)
            res = samps.T




            # candidate_circuit = pm.run(qc)
            # estimator.options.default_shots = shots1 

            # from qiskit.quantum_info import SparsePauliOp

            # def z_on_qubit(k: int, n: int) -> SparsePauliOp:
            #     s = ["I"] * n
            #     s[n - 1 - k] = "Z"          # qubit 0 is rightmost in the string
            #     return SparsePauliOp("".join(s))

            # observables = [z_on_qubit(k, num_qubit) for k in range(num_qubit)]

            # isa_observables = [op.apply_layout(candidate_circuit.layout) for op in observables]

            # if problem_device == "real_backend":
            #     # Zero-Noise Extrapolation (ZNE)
            #     estimator.options.resilience.zne_mitigation = True
            #     estimator.options.resilience.zne.noise_factors = (1, 3, 5)
            #     estimator.options.resilience.zne.extrapolator = "exponential"

            #     # estimator.options.resilience_level = 2

            #     # Dynamical decoupling
            #     estimator.options.dynamical_decoupling.enable = True
            #     estimator.options.dynamical_decoupling.sequence_type = "XY4"

            #     # Twirling
            #     estimator.options.twirling.enable_gates = True
            #     estimator.options.twirling.num_randomizations = "auto"



            # job = estimator.run([(candidate_circuit, isa_observables)])
            # pub_result = job.result()[0]

            # expvals_Z = pub_result.data.evs   # length == num_qubits

            # res = [(1 - z) / 2 for z in expvals_Z]



        partial_solution = [res[i].mean() for i in range(num_qubit)] 


        if selection == "ordered":
            part_sol[j*num_qubit:(j+1)*num_qubit] = partial_solution

        if selection == "random_ordered":
            for ind in range(num_qubit):
                part_sol[arr[j*num_qubit+ind]] = partial_solution[ind]

        if problem_size <= 1000:
            print(f"Step {i+1}/{steps}, Model: {j+1}, Best solution: {partial_solution}")
        else:
            print(f"Step {i+1}/{steps}, Model: {j+1}")

        if i == steps-1:
            if  hadof_optimiser == "QAOAt":
                samps = qaoa_circuit2(gammas, betas, h_new, J_new, num_qubits=num_qubit)
                samples_unbalanced = samples_dict(samps, num_qubit)
                finalsamps.append(samps)
            elif hadof_optimiser == "SA":
                samples_unbalanced = defaultdict(int)
                solver = SimulatedAnnealingSampler()
                # sampleset = solver.sample_ising(list(h_new.values()), J_new, num_reads = shots2, beta_range = [0,1], beta_schedule_type = 'linear', num_sweeps=50)
                sampleset = solver.sample_ising(list(h_new.values()), J_new, num_reads = shots2, anneal_time = 20)
                # h_new2 = {k[0]: v for k, v in h_new.items()}
                # solver = oj.SASampler()
                # sampleset = solver.sample_ising(h_new2, J_new, num_reads=shots2)
                sampleset = sampleset.to_pandas_dataframe().sort_values("energy").reset_index(drop=True)
                sampleset = sampleset.replace(1, 0).replace(-1, 1)
                for num, row in sampleset.iterrows():
                    if len(sampleset) == shots2:
                        sample_i = "".join(str(round(row[q])) for q in range(len(h_new)))
                        samples_unbalanced[sample_i] += 1
                finalsamps.append(sampleset)
            elif hadof_optimiser == "QAOAc":
                samps = qaoa_circuit2(params[0], params[1])
                samples_unbalanced = samples_dict(samps, num_qubit)
                finalsamps.append(samps)
            elif hadof_optimiser == "FALQON":
                samps = falqon_circuit2(beta)
                samples_unbalanced = samples_dict(samps, num_qubit)
                finalsamps.append(samps)
            elif hadof_optimiser == "QAOAt-qiskit":
                qc = qaoa_circuit1(gammas, betas, h_new, J_new, num_qubits=num_qubit)
                qc.measure_all()
                candidate_circuit = pm.run(qc)
                sampler.options.default_shots = shots2


                # Set simple error suppression/mitigation options
                if problem_device == "real_backend":
                    sampler.options.dynamical_decoupling.enable = True
                    sampler.options.dynamical_decoupling.sequence_type = "XY4"
                    # sampler.options.twirling.enable_gates = True
                    # sampler.options.twirling.num_randomizations = "auto"



                pub = (candidate_circuit,)
                job = sampler.run([pub], shots=shots2)
                final_distribution_bin = job.result()[0].data.meas.get_counts()
                samples_unbalanced = {k[::-1]: v for k, v in final_distribution_bin.items()}
                bits = job.result()[0].data.meas.get_bitstrings()
                samps = np.array([list(map(int, s))[::-1] for s in bits], dtype=int)
                finalsamps.append(samps)
                



            final_probs.append(samples_unbalanced)
            sampleprobs.append([res[i].mean() for i in range(num_qubit)])


    return finalsamps, final_probs, sampleprobs