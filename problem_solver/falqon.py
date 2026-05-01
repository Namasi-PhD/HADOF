import pennylane as qml
import numpy as np

def build_ising_hamiltonian(h_dict, J_dict):
    """
    Build an Ising Hamiltonian using PennyLane with input dictionaries.
    
    Args:
        h_dict (dict): Dictionary of local fields with keys as one-element tuples.
                       Example: {(0,): -51.67, (1,): -48.08, ...}
        J_dict (dict): Dictionary of couplings with keys as two-element tuples.
                       Example: {(0, 1): 19.1292, (0, 2): 19.1292, ...}
    
    Returns:
        qml.Hamiltonian: The constructed Ising Hamiltonian.
    """
    coeffs = []
    obs = []
    
    # Process local field terms: h_i * Z_i
    for key, coeff in h_dict.items():
        if len(key) != 1:
            raise ValueError("Local field dictionary keys must be tuples of length 1.")
        qubit = key[0]
        coeffs.append(coeff)
        obs.append(qml.PauliZ(qubit))
    
    # Process coupling terms: J_ij * Z_i Z_j
    for key, coeff in J_dict.items():
        if len(key) != 2:
            raise ValueError("Coupling dictionary keys must be tuples of length 2.")
        qubit1, qubit2 = key
        coeffs.append(coeff)
        obs.append(qml.PauliZ(qubit1) @ qml.PauliZ(qubit2))
    
    return qml.Hamiltonian(coeffs, obs)

def mixer(qubits):
    """
    Build a driver Hamiltonian H_d = sum_{k} X_k.
    """
    coeffs = []
    obs = []
    for k in range(qubits):
        coeffs.append(1.0)
        obs.append(qml.PauliX(k))
    return qml.Hamiltonian(coeffs, obs)

def build_commutator_hamiltonian(h_dict, J_dict):
    """
    Build the commutator operator i[H_d, H_c] for the Ising model
    where the cost Hamiltonian is:
    
        H_c = sum_i h_i Z_i + sum_{i<j} J_{ij} Z_i Z_j,
    
    and the driver Hamiltonian is given by:
    
        H_d = sum_k X_k.
    
    Using the Pauli commutation relations,
    
        i[X_k, Z_k] = 2 Y_k,
        i[X_k, Z_iZ_j] = 2δ_{k,i} Y_k Z_j + 2δ_{k,j} Z_i Y_k,
    
    it follows that:
    
        i[H_d, H_c] = 2 ∑_i h_i Y_i + 2 ∑_{i<j} J_{ij} (Y_i Z_j + Z_i Y_j).
    
    Args:
        h_dict (dict): Local fields as one-element tuple keys.
        J_dict (dict): Couplings as two-element tuple keys.
    
    Returns:
        qml.Hamiltonian: The commutator Hamiltonian.
    """
    coeffs = []
    obs = []
    
    # Local field contributions:
    # For each qubit i: 2 * h_i Y_i.
    for key, h in h_dict.items():
        if len(key) != 1:
            raise ValueError("Local field keys must be tuples of length 1.")
        i = key[0]
        coeffs.append(2 * h)
        obs.append(qml.PauliY(i))
    
    # Coupling contributions:
    # For each coupling (i,j): add 2 J_{ij} (Y_i Z_j + Z_i Y_j).
    for key, J in J_dict.items():
        if len(key) != 2:
            raise ValueError("Coupling keys must be tuples of length 2.")
        i, j = key
        coeffs.append(2 * J)
        obs.append(qml.PauliY(i) @ qml.PauliZ(j))
        coeffs.append(2 * J)
        obs.append(qml.PauliZ(i) @ qml.PauliY(j))
    
    return qml.Hamiltonian(coeffs, obs)




def create_falqon_circuit(problem_device, qml_device, shots1, shots2, num_qubit, h, J, backend):


    # -----------------------------   QAOA circuit ------------------------------------
    if problem_device == "real_backend" or problem_device == "fake_backend":
        try:
            dev1 = qml.device("qiskit.remote", shots=shots1, wires=num_qubit, backend=backend)
        except Exception as e:
            print(e)
    else:
        dev1 = qml.device(qml_device, shots=shots1, wires=num_qubit)


    H_cost = build_ising_hamiltonian(h, J)
    H_driver = mixer(qubits=num_qubit)  # 12 qubits, matching our h and J dictionaries
    H_comm = build_commutator_hamiltonian(h, J)



    def falqon_layer(beta_k, cost_h, driver_h, delta_t):
        qml.ApproxTimeEvolution(cost_h, delta_t, 1)
        qml.ApproxTimeEvolution(driver_h, delta_t * beta_k, 1)

    delta_t = 0.05
    def build_ansatz(cost_h, driver_h, delta_t):
        def ansatz(beta, **kwargs):
            layers = len(beta)
            for w in dev1.wires:
                qml.Hadamard(wires=w)
            qml.layer(
                falqon_layer,
                layers,
                beta,
                cost_h=cost_h,
                driver_h=driver_h,
                delta_t=delta_t
            )

        return ansatz

    @qml.qnode(dev1, interface="autograd")
    def expval_circuit(beta):
        ansatz = build_ansatz(H_cost, H_driver, delta_t)
        ansatz(beta)
        return qml.expval(H_comm)
    
    
    @qml.qnode(dev1, interface="autograd")
    def prob_circuit1(res_beta):
        ansatz = build_ansatz(H_cost, H_driver, delta_t)
        ansatz(res_beta)
        # return qml.probs(wires=dev.wires)
        return [qml.sample(wires=[k]) for k in range(num_qubit)]
    
    if problem_device == "real_backend":
        try:
            dev2 = qml.device("qiskit.remote", shots=shots2, wires=num_qubit, backend=backend)
        except Exception as e:
            print(e)
    else:
        dev2 = qml.device(qml_device, shots=shots2, wires=num_qubit)

    @qml.qnode(dev2, interface="autograd")
    def prob_circuit2(res_beta):
        ansatz = build_ansatz(H_cost, H_driver, delta_t)
        ansatz(res_beta)
        # return qml.probs(wires=dev.wires)
        return qml.sample()
    
    return prob_circuit1, prob_circuit2, expval_circuit