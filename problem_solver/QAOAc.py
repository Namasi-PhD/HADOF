import pennylane as qml
from pennylane import qaoa

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


 

def create_qaoac_circuit(problem_device, qml_device, shots1, shots2, num_qubit, step_layers, h, J, backend):

    # -----------------------------   QAOA circuit ------------------------------------
    if problem_device == "real_backend" or problem_device == "fake_backend":
        try:
            dev1 = qml.device("qiskit.remote", shots=shots1, wires=num_qubit, backend=backend)
        except Exception as e:
            print(e)
    else:
        dev1 = qml.device(qml_device, shots=shots1, wires=num_qubit)

    H_cost = build_ising_hamiltonian(h, J)
    H_driver = mixer(qubits=num_qubit)

    def qaoa_layer(gamma, alpha):
        qaoa.cost_layer(gamma, H_cost)
        qaoa.mixer_layer(alpha, H_driver)

    wires = range(num_qubit)

    def circuit(params, **kwargs):
        for w in wires:
            qml.Hadamard(wires=w)
        qml.layer(qaoa_layer, step_layers, params[0], params[1])

    @qml.qnode(dev1)
    def cost_function(params):
        circuit(params)
        return qml.expval(H_cost)

    @qml.qnode(dev1)
    def probability_circuit1(gamma, alpha):
        circuit([gamma, alpha])
        return [qml.sample(wires=[k]) for k in range(num_qubit)]
    
    if problem_device == "real_backend" or problem_device == "fake_backend":
        try:
            dev2 = qml.device("qiskit.remote", shots=shots2, wires=num_qubit, backend=backend)
        except Exception as e:
            print(e)
    else:
        dev2 = qml.device(qml_device, shots=shots2, wires=num_qubit)

    @qml.qnode(dev2)
    def probability_circuit2(gamma, alpha):
        circuit([gamma, alpha])
        return qml.sample()

    
    return probability_circuit1, probability_circuit2, cost_function





