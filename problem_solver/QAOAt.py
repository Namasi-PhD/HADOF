import pennylane as qml
import numpy as np



def create_qaoa_circuit(problem_device, qml_device, shots1, shots2, num_qubit, backend):

    # -----------------------------   QAOA circuit ------------------------------------
    if problem_device == "real_backend" or problem_device == "fake_backend":
        try:
            dev1 = qml.device("qiskit.remote", shots=shots1, wires=num_qubit, backend=backend)
        except Exception as e:
            print(e)
    else:
        dev1 = qml.device(qml_device, shots=shots1, wires=num_qubit)
    # dev = qml.device("lightning.qubit", wires=20)
    @qml.qnode(dev1)
    def qaoa_circuit1(gammas, betas, h, J, num_qubits):
        p = len(gammas)
        # Apply the initial layer of Hadamard gates to all qubits
        for i in range(num_qubits):
            qml.Hadamard(wires=i)
        # repeat p layers the circuit shown in Fig. 1
        for layer in range(p):
            wmax = max(
                np.max(np.abs(list(h.values()))), np.max(np.abs(list(h.values())))
            ) if np.max(np.abs(list(h.values()))) !=0 else 1 
            # ---------- COST HAMILTONIAN ----------
            for ki, v in h.items():  # single-qubit terms
                qml.RZ(2 * gammas[layer] * v / wmax, wires=ki[0])
            for kij, vij in J.items():  # two-qubit terms
                qml.CNOT(wires=[kij[0], kij[1]])
                qml.RZ(2 * gammas[layer] * vij / wmax, wires=kij[1])
                qml.CNOT(wires=[kij[0], kij[1]])
            # ---------- MIXER HAMILTONIAN ----------
            for i in range(num_qubits):
                qml.RX(-2 * betas[layer], wires=i)
        return [qml.sample(wires=[k]) for k in range(num_qubits)]  # Sample from the circuit
        # return qml.probs()



    if problem_device == "real_backend" or problem_device == "fake_backend":
        try:
            dev2 = qml.device("qiskit.remote", shots=shots2, wires=num_qubit, backend=backend)
        except Exception as e:
            print(e)
    else:
        dev2 = qml.device(qml_device, shots=shots2, wires=num_qubit)
    # dev = qml.device("lightning.qubit", wires=20)
    @qml.qnode(dev2)
    def qaoa_circuit2(gammas, betas, h, J, num_qubits):
        p = len(gammas)
        # Apply the initial layer of Hadamard gates to all qubits
        for i in range(num_qubits):
            qml.Hadamard(wires=i)
        # repeat p layers the circuit shown in Fig. 1
        for layer in range(p):
            wmax = max(
                np.max(np.abs(list(h.values()))), np.max(np.abs(list(h.values())))
            ) if np.max(np.abs(list(h.values()))) !=0 else 1 
            # ---------- COST HAMILTONIAN ----------
            for ki, v in h.items():  # single-qubit terms
                qml.RZ(2 * gammas[layer] * v / wmax, wires=ki[0])
            for kij, vij in J.items():  # two-qubit terms
                qml.CNOT(wires=[kij[0], kij[1]])
                qml.RZ(2 * gammas[layer] * vij / wmax, wires=kij[1])
                qml.CNOT(wires=[kij[0], kij[1]])
            # ---------- MIXER HAMILTONIAN ----------
            for i in range(num_qubits):
                qml.RX(-2 * betas[layer], wires=i)
        return qml.sample()  # Sample from the circuit
        # return qml.probs()

    return qaoa_circuit1, qaoa_circuit2