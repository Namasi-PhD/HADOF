import numpy as np
from qiskit import QuantumCircuit


def create_qaoa_circuit(problem_device, num_qubit, backend, session):


    if problem_device == "CPU":
        from qiskit_aer import AerSimulator
        backend = AerSimulator(method='statevector')


    def qaoa_circuit1(gammas, betas, h, J, num_qubits):
        """
        Build a QAOA circuit in Qiskit equivalent to the PennyLane qaoa_circuit2.
        
        gammas, betas: lists of parameters (can be Parameter objects or floats)
        h: dict with keys as (i,) and values as coefficients  -- single qubit terms
        J: dict with keys as (i,j) and values as coupling coefficients -- two-qubit terms
        num_qubits: number of qubits
        """
        p = len(gammas)
        qc = QuantumCircuit(num_qubits)

        # ----- Initial layer of Hadamard gates -----
        for i in range(num_qubits):
            qc.h(i)

        # Repeat p QAOA layers
        for layer in range(p):
            # compute normalisation wmax (same as your code)
            coeff_vals = list(h.values()) + list(J.values())
            wmax = max(abs(x) for x in coeff_vals) if any(coeff_vals) else 1
            
            # ---------- COST HAMILTONIAN ----------
            # Single-qubit Z terms
            for (i,), v in h.items():
                qc.rz(2 * gammas[layer] * v / wmax, i)

            # Two-qubit ZZ terms (via CNOT, RZ, CNOT decomposition)
            for (i, j), v in J.items():
                qc.cx(i, j)
                qc.rz(2 * gammas[layer] * v / wmax, j)
                qc.cx(i, j)
                # qc.rzz(2 * gammas[layer] * v / wmax, i, j)


            # ---------- MIXER HAMILTONIAN ----------
            for q in range(num_qubits):
                qc.rx(-2 * betas[layer], q)

        return qc
    
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    pm = generate_preset_pass_manager(optimization_level=3, backend=backend)
    from qiskit_ibm_runtime import EstimatorV2 as Estimator
    from qiskit_ibm_runtime import SamplerV2 as Sampler

    if problem_device == "real_backend" and session is not None:
        # estimator = Estimator(mode=session)
        sampler = Sampler(mode=session)
    else:
        # estimator = Estimator(backend=backend)
        sampler = Sampler(backend=backend)



    return qaoa_circuit1, pm, sampler