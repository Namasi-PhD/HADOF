# HADOF

**Scalable Quantum Optimisation using HADOF: Hamiltonian Auto-Decomposition Optimisation Framework**

**Paper Reference:**

Namasi G. Sankar, Georgios Miliotis, and Simon Caton. Scalable quantum optimisation using HADOF: Hamiltonian auto-decomposition optimisation framework. In Proceedings of the 3rd International Workshop on AI for Quantum and Quantum for AI (AIQxQIA 2025), co-located with the 28th European Conference on Artificial Intelligence (ECAI 2025), volume 4153 of CEUR Workshop Proceedings, pages 63–72, Bologna, Italy, 2025. CEUR-WS.org.

**BibTex:**

@inproceedings{Sankar2025HADOF,
  author    = {Namasi G. Sankar and Georgios Miliotis and Simon Caton},
  title     = {Scalable Quantum Optimisation using HADOF: Hamiltonian Auto-Decomposition Optimisation Framework},
  booktitle = {Proceedings of the 3rd International Workshop on AI for Quantum and Quantum for AI (AIQxQIA 2025), co-located with the 28th European Conference on Artificial Intelligence (ECAI 2025)},
  series    = {CEUR Workshop Proceedings},
  volume    = {4153},
  pages     = {63--72},
  year      = {2025},
  publisher = {CEUR-WS.org},
  address   = {Bologna, Italy},
  url       = {https://ceur-ws.org/Vol-4153/paper9.pdf}
}

**For more information on my work:** [My Website](https://namasi-phd.github.io)

**Introduction**

QUBO is a general framework for many NP-hard combinatorial optimisation problems and maps naturally to quantum Hamiltonians, making it suitable for both quantum and classical optimisation methods. While quantum algorithms such as QAOA, QA, GAS, and FALQON may offer advantages for optimisation, current NISQ hardware is too limited for many practical QUBO instances.

To address this, HADOF iteratively decomposes a global quantum Hamiltonian into small sub-Hamiltonians and refines a global solution using sampled distributions. HADOF is compatible with optimisation methods that return sampleable distributions biased toward good solutions. This allows it to scale beyond available qubits while preserving access to multiple good candidate solutions.

**General Overview** HADOF proceeds iteratively: 1) Encode the full QUBO as a Hamiltonian. 2) Solve small sub-Hamiltonians using a sampling-based optimiser. 3) Estimate marginal variable probabilities from samples. 4) Update the sub-Hamiltonians using this global information. 5) Aggregate sampled sub-solutions into global candidate solutions.

<img width="3780" height="1890" alt="overview" src="https://github.com/user-attachments/assets/0d1c9c29-8579-418a-8b20-e62cfde03275" />


**Results and Discussion**

We compared HADOF against classically simulated Pennylane QAOA, D-Wave Ocean SA, and IBM CPLEX on 100 random QUBO instances per problem size, using the same instances across methods. For n=10 to 80, objectives were scaled to CPLEX; for larger problems, where CPLEX became intractable, they were scaled to global SA.

**Scalability and Runtime** CPLEX scales exponentially, while SA and HADOF scale much better. Results showed that HADOF remains practical up to 500 variables. The 5-qubit HADOF QAOA is faster than the 10-qubit version, while global SA is the fastest overall.

**Solution Quality** For small problems, SA matches CPLEX. HADOF loses some accuracy as size increases, but remains competitive: HADOF SA stays near 0.98, while HADOF QAOA stays around 0.975 for larger problems, with 5-qubit circuits generally outperforming 10-qubit circuits. 

**Modularity** HADOF is modular and can be combined with any optimiser that produces a sampleable distribution biased toward good solutions. In this work we use SA and QAOA, but QA and FALQON are also compatible.

**Testing on a Real Device** HADOF QAOA was run on a 20-variable QUBO using IBM quantum hardware through QiskitRuntimeService, with 5-qubit circuits and 10 layers. The result achieved 0.84 of the CPLEX objective and took 6m 42s including queue and classical overhead, showing that the framework is directly executable on real hardware.

**Summary** HADOF enables hardware-efficient optimisation by solving only small subproblems at a time, independent of global problem size. It scales to at least n=500, maintains near-optimal objective values, and returns multiple high-quality candidate solutions.




