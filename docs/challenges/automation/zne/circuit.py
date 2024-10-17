from numpy import pi
from qiskit.circuit import Parameter, ParameterVector, QuantumCircuit

class ExampleCircuit(QuantumCircuit):
    """Parameterized MBL non-periodic chain (i.e. 1D) quantum circuit.

    Parameters correspond to interaction strength (θ), and
    disorders vector (φ) with one entry per qubit. In 1D,
    θ < 0.16π ≈ 0.5 corresponds to the MBL regime; beyond such
    critical value the dynamics become ergodic (i.e. thermal).
    Disorders are random on a qubit by qubit basis [1].
    
    Args:
        num_qubits: number of qubits (must be even).
        depth: two-qubit depth.
        barriers: if True adds barriers between layers

    Notes:
        [1] Shtanko et.al. Uncovering Local Integrability in Quantum
            Many-Body Dynamics, https://arxiv.org/abs/2307.07552
    """

    def __init__(self, num_qubits: int, depth: int, *, barriers: bool = False) -> None:
        super().__init__(num_qubits)

        theta = Parameter("θ")
        phis = ParameterVector("φ", num_qubits)
        
        self.x(range(1, num_qubits, 2))
        for layer in range(depth):
            layer_parity = layer % 2
            if barriers and layer > 0:
                self.barrier()
            for qubit in range(layer_parity, num_qubits - 1, 2):
                self.cz(qubit, qubit + 1)
                self.u(theta, 0, pi, qubit)
                self.u(theta, 0, pi, qubit + 1)
            for qubit in range(num_qubits):
                self.p(phis[qubit], qubit)