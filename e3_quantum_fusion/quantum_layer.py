"""E3 variational quantum circuit (VQC) layer for the hybrid fusion head.

Locked first-pass configuration (see e2_multimodal_fusion/PARAM_MATCH_SPEC.md sec 5
and docs/plans/E3_STRUCTURE_AND_IMPLEMENTATION_PLAN.md):

    qubits:        4
    encoding:      angle encoding (the 4-dim bottleneck is scaled by tanh*pi
                   upstream, then fed as rotation angles)
    ansatz depth:  2 layers
    trainable:     RX and RY per qubit per layer  => 2 * 4 * 2 = 16 params
    entanglement:  CNOT ring
    measurement:   Pauli-Z expectation per qubit  => 4 outputs
    simulator:     default.qubit (CPU) for the first pass
                   (lightning.qubit is installed and faster for full training)
"""

import pennylane as qml  # noqa: F401  (used once the QNode is implemented)
import torch
import torch.nn as nn

N_QUBITS = 4
N_LAYERS = 2
# 2 rotations (RX, RY) per qubit per layer
N_TRAINABLE = N_LAYERS * N_QUBITS * 2  # == 16
WEIGHT_SHAPES = {"weights": (N_LAYERS, N_QUBITS, 2)}


class VQCLayer(nn.Module):
    """4-qubit VQC mapping a [B, 4] bottleneck to [B, 4] Pauli-Z expectations.

    Replaces the classical ``Linear(4, 4, bias=False)`` decision sub-network in the
    matched comparison. ``sum(p.numel() for p in self.parameters())`` must equal
    ``N_TRAINABLE`` (16) so E3-4 matches E2-matched on the post-bottleneck head.
    """

    def __init__(self, n_qubits: int = N_QUBITS, n_layers: int = N_LAYERS, sim: str = "default.qubit"):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.sim = sim

        dev = qml.device(sim, wires=n_qubits)
        diff_method = "backprop" if sim == "default.qubit" else "adjoint"

        @qml.qnode(dev, interface="torch", diff_method=diff_method)
        def circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="X")
            for layer in range(n_layers):
                for q in range(n_qubits):
                    qml.RX(weights[layer, q, 0], wires=q)
                    qml.RY(weights[layer, q, 1], wires=q)
                for q in range(n_qubits):  # CNOT ring entanglement
                    qml.CNOT(wires=[q, (q + 1) % n_qubits])
            return [qml.expval(qml.PauliZ(q)) for q in range(n_qubits)]

        weight_shapes = {"weights": (n_layers, n_qubits, 2)}
        self.qlayer = qml.qnn.TorchLayer(circuit, weight_shapes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, n_qubits] already scaled into angles (tanh*pi applied upstream)
        # returns: [B, n_qubits] stacked Pauli-Z expectations
        return self.qlayer(x)
