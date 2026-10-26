import pennylane as qml
import torch
import torch.nn as nn

class QuantumLayer(nn.Module):
    """
    Self-contained Variational Quantum Circuit layer.
    """
    def __init__(self, n_qubits: int = 4, n_layers: int = 2, device_name: str = "default.qubit",**device_kwargs):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        # Device and QNode initialized dynamically per instance
        dev = qml.device(device_name, wires=n_qubits,**device_kwargs)

        @qml.qnode(dev, interface="torch")
        def _circuit(inputs, weights):
            qml.AngleEmbedding(features=inputs, wires=range(n_qubits), rotation='X')
            for layer in range(n_layers):
                for i in range(n_qubits):
                    qml.RY(weights[layer, i, 0], wires=i)
                    qml.RX(weights[layer, i, 1], wires=i)
                for i in range(n_qubits):
                    qml.CNOT(wires=[i, (i + 1) % n_qubits])
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        weight_shapes = {"weights": (n_layers, n_qubits, 2)}
        self.qlayer = qml.qnn.TorchLayer(_circuit, weight_shapes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # PennyLane default.qubit executes on CPU; handle cross-device tensor transfer
        return self.qlayer(x.cpu())

    def parameter_count(self) -> int:
        """Returns trainable parameter count for Tier A/B matching protocol."""
        return self.n_qubits * self.n_layers * 2


