# E3-4 VQC Circuit and Layout

Date: 2026-06-04

This note explains the E3-4 variational quantum circuit (VQC) used in the first
controlled quantum fusion run. It connects the implemented code to the circuit
layout and the theory behind each block.

Primary code:

```text
e3_quantum_fusion/models.py
e3_quantum_fusion/quantum_layer.py
e3_quantum_fusion/scripts/train_e3.py
```

Related specs/results:

```text
e2_multimodal_fusion/PARAM_MATCH_SPEC.md
docs/by_stage/04_e3_quantum_fusion/E2_MATCHED_E3_4_COMPARISON.md
```

## 1. Position in the Full Model

E3-4 does not retrain the image or IoT encoders. It consumes the same frozen
embedding cache used by E2:

```text
E0 image encoder embedding: 1280 dims
E1 IoT encoder embedding:     64 dims
Fused vector:              1344 dims
```

The E3-4 head is:

```text
1344 fused features
  -> Linear(1344,128)
  -> BatchNorm1d(128)
  -> ReLU
  -> Dropout(0.1)
  -> Linear(128,4)
  -> tanh * pi
  -> 4-qubit VQC
  -> Linear(4,25)
  -> logits over 25 crop-disease-stage classes
```

In code, this is `E3HybridHead`:

```python
class E3HybridHead(nn.Module):
    def __init__(self, bottleneck=4, num_classes=25, sim="default.qubit"):
        super().__init__()
        self.projection = FusionProjection(bottleneck=bottleneck)
        self.vqc = VQCLayer(n_qubits=bottleneck, sim=sim)
        self.readout = nn.Linear(bottleneck, num_classes)

    def forward(self, x):
        z = self.projection(x)
        z = pi * torch.tanh(z)
        z = self.vqc(z)
        return self.readout(z)
```

The quantum circuit replaces only the 4-to-4 decision sub-network. The large
1344-to-4 projection and final 4-to-25 readout are classical and are shared with
the matched comparator.

## 2. Why the Bottleneck Is 4

The first E3 model uses 4 qubits. Each scalar in the 4-dimensional bottleneck is
encoded into one qubit as a rotation angle:

```text
bottleneck z = [z0, z1, z2, z3]
qubits      = [q0, q1, q2, q3]
```

The projection output is passed through `tanh` and multiplied by `pi`:

```text
theta_i = pi * tanh(z_i)
```

This bounds each quantum input angle:

```text
theta_i in [-pi, pi]
```

That bound matters because rotation gates are periodic. Without a bounded angle
range, very large classical activations could wrap around the Bloch sphere many
times and make the input encoding unstable or hard to interpret.

## 3. Circuit Layout

For each input sample, the VQC has this structure:

```text
Input angles:
theta0, theta1, theta2, theta3

Angle encoding:
q0: RX(theta0)
q1: RX(theta1)
q2: RX(theta2)
q3: RX(theta3)

Trainable layer 1:
q0: RX(w[0,0,0]) -> RY(w[0,0,1])
q1: RX(w[0,1,0]) -> RY(w[0,1,1])
q2: RX(w[0,2,0]) -> RY(w[0,2,1])
q3: RX(w[0,3,0]) -> RY(w[0,3,1])

CNOT ring:
q0 -> q1
q1 -> q2
q2 -> q3
q3 -> q0

Trainable layer 2:
q0: RX(w[1,0,0]) -> RY(w[1,0,1])
q1: RX(w[1,1,0]) -> RY(w[1,1,1])
q2: RX(w[1,2,0]) -> RY(w[1,2,1])
q3: RX(w[1,3,0]) -> RY(w[1,3,1])

CNOT ring:
q0 -> q1
q1 -> q2
q2 -> q3
q3 -> q0

Measurement:
<Z0>, <Z1>, <Z2>, <Z3>
```

The 4 measured expectation values become a classical 4-vector:

```text
vqc_output = [<Z0>, <Z1>, <Z2>, <Z3>]
```

Each value is in:

```text
[-1, 1]
```

The final classical readout maps those four measured values to class logits:

```text
Linear(4,25)
```

## 4. Implemented PennyLane Circuit

The implemented circuit is in `e3_quantum_fusion/quantum_layer.py`:

```python
@qml.qnode(dev, interface="torch", diff_method=diff_method)
def circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="X")

    for layer in range(n_layers):
        for q in range(n_qubits):
            qml.RX(weights[layer, q, 0], wires=q)
            qml.RY(weights[layer, q, 1], wires=q)

        for q in range(n_qubits):
            qml.CNOT(wires=[q, (q + 1) % n_qubits])

    return [qml.expval(qml.PauliZ(q)) for q in range(n_qubits)]
```

The QNode is wrapped as a Torch layer:

```python
weight_shapes = {"weights": (n_layers, n_qubits, 2)}
self.qlayer = qml.qnn.TorchLayer(circuit, weight_shapes)
```

This lets PyTorch optimize the quantum parameters together with the surrounding
classical projection and readout.

## 5. Trainable Parameter Count

The VQC uses:

```text
n_layers = 2
n_qubits = 4
trainable rotations per qubit per layer = 2  # RX and RY
```

So the VQC has:

```text
2 layers * 4 qubits * 2 rotations = 16 trainable parameters
```

In tensor form:

```text
weights.shape = [2, 4, 2]
```

This exactly matches the classical comparator's middle layer:

```text
Linear(4,4,bias=False) = 4 * 4 = 16 trainable parameters
```

The total head parameters are also matched:

| Component | Parameters |
|---|---:|
| Shared projection, `1344 -> 128 -> 4` | 172,932 |
| Decision sub-network | 16 |
| Readout, `4 -> 25` | 125 |
| Total | 173,073 |

This is why the controlled claim is E2-matched vs E3-4, not E2-current vs E3-4.
E2-current is the stronger classical baseline, while E2-matched is the fair
parameter-matched comparator.

## 6. Entanglement Choice

The CNOT ring connects every qubit to a neighbor and closes the loop:

```text
0 -> 1 -> 2 -> 3 -> 0
```

This allows the model to represent interactions among the four bottleneck
features. Without entanglement, each qubit would mostly process its own encoded
feature independently. With the ring, the circuit can learn joint effects among
image-IoT fused features after compression.

The ring is also small and predictable. It avoids a dense all-to-all
entanglement pattern, which would add circuit depth and make the circuit harder
to simulate or run on NISQ-style hardware.

## 7. Measurement and Readout

Each qubit is measured using Pauli-Z expectation:

```text
<Z_i> = P(q_i = 0) - P(q_i = 1)
```

The output is continuous, not a sampled bit:

```text
<Z_i> in [-1, 1]
```

Those four values are treated as learned quantum features. The final
`Linear(4,25)` layer converts them to logits for the 25 labels. Cross entropy is
then applied to the logits in the usual PyTorch training loop.

## 8. Gradient Method and Device

The VQC supports two CPU simulators in the current environment:

```text
default.qubit
lightning.qubit
```

The implementation chooses the differentiation method by simulator:

```python
diff_method = "backprop" if sim == "default.qubit" else "adjoint"
```

Reason:

```text
default.qubit supports backprop for this Torch QNode.
lightning.qubit does not support backprop for this circuit, but works with adjoint.
```

The completed E3-4 run used:

```text
simulator: lightning.qubit
device: CPU
```

PyTorch CUDA is available for the classical models, but `lightning.gpu` is not
installed in the current Windows environment, so the VQC itself did not run on
GPU.

## 9. Tensor Shape Flow

For batch size `B`, the E3-4 tensor flow is:

| Step | Shape |
|---|---:|
| Fused cached embeddings | `[B, 1344]` |
| Shared projection output | `[B, 4]` |
| Angle-scaled bottleneck | `[B, 4]` |
| VQC output | `[B, 4]` |
| Final logits | `[B, 25]` |

The smoke test verifies this on real validation embeddings:

```text
E3-4 bottleneck: [8, 4]
E3-4 logits:     [8, 25]
total params:    173,073
post params:     16
```

## 10. Thesis Interpretation

The VQC should be described as a compact nonlinear decision sub-network operating
on a learned 4-dimensional multimodal bottleneck. It is not a quantum image
encoder and it does not process raw IoT time series directly.

Suggested thesis wording:

```text
The hybrid model first projects frozen image and environmental embeddings into a
four-dimensional bottleneck. The bottleneck values are bounded with tanh and
scaled to rotation angles in [-pi, pi], then encoded into a four-qubit variational
quantum circuit using RX angle encoding. Two trainable RX/RY layers with CNOT-ring
entanglement transform the quantum state, and Pauli-Z expectation values from the
four qubits form a four-dimensional quantum feature vector. A final classical
linear readout maps these features to the 25 crop-disease-stage classes.
```

Important limitation:

```text
Any performance difference should be attributed only to replacing the matched
4-to-4 decision sub-network, because the frozen encoders, 1344-dimensional input,
projection, readout, split, labels, and evaluation metrics are held fixed.
```
