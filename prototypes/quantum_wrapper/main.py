import torch
import torch.nn as nn
import torch.optim as optim
from quantum_layer import QuantumLayer
from data_processing import extract_mobilenet_features, normalize_iot_data, get_processed_data

class HybridModel(nn.Module):
    def __init__(self, n_classes=4):
        super().__init__()
        self.fc1 = nn.Linear(1010, 4) 
        self.quantum_layer = QuantumLayer(n_qubits=4, n_layers=2)
        self.fc2 = nn.Linear(4, n_classes) 

    def forward(self, image_features, iot_data):
        combined_features = torch.cat((image_features, iot_data), dim=1)
        x = torch.relu(self.fc1(combined_features))
        x = torch.sigmoid(x) * torch.pi 
        
        # Quantum layer processing with device synchronization
        q_out = self.quantum_layer(x).to(image_features.device)
        
        out = self.fc2(q_out)
        return out

def train_loop():
    print("Initializing Tier A Hybrid Quantum-Classical Model...")
    model = HybridModel(n_classes=4)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    epochs = 3
    batch_size = 4
    
    print("\nStarting Training Loop...")
    for epoch in range(epochs):
        mock_images, mock_sensor_data = get_processed_data(batch_size)
        
        image_features = extract_mobilenet_features(mock_images).to(device)
        # Optimized NumPy to PyTorch tensor conversion
        iot_data = torch.from_numpy(normalize_iot_data(mock_sensor_data)).float().to(device)
        
        labels = torch.randint(0, 4, (batch_size,)).to(device)
        
        optimizer.zero_grad()
        outputs = model(image_features, iot_data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f}")
        
    print("\nTraining complete! Quantum parameters successfully updated.")

if __name__ == "__main__":
    train_loop()