import torch
import numpy as np

def extract_mobilenet_features(image_batch):
    """
    Simulates where teammates extract MobileNetV3 image features.
    """
    # Mocking MobileNetV3 feature extraction
    # Normally, this would be:
    # model = torchvision.models.mobilenet_v3_small(pretrained=True)
    # features = model(image_batch)
    print("Extracting features using MobileNetV3...")
    batch_size = image_batch.shape[0] if hasattr(image_batch, 'shape') else 1
    # Returning mock features (e.g., 1000 dimensional output from MobileNetV3)
    return torch.rand(batch_size, 1000)

def normalize_iot_data(sensor_data):
    """
    Simulates where teammates normalize IoT data.
    """
    print("Normalizing IoT sensor data...")
    # Mock normalization: mean 0, std 1
    mean = np.mean(sensor_data, axis=0)
    std = np.std(sensor_data, axis=0)
    # Avoid division by zero
    std = np.where(std == 0, 1e-6, std)
    return (sensor_data - mean) / std

def get_processed_data(batch_size=4):
    """
    Mock function to get a combined batch of raw simulated data.
    """
    mock_images = torch.rand(batch_size, 3, 224, 224)
    mock_sensor_data = np.random.rand(batch_size, 10)
    
    return mock_images, mock_sensor_data
