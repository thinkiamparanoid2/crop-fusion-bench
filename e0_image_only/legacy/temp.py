import torch

print("PyTorch Version:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())
print("CUDA Device Count:", torch.cuda.device_count())

if torch.cuda.is_available():
    print("CUDA Version (built with):", torch.version.cuda)
    print("Current CUDA Device:", torch.cuda.current_device())
    print("CUDA Device Name:", torch.cuda.get_device_name(torch.cuda.current_device()))
    
    # Try to create a tensor on GPU
    test_tensor = torch.tensor([1.0, 2.0, 3.0]).cuda()
    print("Test tensor on GPU:", test_tensor.device)
else:
    print("⚠️  CUDA is NOT available - PyTorch will use CPU only")