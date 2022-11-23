import torch

tensor = torch.rand(3, 4)
## print(f"Device tensor is stored on: {tensor.device}")

print("is availabel", torch.cuda.is_available())

tensor = tensor.to('cuda')
print(f"Device tensor is stored on: {tensor.device}")

print("count", torch.cuda.device_count())

print(torch.__version__)
print(torch.cuda)
print(torch.cuda.get_device_name(0))
print(torch.cuda.current_device())
print(torch.version.cuda)
