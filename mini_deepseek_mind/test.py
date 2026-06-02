import torch
print('torch:', torch.__version__)
print('cuda:', torch.cuda.is_available())
print('cuda devices:', torch.cuda.device_count())