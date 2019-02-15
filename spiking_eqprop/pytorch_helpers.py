import torch
_DEFAULT_DEVICE = 'cpu'


def set_default_device(device):
    global _DEFAULT_DEVICE
    _DEFAULT_DEVICE = device


def get_default_device():
    return _DEFAULT_DEVICE


def setup_cuda_if_available(model):
    cuda = torch.cuda.is_available()
    if cuda:
        model.to('cuda')
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
        set_default_device('cuda')
    else:
        torch.set_default_tensor_type(torch.FloatTensor)
    print(f'Cuda: {cuda}')
    return cuda


def to_default_tensor(data):
    return torch.tensor(data).to(torch.get_default_dtype())


