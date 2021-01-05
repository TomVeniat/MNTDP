from torch.utils.data import Dataset


class MyTensorDataset(Dataset):
    def __init__(self, *tensors, transforms=None):
        if transforms:
            assert tensors[0][0].dim() == 3  # Only Images for now
        self.transforms = transforms

        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors

    def __getitem__(self, index):
        res = [tensor[index] for tensor in self.tensors]
        if self.transforms:
            res[0] = self.transforms(res[0])
        return tuple(res)

    def __len__(self):
        return self.tensors[0].size(0)
