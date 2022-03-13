from torch.utils.data import Dataset
import torch
import numpy as np

seed = 5
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False


class SarcasmDataset(Dataset):
    def __init__(self, data, labels, lengths):
        """
        PyTorch dataset class
        Args:
            data - list[list[]]
            labels - list()
        Return:
            None
        """
        self.data = data
        self.labels = labels
        self.lengths = lengths

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        labels = np.array(self.labels[index])
        labels = torch.from_numpy(labels).long()
        sen = self.data[index]
        l = self.lengths[index]
        # print(f"l.shape: {l.shape}")

        return sen, labels, l
