from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from data.dataset import ProteinDataset

def create_dataset(opt):
    data = ProteinDataset(opt)
    dataloader = DataLoader(data, batch_size=opt.batch_size, shuffle=(not opt.no_shuffle), num_workers=int(opt.num_threads))
    return dataloader
