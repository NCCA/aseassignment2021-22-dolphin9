import BandaiDataset as bd
import torch
from torch.utils.data import DataLoader, random_split
import multiprocessing as mp


MODEL_DIR = "./saved_models"
DATASET_DIR = "./datasets/data/"
FILELIST_PATH = "datafiles.txt"
set_frame = 50

dataset = bd.BandaiDataset(FILELIST_PATH)

def custom_collate_fn(batch):
    motion_batch_tensor = torch.FloatTensor(len(batch),50,640,480)
    motion_tensors = []
    labels = []

    for item in batch:
        motion_tensor = dataset[item[0]].get_motion_tensor(50) # load an motion as a tensor(frames,width,height)
        motion_tensors.append(motion_tensor) # put motions into a list : to be checked 
        labels.append(dataset[item[0]].label)

    torch.cat(motion_tensors, out=motion_batch_tensor)
    label_batch_tensor = torch.LongTensor(labels)
    return (motion_batch_tensor,label_batch_tensor)

def load_data(file_list_path= '', data_path='', batch_sz = 5, train_val_test_split = [0.7,0.1,0.2]):
    assert sum(train_val_test_split) == 1, "Train, val and test fractions should sum to 1!" 
    dataset = bd.BandaiDataset(data_path)
    dataset.load()

    tr_va_te = []
    n_cpus = mp.cpu_count()
    
    for frac in train_val_test_split:
        num = round(frac * dataset.num_of_files)
        tr_va_te.append(num)
    
    if tr_va_te[0] != (dataset.num_of_files - tr_va_te[1] - tr_va_te[2]):
        tr_va_te[0] = (dataset.num_of_files - tr_va_te[1] - tr_va_te[2])
    
    #assert 1==2, f"tr_va_te = {tr_va_te}"

    train_split, val_split, test_split = random_split(dataset, tr_va_te)

    train_dl = DataLoader(train_split,
                          batch_size=batch_sz,
                          shuffle=True,
                          collate_fn=custom_collate_fn,
                          num_workers=n_cpus
                        )
    val_dl = DataLoader(val_split,
                        batch_size=batch_sz,
                        shuffle=True,
                        collate_fn=custom_collate_fn,
                        num_workers=n_cpus)
    test_dl = DataLoader(test_split,
                         batch_size=batch_sz,
                         shuffle=True,
                         collate_fn=custom_collate_fn,
                         num_workers=n_cpus)

    return train_dl, val_dl, test_dl