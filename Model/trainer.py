import Model.BandaiDataset as bd
import numpy as np
import torch    
import torch.nn as nn
from torchsummary import summary
import multiprocessing as mp
from torchsummary import summary
from torch.utils.data import DataLoader, random_split

import Model.mymodel as mymodel
import Model.Resnet as Resnet

set_frame = 50
epochs = 6
batch_sz = 8
checkpoint_frequency = 3
learning_rate = 0.00005
gamma = 0.5

SAVE_DIR = '../save_models'

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EarlyStopper:
    def __init__(self, patience =  1, tolerance = 0):
        self.patience = patience
        self.tolerance = tolerance

        self.epoch_counter = 0
        self.max_validation_acc = np.NINF

    def should_stop(self, validation_acc):
        if validation_acc > self.max_validation_acc:
            self.max_validation_acc = validation_acc
            self.epoch_counter = 0
        elif validation_acc < (self.max_validation_acc - self.tolerance):
            self.epoch_counter += 1
            if(self.epoch_counter >= self.patience):
                return True
        return False
    

def get_simple_conv_net():
    model = mymodel.ConvNet()
    summary(model, input_size=(50, 640, 480), device='CPU')
    return model

def get_resnet():
    model = Resnet.ResNet(Resnet.ResBlock, 50, outputs=10)
    summary(model, input_size=(50,640,480),device='CPU')
    return model


def custom_collate_fn(batch):
    motion_batch_tensor = torch.FloatTensor(len(batch),50,480,640)
    motion_tensors = []
    labels = []
    #print(type(batch))

    for item in batch:
        #print(item)
        motion_tensor = item.get_motion_tensor(50) # load an motion as a tensor(frames,width,height)
        motion_tensors.append(motion_tensor.unsqueeze(0)) # put motions into a list : to be checked 
        labels.append(item.label)

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

def save_checkpoint(model, epoch, save_dir = SAVE_DIR):
    filename = f"checkout_{epoch}.pth"
    save_path = f"{save_dir}/{filename}"
    torch.save(model.state_dict(), save_path)

    motion = bd.Motion()
    motion.adjust(set_frame)

def train_model(model, epochs, dataloaders,
                      optimizer, lr_scheduler, writer,
                      early_stopper,checkpoint_frequency):
    msg = ""

    for epoch in range(epochs):
        ################# TRAINING ####################
        model.train()
        train_dl = dataloaders['train']

        total_steps_train = len(train_dl)
        correct_train = 0
        total_train = 0
        loss_train = 0

        for batch_num, (motion_batch, label_batch) in enumerate(train_dl):
            batch_sz = len(motion_batch)
            label_batch = label_batch.to(DEVICE)
            motion_batch = motion_batch.to(DEVICE)
            output = model(motion_batch)
            loss_train = nn.CrossEntropyLoss()(output, label_batch)

            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()

            preds_train = torch.argmax(output, dim=1)
            correct_train += int(torch.eq(preds_train, label_batch).sum())
            total_train += batch_sz
            minibatch_accuracy_train = 100 * correct_train / total_train

            #### Fancy printing stuff, you can ignore this! ######
            if (batch_num + 1) % 4 == 0:
                print(" " * len(msg), end='\r')
                msg = f'Train epoch[{epoch+1}/{epochs}], MiniBatch[{batch_num + 1}/{total_steps_train}], Loss: {loss_train.item():.5f}, Acc: {minibatch_accuracy_train:.5f}, LR: {lr_scheduler.get_last_lr()[0]:.5f}'
                print (msg, end='\r' if epoch < epochs else "\n",flush=True)
            #### Fancy printing stuff, you can ignore this! ######
                
        lr_scheduler.step()

        ########################################################################
        print("") # Create newline between progress bars
        #######################VALIDATION STEP##################################

        model.eval()
        val_dl = dataloaders['val']

        total_steps_val = len(val_dl)
        correct_val = 0
        total_val = 0
        loss_val = 0

        for batch_num, (motion_batch,label_batch) in enumerate(val_dl):
            batch_sz = len(motion_batch)
            motion_batch = motion_batch.to(DEVICE)
            label_batch = label_batch.to(DEVICE)

            with torch.no_grad():
                output = model(motion_batch)
                #print(output)
                loss_val = nn.CrossEntropyLoss()(output, label_batch)

                preds_val = torch.argmax(output, dim = 1)

                correct_val += int(torch.eq(preds_val, label_batch).sum())
                total_val += batch_sz
                minibatch_accuracy_val = 100 * correct_val / total_val

                #### Fancy printing stuff, you can ignore this! ######
                if (batch_num + 1) % 4 == 0:
                    print(" " * len(msg), end='\r')
                    msg = f'Eval epoch[{epoch+1}/{epochs}], MiniBatch[{batch_num + 1}/{total_steps_val}], Loss: {loss_val.item():.5f}, Acc: {minibatch_accuracy_val:.5f}'
                    if early_stopper.epoch_counter > 0:
                        msg += f", Epochs without improvement: {early_stopper.epoch_counter}"
                    print (msg, end='\r' if epoch < epochs else "\n",flush=True)
                #### Fancy printing stuff, you can ignore this! ######
                    
        ########################################################################
        print("")  # Create newline between progress bars
        ########################################################################

        epoch_train_acc = 100 * correct_train / total_steps_train
        epoch_val_acc = 100 * correct_val/ total_steps_val
        writer.add_scalar("loss/train",loss_train,epoch)
        writer.add_scalar("loss/train",loss_val,epoch)
        writer.add_scalar("Acc/train",epoch_train_acc,epoch)
        writer.add_scalar("Acc/val", epoch_val_acc,epoch)

        #if epoch % checkpoint_frequency == 0:
            #save_checkpoint(model, epoch, "./saved_models")
        if early_stopper.should_stop(epoch_val_acc):
            print(f"\nValidation accuracy has not improved in {early_stopper.epoch_counter} epochs, stopping.")
            #save_checkpoint(model,epoch,"./saved_models")
            return writer

def deployment(last_epoch:int = 57):
    """
        return Torch.load(), dataloders

        dataloaders = {
            'train': train_dl,
            'val': val_dl,
            'test': test_dl
        }
    """
    loaded_net_state_dic = torch.load(f"./saved_models/checkout_{last_epoch}.pth")
    train_dl, val_dl, test_dl = load_data()
    dataloaders = {
    'train': train_dl,
    'val': val_dl,
    'test': test_dl
    }
    return loaded_net_state_dic, dataloaders

def test_model(model, dataloaders):
    model.eval()
    correct = 0
    total = 0
    
    test_dl = dataloaders['test']
    total_steps = len(test_dl)
    msg = ""
    for batch_num, (motion_batch, label_batch) in enumerate(test_dl):
        batch_sz = len(motion_batch)
        label_batch = label_batch.to(DEVICE)
        motion_batch = motion_batch.to(DEVICE)
        out = model(motion_batch)
        preds = torch.argmax(out, dim=1)
        correct += int(torch.eq(preds, label_batch).sum())
        total += label_batch.shape[0]
        if (batch_num + 1) % 5 == 0:
            print(" " * len(msg), end='\r')
            msg = f'Testing batch[{batch_num + 1}/{total_steps}]'
            print (msg, end='\r' if batch_num < total_steps else "\n", flush=True)
    print(f"\nFinal test accuracy for {total} examples: {100 * correct/total:.5f}")