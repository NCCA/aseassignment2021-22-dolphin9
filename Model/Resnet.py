import torch
import torch.nn as nn
import torchsummary as summary

set_frame = 50
epochs = 60
batch_sz = 10
checkpoint_frequency = 3
learning_rate = 0.00005
gamma = 0.5

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample):
        super().__init__()
        if downsample:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            self.shortcut = nn.Sequential()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, inp):
        shortcut = self.shortcut(inp)
        inp = nn.ReLU()(self.bn1(self.conv1(inp)))
        inp = nn.ReLU()(self.bn2(self.conv2(inp)))
        inp = inp + shortcut  # The magic bit that cannot be done with nn.Sequential!
        return nn.ReLU()(inp)
    


class ResNet(nn.Module):
    def __init__(self, in_channels, resblock, outputs):
        super().__init__()
        self.layer0_conv = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.layer0_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer0_bn   = nn.BatchNorm2d(64)
        self.layer0_relu = nn.ReLU()

        self.layer1_res1 = resblock(64, 64, downsample=False)
        self.layer1_res2 = resblock(64, 64, downsample=False)

        self.layer2_res1 = resblock(64, 128, downsample=True)
        self.layer2_res2 = resblock(128, 128, downsample=False)

        self.layer3_res1 = resblock(128, 256, downsample=True)
        self.layer3_res2 = resblock(256, 256, downsample=False)

        self.layer4_res1 = resblock(256, 512, downsample=True)
        self.layer4_res2 = resblock(512, 512, downsample=False)

        self.gap         = nn.AdaptiveAvgPool2d(1)
        self.flat        = nn.Flatten() 
        self.fc          = nn.Linear(512, outputs)

    def forward(self, inp):
        inp = self.layer0_conv(inp)
        inp = self.layer0_pool(inp)
        inp = self.layer0_bn(inp)
        inp = self.layer0_relu(inp)
        
        inp = self.layer1_res1(inp)
        inp = self.layer1_res2(inp)
        
        inp = self.layer2_res1(inp)
        inp = self.layer2_res2(inp)
        
        inp = self.layer3_res1(inp)
        inp = self.layer3_res2(inp)
        
        inp = self.layer4_res1(inp)
        inp = self.layer4_res2(inp)
            
        inp = self.gap(inp)
        inp = self.flat(inp)
        inp = self.fc(inp)

        return inp
    

class PlResNet(pl.LightningModule):
    def __init__(self, in_channels, outputs):
        super().__init__()
        # model arch goes here
        self.model = ResNet(in_channels,ResBlock,outputs)
        
    def forward(self, inp):
        return self.model.forward(inp)
    
    def configure_optimizers(self):
        return SGD(self.parameters(), lr = learning_rate)
    
    def training_step(self, batch, batch_idx):
        image_batch,label_batch = batch
        batch_sz = len(image_batch)
        
        image_batch = image_batch.reshape(batch_sz,1,28,28)
        output = model(image_batch)
        loss = nn.CrossEntropyLoss()(output,label_batch)

        preds_train = torch.argmax(output, dim=1)
        correct_train = int(torch.eq(preds_train, label_batch).sum())
        minibatch_accuracy_train = 100 * correct_train / batch_sz

        self.log("train_acc",minibatch_accuracy_train)
        self.log("train_loss",loss)

    def validation_step(self, batch, batch_idx):
        image_batch,label_batch = batch
        batch_sz = len(image_batch)
        
        image_batch = image_batch.reshape(batch_sz,1,28,28)
        output = model(image_batch)
        loss = nn.CrossEntropyLoss()(output,label_batch)

        preds_val = torch.argmax(output, dim=1)
        correct_val = int(torch.eq(preds_val, label_batch).sum())
        minibatch_accuracy_val = 100 * correct_val / batch_sz

        self.log("val_acc",minibatch_accuracy_val)
        self.log("val_loss",loss)

# convenience function
def get_resnet():
    return ResNet(1, ResBlock, outputs=10)


# Print model summary
summary(get_resnet(), input_size=(1, 28, 28), device="cpu")

