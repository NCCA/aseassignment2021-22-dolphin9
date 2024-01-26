
import torch.nn as nn

class ConvNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=50, out_channels=50, kernel_size=5, stride=1, padding=2) #[(dimension_sz−kernel+2*Padding)/Stride]+1
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(in_channels=50, out_channels=50, kernel_size=5, stride=1, padding=2) #[(dimension_sz−kernel+2*Padding)/Stride]+1
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)  
        '''
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=5, stride=1, padding=2) #[(dimension_sz−kernel+2*Padding)/Stride]+1
        self.relu3 = nn.ReLU()
        #self.pool3 = nn.MaxPool2d(kernel_size=2)  
        ''' 
        self.flat = nn.Flatten()            
        self.fc1 = nn.Linear(50 * 160 * 120, 128)   
        self.relu4 = nn.ReLU()

        self.fc2 = nn.Linear(128, 256)
        self.relu5 = nn.ReLU()

        self.fc3 = nn.Linear(256, 64)
        self.relu6 = nn.ReLU()

        self.fc4 = nn.Linear(64, 10)

    def forward(self, inp):
        inp = self.conv1(inp)
        inp = self.relu1(inp) 
        inp = self.pool1(inp)

        inp = self.conv2(inp)
        inp = self.relu2(inp) 
        inp = self.pool2(inp)
        '''
        inp = self.conv3(inp)
        inp = self.relu3(inp) 
        #inp = self.pool3(inp)       
        ''' 
        inp = self.flat(inp)

        inp = self.fc1 (inp) 
        inp = self.relu4(inp)            
        inp = self.fc2(inp) 
        inp = self.relu5(inp)
        inp = self.fc3(inp) 
        inp = self.relu6(inp)
        out = self.fc4(inp)
        return out
    
    