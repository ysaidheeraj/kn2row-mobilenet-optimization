import torch
import torch.nn as nn
import torch.nn.functional as F
import ctypes
import numpy as np
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

global totalTimeTaken
totalTimeTaken = 0
class CustomConv2dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, out_channels,  bias=None, stride=1, padding=0):
        # Flatten the input and weight matrices
        input_flatten = input.view(input.size(0), -1)
        weight_flatten = weight.view(weight.size(0), -1)

        # Call the custom C++ convolution function
        op_dim = (input.size(2) - weight.size(2) + (2 * padding)) // stride 
        op_dim += 1
        output_array = np.empty((op_dim, op_dim, out_channels), dtype=np.float32)
        cpp_lib = ctypes.CDLL('./kn2row.so')  # Update with the path to your compiled C++ code
        cpp_lib.kn2row.restype = ctypes.c_double
        timetaken_microseconds = cpp_lib.kn2row(input_flatten.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                       input.size(2), input.size(3), input.size(1), out_channels,
                       weight_flatten.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                       weight.size(2), stride, padding, op_dim,
                       output_array.flatten().ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        
        global totalTimeTaken
        totalTimeTaken += timetaken_microseconds

        # Reshape the flattened result back to the original shape
        result = output_array.reshape((1, out_channels, op_dim, op_dim))
        result_tensor = torch.from_numpy(result)

        return result_tensor

class CustomConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(CustomConv2d, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels))
        self.stride = stride
        self.padding = padding
        self.out_channels = out_channels

    def forward(self, x): 
        return CustomConv2dFunction.apply(x, self.weight, self.out_channels,  self.bias, self.stride, self.padding)

class MNetConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(MNetConv2d, self).__init__()
        self.conv2d = CustomConv2d(in_channels, out_channels, 3, stride=stride, padding=1)

    def forward(self, x):
        x = self.conv2d(x)
        x = F.relu(x)
        return x

class MobileNetV1(nn.Module):
    def __init__(self, num_classes=1000):
        super(MobileNetV1, self).__init__()

        self.model = nn.Sequential(
            CustomConv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            MNetConv2d(32, 64, 1),
            MNetConv2d(64, 128, 2),
            MNetConv2d(128, 128, 1),
            MNetConv2d(128, 256, 2),
            MNetConv2d(256, 256, 1),
            MNetConv2d(256, 512, 2),
            MNetConv2d(512, 512, 1),
            MNetConv2d(512, 512, 1),
            MNetConv2d(512, 512, 1),
            MNetConv2d(512, 512, 1),
            MNetConv2d(512, 512, 1),
            MNetConv2d(512, 1024, 1),
            MNetConv2d(1024, 1024, 1),
            nn.AdaptiveAvgPool2d(1)
        )

        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


transform = transforms.Compose([
    transforms.ToTensor(),
])

cifar10_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
cifar10_dataloader = DataLoader(cifar10_dataset, batch_size=1, shuffle=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device: ", device)
model = MobileNetV1(num_classes=10)
for i, data in enumerate(cifar10_dataloader):
    #Checking for 10 images
    if i >= 10:
        break
    input_data, _ = data  # The second element is the label, which is not needed in this case
    output = model(input_data)
print("Output shape:", output.shape)
# Here we get the totalTimeTaken in microseconds from the cpp function
print("Average Time taken for 10 iterations = ", totalTimeTaken / (10.0 ** 7), " seconds")