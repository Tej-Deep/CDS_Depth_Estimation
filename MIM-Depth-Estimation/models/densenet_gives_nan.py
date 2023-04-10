import torch
import torch.nn as nn
import torchvision
from torchinfo import summary

class Decoder_block(nn.Module):
    def __init__(self, in_channel, out_channel, kernel, stride, padding=1, out_padding=1, act = 'relu') -> None:
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels=in_channel,\
                                            out_channels=out_channel,\
                                            kernel_size=kernel,\
                                             stride=stride,\
                                              padding=padding,
                                              output_padding=out_padding)
        if act == 'relu':
            self.activation = nn.ReLU()
        elif act == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.Identity()
        
    def forward(self, x):
        print(x.shape)
        return self.activation(self.upsample(x))
    
class Decoder(nn.Module):
    def __init__(self, num_layers, channels, kernels, strides, activations) -> None:
        super().__init__()
        assert len(channels) -1 == len(kernels) and len(strides) == len(kernels) and num_layers == len(strides)
        assert num_layers == len(activations)
        self.layers = []
        for i in range(num_layers):
            self.layers.append(Decoder_block(in_channel=channels[i],\
                                             out_channel=channels[i+1],\
                                              kernel=kernels[i],\
                                                stride=strides[i],\
                                                 act=activations[i]))
        self.model = nn.Sequential(*self.layers)
    def forward(self, x):
        print(x.shape)
        return self.model(x)

class enc_dec_model(nn.Module):
    def __init__(self, max_depth) -> None:
        super().__init__()
        # quant_model = torchvision.models.get_model("quantized_mobilenet_v3_large", weights="DEFAULT")
        # self.encoder = quant_model
        self.encoder = torchvision.models.densenet201(weights=torchvision.models.DenseNet201_Weights.DEFAULT)
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder = torch.nn.Sequential(*(list(self.encoder.children())[:-2]))
        self.decoder = Decoder(num_layers=5,\
                                channels=[3,3,3,3,3,3],\
                                kernels=[3,3,3,3,2],\
                                strides = [2,2,2,2,2],\
                                 activations=['relu', 'relu', 'relu' ,'relu', 'sigmoid'])
        self.conv1 = nn.Conv2d(3, 1, 1, 1)
        self.max_depth = max_depth
    def forward(self, x):
        x = self.encoder(x)
        x = self.conv1(x)
        x = x.squeeze(dim=1)
        x = x*self.max_depth
        return {'pred_d':x}
    
if __name__ == "__main__":
    model = enc_dec_model().cuda()
    print(model)
    summary(model, input_size=(64,3,448,448))