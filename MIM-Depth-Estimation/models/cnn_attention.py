import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
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
        return self.model(x)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 2)
        self.conv2 = nn.Conv2d(32, 64, 5, 2)
        self.conv3 = nn.Conv2d(64, 128, 5, 2, 1)
        self.conv4 = nn.Conv2d(128, 2048, 5, 2, 2)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)

        # self.attention1 = torch.nn.MultiheadAttention(32, 2)
        # self.attention2 = torch.nn.MultiheadAttention(64, 1)
        self.attention3 = torch.nn.MultiheadAttention(128, 8)
        self.attention4 = torch.nn.MultiheadAttention(2048, 8)
    
    def forward(self, x, mask=None):

        def apply_attention(x, mask, attention):
            orig_shape = x.shape
            x = torch.flatten(x, start_dim=2)
            x = x.permute(2, 0, 1)
            x, _ = attention(x, x, x, attn_mask=mask)
            x = x.permute(1, 2, 0)
            x = torch.reshape(x, orig_shape)
            return x
        
        x = self.conv1(x)
        # x = apply_attention(x, mask, self.attention1)
        x = F.relu(x)
        x = self.conv2(x)
        # x = apply_attention(x, mask, self.attention2)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.conv3(x)
        x = apply_attention(x, mask, self.attention3)
        x = F.relu(x)
        x = self.conv4(x)
        # x = apply_attention(x, mask, self.attention4)
        x = F.relu(x)
        output = x
        return output
    
class enc_dec_model(nn.Module):
    def __init__(self, max_depth) -> None:
        super().__init__()
        # self.encoder = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
        # for param in self.encoder.parameters():
        #     param.requires_grad = False
        # self.encoder = torch.nn.Sequential(*(list(self.encoder.children())[:-2]))
        self.encoder = Net()
        # self.bridge = nn.Conv2d(2048, 2048, 1, 1)
        self.decoder = Decoder(num_layers=5,\
                                channels=[2048,256,128,64,32,1],\
                                kernels=[3,3,3,3,3],\
                                strides = [2,2,2,2,2],\
                                 activations=['relu', 'relu', 'relu' ,'relu', 'sigmoid'])
        self.max_depth = max_depth
    def forward(self, x):
        x = self.encoder(x)
        # x = self.bridge(x)
        x = self.decoder(x)
        x = x*self.max_depth
        return {'pred_d':x}
    
if __name__ == "__main__":
    # model = Decoder(num_layers=5,\
    #                 channels=[2048,256,128,64,32,1],\
    #                 kernels=[3,3,3,3,3],\
    #                 strides = [2,2,2,2,2])
    model = enc_dec_model(max_depth=10)
    # .cuda()
    print(model)
    summary(model, input_size=(64,3,448,448))