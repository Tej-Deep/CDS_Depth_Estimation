import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torchinfo import summary

class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)

        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)

        self.relu = nn.LeakyReLU(0.2)

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = F.leaky_relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.leaky_relu(x)

        return x
        
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        # num_layers=5,\
        #                         channels=[2048,256,128,64,32,1],\
        #                         kernels=[3,3,3,3,3],\
        #                         strides = [2,2,2,2,2],\
        #                          activations=['relu', 'relu', 'relu' ,'relu', 'sigmoid'])
        # self.convT0 = nn.ConvTranspose2d(2048, 512, 3, 2, padding=1, output_padding=1)
        self.convT1 = nn.ConvTranspose2d(2048, 256, 3, 2, padding=1, output_padding=1)
        self.convT2 = nn.ConvTranspose2d(256, 128, 3, 2, padding=1, output_padding=1)
        self.convT3 = nn.ConvTranspose2d(128, 64, 3, 2, padding=1, output_padding=1)
        self.convT4 = nn.ConvTranspose2d(64, 32, 3, 2, padding=1, output_padding=1)
        self.convT5 = nn.ConvTranspose2d(32, 1, 3, 2, padding=1, output_padding=1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.dropout3 = nn.Dropout(0.25)
        self.dropout4 = nn.Dropout(0.25)

        # self.attention1 = torch.nn.MultiheadAttention(32, 2)
        self.attention2 = torch.nn.MultiheadAttention(128, 8)
        # self.attention3 = torch.nn.MultiheadAttention(64, 4)
        # self.attention4 = torch.nn.MultiheadAttention(32, 2)


        
        self.convblock1 = conv_block(256, 256)
        self.convblock2 = conv_block(128, 128)
        self.convblock3 = conv_block(64, 64)
        self.convblock4 = conv_block(32, 32)

    def forward(self, x, mask=None):

        def apply_attention(x, mask, attention):
            orig_shape = x.shape
            x = torch.flatten(x, start_dim=2)
            x = x.permute(2, 0, 1)
            x, _ = attention(x, x, x, attn_mask=mask)
            x = x.permute(1, 2, 0)
            x = torch.reshape(x, orig_shape)
            return x
        
        # x = self.convT0(x)
        # x = F.leaky_relu(x)
        # # x = torch.nn.functional.interpolate(x, scale_factor=2)
        # #print(x.shape)

        x = self.convT1(x)
        x = F.leaky_relu(x)
        x = self.convblock1(x)
        x = self.dropout1(x)
        # x = torch.nn.functional.interpolate(x, scale_factor=2)
        #print(x.shape)


        x = self.convT2(x)
        x = apply_attention(x, mask, self.attention2)
        x = F.leaky_relu(x)
        x = self.convblock2(x)
        x = self.dropout2(x)
        # x = torch.nn.functional.interpolate(x, scale_factor=2)
        #print(x.shape)


        x = self.convT3(x)
        # x = apply_attention(x, mask, self.attention3)   
        x = F.leaky_relu(x)
        x = self.convblock3(x)
        x = self.dropout3(x)
        # x = torch.nn.functional.interpolate(x, scale_factor=2)
        #print(x.shape)


        x = self.convT4(x)
        # x = apply_attention(x, mask, self.attention4)
        x = F.leaky_relu(x)
        x = self.convblock4(x)
        x = self.dropout4(x)
        # x = torch.nn.functional.interpolate(x, scale_factor=2)
        #print(x.shape)


        x = self.convT5(x)
        # x = apply_attention(x, mask, self.attention5)
        x = F.sigmoid(x)
        # x = torch.nn.functional.interpolate(x, scale_factor=2)
        #print(x.shape)

        output = x
        # #print(x.shape)
        return output

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 2)
        self.conv2 = nn.Conv2d(32, 128, 3, 2, 1)
        self.conv3 = nn.Conv2d(128, 512, 5, 2, 1)
        self.conv4 = nn.Conv2d(512, 2048, 5, 2, 2)
        self.conv5 = nn.Conv2d(1024, 2048, 3, 2, padding=1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.dropout3 = nn.Dropout(0.25)
        self.dropout4 = nn.Dropout(0.25)
        self.dropout5 = nn.Dropout(0.25)

        # self.attention1 = torch.nn.MultiheadAttention(128, 2)
        self.attention2 = torch.nn.MultiheadAttention(512, 8)
        self.attention3 = torch.nn.MultiheadAttention(2048, 8)
        
        self.batch2d1 = nn.BatchNorm2d(32)
        self.batch2d2 = nn.BatchNorm2d(128)
        self.batch2d3 = nn.BatchNorm2d(512)
        self.batch2d4 = nn.BatchNorm2d(2048)
        self.batch2d5 = nn.BatchNorm2d(2048)
    
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
        x = F.leaky_relu(x)
        x = self.batch2d1(x)
        x = self.dropout1(x)
        #print(x.shape)

        x = self.conv2(x)
        # x = apply_attention(x, mask, self.attention1)
        x = F.leaky_relu(x)
        x = F.max_pool2d(x, 2)
        x = self.batch2d2(x)
        x = self.dropout2(x)
        #print(x.shape)

        x = self.conv3(x)
        x = apply_attention(x, mask, self.attention2)
        x = F.leaky_relu(x)
        x = self.batch2d3(x)
        x = self.dropout3(x)
        #print(x.shape)

        x = self.conv4(x)
        x = apply_attention(x, mask, self.attention3)
        x = F.leaky_relu(x)
        x = self.batch2d4(x)
        x = self.dropout4(x)
        #print(x.shape)

        # x = self.conv5(x)
        # x = F.leaky_relu(x)
        # x = self.batch2d5(x)
        # x = self.dropout5(x)
        # #print(x.shape)

        # print(x.shape)
        output = x
        return output
    
class enc_dec_model(nn.Module):
    def __init__(self, max_depth) -> None:
        super().__init__()
        self.densenet = torchvision.models.densenet201(weights=torchvision.models.DenseNet201_Weights.DEFAULT)
        for param in self.densenet.features.parameters():
            param.requires_grad = False
        for param in self.densenet.features.denseblock4.denselayer32.parameters():
            param.requires_grad = True
        for param in self.densenet.features.norm5.parameters():
            param.requires_grad = True
        print((list(self.densenet.children())[:-1]))
        self.densenet = torch.nn.Sequential(*(list(self.densenet.children())[:-2]))
        self.encoder = Net()
        self.decoder = Decoder()
        self.max_depth = max_depth
        self.batch2d = nn.BatchNorm2d(3)

    def forward(self, x):
        x = self.densenet(x)
        x = self.batch2d(x)
        x = self.encoder(x)
        x = self.decoder(x)
        #print(x.shape)
        x = x*self.max_depth
        return {'pred_d':x}
    
if __name__ == "__main__":
    model = enc_dec_model(max_depth=10).cuda()
    print(model)
    summary(model, input_size=(64,3,448,448))