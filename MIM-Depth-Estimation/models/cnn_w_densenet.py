import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torchinfo import summary
from math import sqrt
# torch.autograd.set_detect_anomaly(True)

class Conv_Block(nn.Module):
    def __init__(self, in_c, out_c, activation_fn=nn.LeakyReLU):
        super().__init__()

        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)

        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)

        self.activfn = activation_fn()

        self.dropout = nn.Dropout(0.25)
    
    def forward(self, inputs):
        
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.activfn(x)
        # x = self.dropout(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activfn(x)
        # x = self.dropout(x)

        return x

class MultiHeadA(nn.Module):
    def __init__(self, out_c):
        super(MultiHeadA, self).__init__()
        self.attention = torch.nn.MultiheadAttention(out_c, max(out_c//512, 1))

    def forward(self, x):
        def apply_attention(x, attention, mask=None):
            orig_shape = x.shape
            x = torch.flatten(x, start_dim=2)
            x = x.permute(2, 0, 1)
            x, _ = attention(x, x, x, attn_mask=mask)
            x = x.permute(1, 2, 0)
            x = torch.reshape(x, orig_shape)
            return x
    
        x = apply_attention(x, self.attention)

        return x

class Encoder_Block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv = Conv_Block(in_c, out_c)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)

        return x, p
    
class Enc_Dec_Model(nn.Module):
    def __init__(self):
        super(Enc_Dec_Model, self).__init__()
        self.encoder1 = Encoder_Block(3, 128)
        self.encoder2 = Encoder_Block(128, 256)
        self.encoder3 = Encoder_Block(256, 512)
        # self.MHA1 = MultiHeadA(512)

        """ Bottleneck """
        self.bottleneck = Conv_Block(512, 1024)

        """ Decoder """
        # self.MHA2 = MultiHeadA(1024)
        self.d1 = Decoder_Block(1024, 512)
        self.d2 = Decoder_Block(512, 256)
        self.d3 = Decoder_Block(256, 128)
        
        """ Classifier """
        self.outputs = nn.Conv2d(128, 1, kernel_size=1, padding=0)
    
    def forward(self, x):

        """ Encoder """
        s1, p1 = self.encoder1(x)
        s2, p2 = self.encoder2(p1)
        s3, p3 = self.encoder3(p2)
        # p3 = self.MHA1(p3)

        """ Bottleneck """
        b = self.bottleneck(p3)

        """ Decoder """
        # b = self.MHA2(b)
        d1 = self.d1(b, s3)
        d2 = self.d2(d1, s2)
        d3 = self.d3(d2, s1)      

        """ Classifier """
        outputs = self.outputs(d3)
        out_depth = torch.sigmoid(outputs)
        return out_depth

class Decoder_Block(nn.Module):
    def __init__(self, in_c, out_c, activation_fn=nn.LeakyReLU):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = Conv_Block(out_c+out_c, out_c, activation_fn)

    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)

        return x
    
class CNN_w_Densenet(nn.Module):
    def __init__(self, max_depth) -> None:
        super().__init__()
        self.densenet = torchvision.models.densenet201(weights=torchvision.models.DenseNet201_Weights.DEFAULT)
        for param in self.densenet.features.parameters():
            param.requires_grad = False
        for param in self.densenet.features.denseblock4.denselayer32.parameters():
            param.requires_grad = True
        for param in self.densenet.features.norm5.parameters():
            param.requires_grad = True

        self.densenet = torch.nn.Sequential(*(list(self.densenet.children())[:-2]))
        self.enc_dec_model = Enc_Dec_Model()
        self.max_depth = max_depth

    def forward(self, x):
        x = self.densenet(x)
        x = self.enc_dec_model(x)
        x = x*self.max_depth
        # print(x.shape)
        return {'pred_d':x}
    
if __name__ == "__main__":
    model = CNN_w_Densenet(max_depth=10).cuda()
    print(model)
    summary(model, input_size=(64,3,448,448))