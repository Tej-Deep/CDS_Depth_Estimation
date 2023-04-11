import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torchinfo import summary
from math import sqrt
# torch.autograd.set_detect_anomaly(True)

class attention_gate(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
 
        self.Wg = nn.Sequential(
            nn.Conv2d(in_c[0], out_c, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_c)
        )
        self.Ws = nn.Sequential(
            nn.Conv2d(in_c[1], out_c, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_c)
        )
        self.relu = nn.ReLU(inplace=True)
        self.output = nn.Sequential(
            nn.Conv2d(out_c, out_c, kernel_size=1, padding=0),
            nn.Sigmoid()
        )
 
    def forward(self, g, s):
        Wg = self.Wg(g)
        Ws = self.Ws(s)
        out = self.relu(Wg + Ws)
        out = self.output(out)
        return out 
    
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
        self.encoder1 = Encoder_Block(3, 64)
        self.encoder2 = Encoder_Block(64, 128)
        self.encoder3 = Encoder_Block(128, 256)
        """ Bottleneck """
        self.bottleneck = Conv_Block(256, 512)

        """ Decoder """
        self.d1 = Decoder_Block([512, 256], 256)
        self.d2 = Decoder_Block([256, 128], 128)
        self.d3 = Decoder_Block([128, 64], 64)
        
        """ Classifier """
        self.outputs = nn.Conv2d(64, 1, kernel_size=1, padding=0)
    
    def forward(self, x):

        """ Encoder """
        s1, p1 = self.encoder1(x)
        s2, p2 = self.encoder2(p1)
        s3, p3 = self.encoder3(p2)

        """ Bottleneck """
        b = self.bottleneck(p3)

        """ Decoder """
        d1 = self.d1(b, s3)
        d2 = self.d2(d1, s2)
        d3 = self.d3(d2, s1)      

        """ Classifier """
        outputs = self.outputs(d3)
        out_depth = torch.sigmoid(outputs)
        return out_depth

class Decoder_Block(nn.Module):

    def __init__(self, in_c, out_c):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.ag = attention_gate(in_c, out_c)
        self.c1 = Conv_Block(in_c[0]+out_c, out_c)

    def forward(self, x, s):
        x = self.up(x)
        s = self.ag(x, s)
        x = torch.cat([x, s], axis=1)
        x = self.c1(x)
        return x
    
class Densenet(nn.Module):
    def __init__(self, max_depth) -> None:
        super().__init__()
        self.densenet = torchvision.models.densenet201(weights=torchvision.models.DenseNet201_Weights.DEFAULT)
        for param in self.densenet.features.parameters():
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
    model = Densenet(max_depth=10).cuda()
    print(model)
    summary(model, input_size=(64,3,448,448))