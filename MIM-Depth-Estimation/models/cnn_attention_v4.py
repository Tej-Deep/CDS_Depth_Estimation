import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torchinfo import summary
    
class NewDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.convT1 = nn.ConvTranspose2d(2048, 256, 3, 2, padding=1, output_padding=1)
        self.convT2 = nn.ConvTranspose2d(256, 128, 3, 2, padding=1, output_padding=1)
        self.convT3 = nn.ConvTranspose2d(128, 64, 3, 2, padding=1, output_padding=1)
        self.convT4 = nn.ConvTranspose2d(64, 32, 3, 2, padding=1, output_padding=1)
        self.convT5 = nn.ConvTranspose2d(32, 1, 3, 2, padding=1, output_padding=1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.dropout3 = nn.Dropout(0.25)
        self.dropout4 = nn.Dropout(0.25)

        self.attention2 = torch.nn.MultiheadAttention(128, 8)
    
    def forward(self, x, mask=None):

        def apply_attention(x, mask, attention):
            orig_shape = x.shape
            x = torch.flatten(x, start_dim=2)
            x = x.permute(2, 0, 1)
            x, _ = attention(x, x, x, attn_mask=mask)
            x = x.permute(1, 2, 0)
            x = torch.reshape(x, orig_shape)
            return x
        
        x = self.convT1(x)
        x = F.relu(x)

        x = self.convT2(x)
        x = apply_attention(x, mask, self.attention2)
        x = F.relu(x)
        x = self.dropout1(x)
        x = F.relu(x)

        x = self.convT3(x) 
        x = self.dropout2(x)
        x = F.relu(x)

        x = self.convT4(x)
        x = F.relu(x)

        x = self.convT5(x)
        x = F.sigmoid(x)

        output = x
        return output
    
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 2)
        self.conv2 = nn.Conv2d(32, 64, 5, 2)
        self.conv3 = nn.Conv2d(64, 128, 5, 1, 3)
        self.conv4 = nn.Conv2d(128, 2048, 5, 2, 2)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)

        self.attention1 = torch.nn.MultiheadAttention(32, 2)
        self.attention2 = torch.nn.MultiheadAttention(64, 1)
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
        x = apply_attention(x, mask, self.attention1)
        x = F.relu(x)
        x = self.conv2(x)
        x = apply_attention(x, mask, self.attention2)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.conv3(x)
        x = apply_attention(x, mask, self.attention3)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv4(x)
        x = apply_attention(x, mask, self.attention4)
        x = F.relu(x)
        output = x
        return output
    
class enc_dec_model(nn.Module):
    def __init__(self, max_depth) -> None:
        super().__init__()
        self.encoder = Net()
        self.decoder = NewDecoder()

        self.max_depth = max_depth
    def forward(self, x):
        x = self.encoder(x)
        # x = self.bridge(x)
        x = self.decoder(x)
        print(x.shape)
        x = x*self.max_depth
        return {'pred_d':x}
    
if __name__ == "__main__":
    model = enc_dec_model(max_depth=10).cuda()
    print(model)
    summary(model, input_size=(64,3,448,448))