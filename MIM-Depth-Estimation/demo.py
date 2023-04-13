import gradio as gr
import torch
from models.pretrained_decv2 import enc_dec_model
from configs.test_options import TestOptions
import numpy as np
import cv2

opt = TestOptions()
args = opt.initialize().parse_args()
print(args)

def load_model(ckpt, model, optimizer=None):
    ckpt_dict = torch.load(ckpt, map_location='cpu')
    # keep backward compatibility
    if 'model' not in ckpt_dict and 'optimizer' not in ckpt_dict:
        state_dict = ckpt_dict
    else:
        state_dict = ckpt_dict['model']
    weights = {}
    for key, value in state_dict.items():
        if key.startswith('module.'):
            weights[key[len('module.'):]] = value
        else:
            weights[key] = value

    model.load_state_dict(weights)

    if optimizer is not None:
        optimizer_state = ckpt_dict['optimizer']
        optimizer.load_state_dict(optimizer_state)


def predict(img):
    ckpt_dir = "C:/Users/ptejd/Documents/Computational_Data_Science/CDS_Depth_Estimation/MIM-Depth-Estimation/logs/0409_Resnet_pretrained_v3dec_small_lr/epoch_08_model.ckpt"
    model = enc_dec_model(args.max_depth_eval)
    load_model(ckpt_dir,model)
    img = torch.tensor(img).permute(2, 0, 1).float()
    input_RGB = img.unsqueeze(0)
    # print(input_RGB.shape)
    with torch.no_grad():
        # if args.shift_window_test:
        #     bs, _, h, w = input_RGB.shape
        #     assert w > h and bs == 1
        #     interval_all = w - h
        #     interval = interval_all // (args.shift_size-1)
        #     sliding_images = []
        #     sliding_masks = torch.zeros((bs, 1, h, w), device=input_RGB.device) 
        #     for i in range(args.shift_size):
        #         sliding_images.append(input_RGB[..., :, i*interval:i*interval+h])
        #         sliding_masks[..., :, i*interval:i*interval+h] += 1
        #     input_RGB = torch.cat(sliding_images, dim=0)
        # if args.flip_test:
        #     input_RGB = torch.cat((input_RGB, torch.flip(input_RGB, [3])), dim=0)
        # print(input_RGB)
        pred = model(input_RGB)
        pred_d = pred['pred_d']
        # if args.flip_test:
        #     batch_s = pred_d.shape[0]//2
        #     pred_d = (pred_d[:batch_s] + torch.flip(pred_d[batch_s:], [3]))/2.0
        # if args.shift_window_test:
        #     pred_s = torch.zeros((bs, 1, h, w), device=pred_d.device)
        #     for i in range(args.shift_size):
        #         pred_s[..., :, i*interval:i*interval+h] += pred_d[i:i+1]
        #     pred_d = pred_s/sliding_masks
        # # print("c")
        pred_d_numpy = pred_d.squeeze().cpu().numpy()
        # pred_d_numpy = (pred_d_numpy - pred_d_numpy.mean())/pred_d_numpy.std()
        pred_d_numpy = (pred_d_numpy / pred_d_numpy.max()) * 255
        pred_d_numpy = pred_d_numpy.astype(np.uint8)
        pred_d_color = cv2.applyColorMap(pred_d_numpy, cv2.COLORMAP_RAINBOW)
        print(pred_d_color.shape)
    # print(type(img))
    # print(img.shape)
    return pred_d_color

demo = gr.Interface(fn = predict, inputs=gr.Image(), outputs = "image")
demo.launch()

"""
python demo.py --max_depth_eval 10.0 --flip_test --shift_window_test --shift_size 2
"""