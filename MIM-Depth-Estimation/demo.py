import gradio as gr
import torch
import os
from models.pretrained_decv2 import enc_dec_model
from models.densenet_v2 import Densenet
from models.unet_resnet18 import ResNet18UNet
from models.unet_resnet50 import UNetWithResnet50Encoder
from configs.test_options import TestOptions
import numpy as np
import cv2

# kb cropping
def cropping(img):
    h_im, w_im = img.shape[:2]

    margin_top = int(h_im - 352)
    margin_left = int((w_im - 1216) / 2)

    img = img[margin_top: margin_top + 352,
                margin_left: margin_left + 1216]

    return img

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(DEVICE)
CWD = os.getcwd()
CKPT_FILE_NAMES = {
    'Indoor':{
        'Resnet_enc':'resnet_nyu_best.ckpt',
        'Unet':'resnet18_unet_epoch_08_model_kitti_and_nyu.ckpt',
        'Densenet_enc':'densenet_epoch_15_model.ckpt'
    },
    'Outdoor':{
        'Resnet_enc':'resnet_encdecmodel_epoch_05_model_nyu_and_kitti.ckpt',
        'Unet':'resnet50_unet_epoch_02_model_nyuandkitti.ckpt',
        'Densenet_enc':'densenet_nyu_then_kitti_epoch_10_model.ckpt'
    }
}
MODEL_CLASSES = {
    'Indoor': {
        'Resnet_enc':enc_dec_model,
        'Unet':ResNet18UNet,
        'Densenet_enc':Densenet
    },

    'Outdoor': {
        'Resnet_enc':enc_dec_model,
        'Unet':UNetWithResnet50Encoder,
        'Densenet_enc':Densenet
    },

}

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


def predict(location, model_name, img):
    ckpt_dir = f"{CWD}/ckpt/{CKPT_FILE_NAMES[location][model_name]}"
    if location == 'Indoor':
        max_depth = 10
    else:
        max_depth = 80
    model = MODEL_CLASSES[location][model_name](max_depth).to(DEVICE)
    load_model(ckpt_dir,model)
    # print(img.shape)
    # assert False 
    if img.shape ==  (375,1242,3):
        img = cropping(img)
    img = torch.tensor(img).permute(2, 0, 1).float().to(DEVICE)
    input_RGB = img.unsqueeze(0)
    print(input_RGB.shape)
    with torch.no_grad():
        pred = model(input_RGB)
        pred_d = pred['pred_d']
        pred_d_numpy = pred_d.squeeze().cpu().numpy()
        # pred_d_numpy = (pred_d_numpy - pred_d_numpy.mean())/pred_d_numpy.std()
        pred_d_numpy = np.clip((pred_d_numpy / pred_d_numpy[15:,:].max()) * 255, 0,255)
        # pred_d_numpy = (pred_d_numpy / pred_d_numpy.max()) * 255
        pred_d_numpy = pred_d_numpy.astype(np.uint8)
        pred_d_color = cv2.applyColorMap(pred_d_numpy, cv2.COLORMAP_RAINBOW)
        pred_d_color = cv2.cvtColor(pred_d_color, cv2.COLOR_BGR2RGB)
        # del model
    return pred_d_color

with gr.Blocks() as demo:
    gr.Markdown("# Monocular Depth Estimation")
    with gr.Row():
        location = gr.Radio(choices=['Indoor', 'Outdoor'],value='Indoor', label = "Select Location Type")
        model_name = gr.Radio(['Unet', 'Resnet_enc', 'Densenet_enc'],value="Densenet_enc" ,label="Select model")
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label = "Input Image for Depth Estimation")
        with gr.Column():
            output_depth_map = gr.Image(label = "Depth prediction Heatmap")
    with gr.Row():
        predict_btn = gr.Button("Generate Depthmap")
        predict_btn.click(fn=predict, inputs=[location, model_name, input_image], outputs=output_depth_map)
    with gr.Row():
        gr.Examples(['./demo_data/Bathroom.jpg', './demo_data/Bedroom.jpg', './demo_data/Bookstore.jpg', './demo_data/Classroom.jpg', './demo_data/Computerlab.jpg', './demo_data/kitti_1.png'], inputs=input_image)    
demo.launch()