import os
import cv2
import numpy as np
locations  = ['bedroom','bookstore', 'classroom', 'computer_lab']
ids = ['00055', '00083', '00283', '00332']
for loc, i in zip(locations,ids):
    gt_path = f"./MIM-Depth-Estimation/data/nyu_depth_v2/official_splits/test/{loc}/sync_depth_{i}.png"
    depth = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED).astype('float32')

    save_path = f"./MIM-Depth-Estimation/data/nyu_depth_v2/official_splits/label/{loc}_label_{i}.png"
    # pred_d_numpy = pred_d.squeeze().cpu().numpy()
    depth = (depth / depth.max()) * 255
    depth = depth.astype(np.uint8)
    pred_d_color = cv2.applyColorMap(depth, cv2.COLORMAP_RAINBOW)
    cv2.imwrite(save_path, pred_d_color)