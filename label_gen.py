import os
import cv2
import numpy as np

gt_path = "./MIM-Depth-Estimation/data/nyu_depth_v2/official_splits/test/bathroom/sync_depth_00045.png"
depth = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED).astype('float32')

save_path = "./MIM-Depth-Estimation/data/nyu_depth_v2/official_splits/label/bathroom_label_00045.png"
# pred_d_numpy = pred_d.squeeze().cpu().numpy()
depth = (depth / depth.max()) * 255
depth = depth.astype(np.uint8)
pred_d_color = cv2.applyColorMap(depth, cv2.COLORMAP_RAINBOW)
cv2.imwrite(save_path, pred_d_color)