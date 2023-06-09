import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

model_type = "DPT_Large"

midas = torch.hub.load("intel-isl/MiDaS", model_type)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform

input_dir = 'dataset/dop/speaker/output/'  # directory with your input images
output_dir = 'dataset/validation/depth/speaker/'  # directory where you want to save depth maps
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for filename in os.listdir(input_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        img_path = os.path.join(input_dir, filename)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        input_batch = transform(img).to(device)

        with torch.no_grad():
            prediction = midas(input_batch)

            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        output = prediction.cpu().numpy()
        output = np.sqrt((output - np.min(output)) / np.ptp(output))  # apply square root to decrease contrast

        # Apply color map
        output_colored = cv2.applyColorMap((output * 255).astype(np.uint8), cv2.COLORMAP_JET)

        # Save depth map
        output_colored_bgr = cv2.cvtColor(output_colored, cv2.COLOR_RGB2BGR)
        output_filename = os.path.join(output_dir, filename.split('.')[0] + '_depth.jpg')
        cv2.imwrite(output_filename, output_colored_bgr)

        # Show depth map
        # plt.imshow(output_colored)
        # plt.show()

print('Depth maps created and saved.')
