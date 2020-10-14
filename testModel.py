import os
import time

import numpy as np
import torch
import cv2
from torchvision import transforms

# 测试基础配置
data_path = "./dataset/test/"
img_path = os.path.join(data_path, 'images/1.jpg')
checkpoint_path = 'checkpoints/2020-10-14_14_19_25/model_epoch_50'

def callNetworkModel(flag_image,device):
    transform = transforms.Compose([transforms.ToTensor(), ])
    trained_model = torch.load(checkpoint_path)
    trained_model = trained_model.to(device)
    # img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(flag_image, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (1920, 1080))
    mat_img = transform(img)
    device_img = torch.reshape(mat_img, [1, 3, 1080, 1920])
    device_img = device_img.to(device)
    with torch.no_grad():
        output = torch.sigmoid((trained_model(device_img)))

    output_np = output.cpu().data.numpy().copy()
    output_np = np.argmax(output_np, axis=1)
    pred = np.zeros((output.shape[0], output.shape[2], output.shape[3], 3))
    pred[np.where(output_np == 1)] = np.array([0.0, 1.0, 0.0])
    end = time.clock()

    device_img = device_img.permute(0, 2, 3, 1).cpu().data.numpy().copy()
    device_img[np.where(output_np == 1)] = np.array([0.8, 0.1, 0.0])
    device_img = np.reshape(device_img, [1080, 1920, 3])
    return device_img
