import numpy as np
import cv2

def center_crop(img, size):
    """
    Get the center crop of the input image
    Args:
        img: input image [BxCxHxW]
        size: size of the center crop (tuple)
    Output:
        img_pad: center crop
        x, y: coordinates of the crop
    """

    if not isinstance(size, tuple):
        size = (size, size)

    img = img.copy()
    w, h = img.shape[1::-1]

    pad_w = 0
    pad_h = 0
    if w < size[0]:
        pad_w = np.uint16((size[0] - w) / 2)
    if h < size[1]:
        pad_h = np.uint16((size[1] - h) / 2)
    img_pad = cv2.copyMakeBorder(img, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    w, h = img_pad.shape[1::-1]

    x1 = w // 2 - size[0] // 2
    y1 = h // 2 - size[1] // 2

    img_pad = img_pad[y1:y1 + size[1], x1:x1 + size[0], :]

    return img_pad, x1, y1


def image_processing(img1_path, img2_path, image_transforms, resize_shape=(512, 512), device='cuda'):
    '''
    It prepares images for pass through to model.
    Images will be cropped by minimal shape, resized by the shape as you want
    Inputs:
    img1_path: path to image A
    img2_path: path to image B
    image_transforms: image transformation before model
    device: cpu or cuda
    '''
    img1_ = cv2.cvtColor(cv2.imread(img1_path), cv2.COLOR_BGR2RGB)
    img2_ = cv2.cvtColor(cv2.imread(img2_path), cv2.COLOR_BGR2RGB)

    sr_H, sr_W, _ = img1_.shape
    tg_H, tg_W, _ = img2_.shape
    crop_sr = min(sr_H, sr_W)
    crop_tg = min(tg_H, tg_W)
    crop = min(crop_tg, crop_sr)
    img1_crop, _, _ = center_crop(img1_, crop)
    img2_crop, _, _ = center_crop(img2_, crop)
    img1_crop = cv2.resize(img1_crop, resize_shape)
    img2_crop = cv2.resize(img2_crop, resize_shape)

    img1 = image_transforms(img1_crop).unsqueeze(0).to(device)
    img2 = image_transforms(img2_crop).unsqueeze(0).to(device)
    return img1, img2




def get_scale(model_scale, img1_path, img2_path, image_transforms, device, resize_shape=(512, 512)):
    """
    It computes the scale factor between images A and B
    Inputs:
        - model_scale: ScaleNet model
        - img1_path: path to image A
        - img2_path: path to image B
        - image_transforms: image transformations before ScaleNet model
        - device: cpu/gpu device
    Outputs:
        - scale: Scale factor
    """
    img1_ = cv2.cvtColor(cv2.imread(img1_path), cv2.COLOR_BGR2RGB)
    img2_ = cv2.cvtColor(cv2.imread(img2_path), cv2.COLOR_BGR2RGB)

    sr_H, sr_W, _ = img1_.shape
    tg_H, tg_W, _ = img2_.shape
    crop_sr = min(sr_H, sr_W)
    crop_tg = min(tg_H, tg_W)
    crop = min(crop_tg, crop_sr)
    img1_crop, _, _ = center_crop(img1_, crop)
    img2_crop, _, _ = center_crop(img2_, crop)
    img1_crop = cv2.resize(img1_crop, resize_shape)
    img2_crop = cv2.resize(img2_crop, resize_shape)

    img1 = image_transforms(img1_crop).unsqueeze(0).to(device)
    img2 = image_transforms(img2_crop).unsqueeze(0).to(device)

    scale = model_scale(img1, img2)
    return scale.detach().to('cpu').numpy()[0]
