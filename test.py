import time
import cv2
import torch
import numpy as np
from torchvision import models
from torchvision import transforms as T
from PIL import Image


def decode_segmap(image, nc=21):
    """
    """
    label_colors = np.array(
        [(0, 0, 0),
        (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
        (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
        (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
        (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    for l in range(0, nc):
        idx = image == l
        r[idx] = label_colors[l, 0]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 2]
    rgb = np.stack([r, g, b], axis=2)
    return rgb


def segment(net, inp):
    # transform= T.Compose([T.Resize(256),
    #                 T.CenterCrop(224),
    #                 T.ToTensor(),
    #                 T.Normalize(mean = [0.485, 0.456, 0.406],
    #                             std = [0.229, 0.224, 0.225])])
    # inp = transform(img).unsqueeze(0)
    out = net(inp)['out']
    om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
    rgb = decode_segmap(om)
    return rgb


if __name__ == '__main__':
    net = models.segmentation.lraspp_mobilenet_v3_large(pretrained=True)
    net.eval()

    cap = cv2.VideoCapture(0)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        t0 = time.time()
        img = cv2.resize(frame, (224, 224)).astype('float32')
        img /= 255.
        img = (img - mean) / std
        img = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0)

        rgb = segment(net, img)
        t1 = time.time()
        print('FPS:', 1 / (t1 - t0))
        cv2.imshow('img', rgb)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break