import argparse
import cv2
import time
import numpy
import torch
from pathlib import Path

import torch.backends.cudnn as cudnn
from numpy import random
from c_utils import *
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, letterbox
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, scale_boxes, Profile
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel


class Preproces():
    r"""图片预处理.

    类中所有函数的输入都是图片序列，类中的所有超参数写死:如不要出现opt.img-size直接写成648.

    Args:
        data_source (Dataset): dataset to sample from
    """
    def __init__(self, weights):
        # 加载模型
        model = attempt_load(weights, map_location="cpu")
        self.model = TracedModel(model, "cpu", 640)

    def detect(self,img):
        # Get names and colors
        names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
        im0 = img
        img = letterbox(img, 640, stride=32)[0]
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to("cpu")
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = self.model(img, augment=False)[0]
        pred = non_max_suppression(pred, 0.4, 0.45, classes=None, agnostic=False)

#增加畸变和裁减代码
        return #返回裁减照片
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='/home/pyk/work/pic-anti-spoofing/CVPR19-Face-Anti-spoofing/yolov7-main/best.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='pic', help='source')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='luzhou', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    parser.add_argument('--jiao', action='store_true', help='识别证件的边角')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--max-det', type=int, default=20, help='maximum detections per image')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        processnet = Preproces(opt.weights)
        weights, view_img, save_txt, trace, save_crop = opt.weights, opt.view_img, opt.save_txt, \
                                                           not opt.no_trace, opt.save_crop
        save_img = not opt.nosave and not opt.source.endswith('.txt')  # save inference images
        webcam = opt.source.isnumeric() or opt.source.endswith('.txt') or opt.source.lower().startswith(
            ('rtsp://', 'rtmp://', 'http://', 'https://'))

        # Directories
        save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Initialize
        set_logging()

        # Set Dataloader
        device = select_device(opt.device)
        vid_path, vid_writer = None, None
        if webcam:
            view_img = check_imshow()
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(opt.source, img_size=640, stride=32)
        else:
            dataset = LoadImages(opt.source, img_size=640, stride=32)
        for path, img, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).to(device)
            img = img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            #图片畸变和裁减
            crop_result = processnet.detect(img)
            #图片遮挡判断

            print('done')
