import argparse
import cv2
import time
import numpy
import torch
from pathlib import Path
# from ultralytics.utils.plotting import Annotator, save_one_box

import torch.backends.cudnn as cudnn
from numpy import random
from c_utils import *
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, letterbox
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, scale_boxes, Profile
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel


def detect(source):
    weights, view_img, save_txt, trace, save_crop = opt.weights, opt.view_img, opt.save_txt, \
                                                           not opt.no_trace, opt.save_crop
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    crop_results = []
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or
                                     old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            shibai_path = save_dir / 'shibai' / f'{p.stem}.jpg'
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                jiaosum = (det[:, -1] == 0).sum()
                occlusionsum = (det[:, -1] == 5).sum()
                backsum = (det[:, -1] == 1).sum()
                frontsum = (det[:, -1] == 4).sum()
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                if backsum + frontsum >= 1:
                    maxarea = 0
                    maxresult = None
                    crop_image = None
                    if jiaosum == 4 and opt.jiao:
                        imtran = imc.copy()
                        jiao_pred = [det[i] for i in range(len(det)) if int(det[i][5]) == 0]
                        jiao_pred = torch.stack(jiao_pred, dim=0)
                        # todo
                        jiao_pts = tensor_to_pts(jiao_pred)
                        jiao_area = pixel_area(jiao_pts)
                        if jiao_area / (gn[0].item() * gn[1].item()) < 0.8 and\
                                is_document_distorted(np.array(jiao_pts)):
                            # 进行透视矫正
                            crop_image = four_point_transform(imtran, jiao_pts)
                        else:
                            for *xyxy, _, cls in reversed(det):
                                if save_crop and (cls == 1 or cls == 4):
                                    crop_image = save_one_box(xyxy, imc, file=save_dir / 'crops' / f'{p.stem}.jpg',
                                                              BGR=True, gain=1.04, save=False)
                                    if crop_image.shape[0]*crop_image.shape[1] > maxarea:
                                        maxarea = crop_image.shape[0]*crop_image.shape[1]
                                        maxresult = crop_image
                            if maxarea != 0:
                                crop_image = maxresult
                    else:
                        for *xyxy, _, cls in reversed(det):
                            if save_crop and (cls == 1 or cls == 4):
                                # if save_crop and (cls == 2):
                                crop_image = save_one_box(xyxy, imc, file=save_dir / 'crops' / f'{p.stem}.jpg',
                                                          BGR=True, gain=1.04, save=False)
                                if crop_image.shape[0] * crop_image.shape[1] > maxarea:
                                    maxarea = crop_image.shape[0] * crop_image.shape[1]
                                    maxresult = crop_image
                        if maxarea != 0:
                            crop_image = maxresult
                    xy = diandao_detect(crop_image)
                    crop_image = diandao(crop_image, det, xy)
                    crop_results.append([crop_image, str(p)[-11:]])
                    crop_file = save_dir / 'crops' / f'{p.stem}.jpg'
                    # print(crop_file)
                    if not os.path.exists(save_dir / 'crops'):
                        # 如果文件夹不存在，创建它
                        os.makedirs(save_dir / 'crops')
                    cv2.imwrite(str(crop_file), crop_image)
                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')
                        if save_img or view_img:  # Add bbox to image
                            label = f'{names[int(cls)]} {conf:.2f}'
                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)
                        # if save_crop and (cls == 1 or cls == 4):
                        if save_crop and (cls == 2 and occlusionsum == 0):
                            save_one_box(xyxy, imc, file=save_dir / 'faces' / f'{p.stem}.jpg', BGR=True, gain=1.04)
                else:
                    shibai_path.parent.mkdir(parents=True, exist_ok=True)  # make directory
                    cv2.imwrite(str(shibai_path), im0)
                # crop_image = diandao_detect(crop_image)
                # xy = diandao_detect(crop_image)
                # crop_image = diandao(crop_image, det, xy)
                # crop_results.append([crop_image, str(p)[-11:]])
                # crop_file = save_dir / 'crops' / f'{p.stem}.jpg'
                # # print(crop_file)
                # if not os.path.exists(save_dir / 'crops'):
                #     # 如果文件夹不存在，创建它
                #     os.makedirs(save_dir / 'crops')
                # cv2.imwrite(str(crop_file), crop_image)

            else:
                shibai_path.parent.mkdir(parents=True, exist_ok=True)  # make directory
                cv2.imwrite(shibai_path, im0)
            # Print time (inference + NMS)
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    print(f" The image with the result is saved in: {save_path}")
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    # if save_txt or save_img:
    #     f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''

    print(f'Done. ({time.time() - t0:.3f}s)')
    return crop_results


def occlusion_detect(images):

    # Initialize
    set_logging()

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

    occlusion_conf = []
    for img, p in images:
        # Padded resize
        img = letterbox(img, 640, stride=32)[0]
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=opt.augment)[0]
        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)

        # Process detections
        conf_list = []
        for i, det in enumerate(pred):  # detections per image
            im0 = img
            p = Path(p)  # to Path
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                # Print results
                occlusionsum = (det[:, -1] == 5).sum()
                if occlusionsum != 0:
                    for *xyxy, conf, cls in reversed(det):
                        if cls == 5:
                            conf_list.append(conf)
        occlusion_conf.append([conf_list, p])
    return occlusion_conf


def diandao_detect(img):

    # Initialize
    set_logging()

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    im0 = img
    # Padded resize
    img = letterbox(img, 640, stride=32)[0]
    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
        pred = model(img, augment=opt.augment)[0]
    # Apply NMS
    pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)

    # Process detections
    for i, det in enumerate(pred):  # detections per image

        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
            for *xyxy, conf, cls in reversed(det):
                if cls == 2 or cls == 3:
                    return xyxy


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
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect(source=opt.source)
                strip_optimizer(opt.weights)
        else:
            device = select_device(opt.device)
            half = device.type != 'cpu'  # half precision only supported on CUDA
            # Load model
            model = attempt_load(opt.weights, map_location=device)  # load FP32 model
            stride = int(model.stride.max())  # model stride
            imgsz = check_img_size(opt.img_size, s=stride)  # check img_size
            if not opt.no_trace:
                model = TracedModel(model, device, opt.img_size)
            if half:
                model.half()  # to FP16
            cropimage = detect(source=opt.source)
            occlusion_result = occlusion_detect(cropimage)
            print('done')
