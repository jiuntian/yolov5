import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import numpy as np

from models.experimental import attempt_load
from utils.datasets import LoadImages, LoadImagesEvalAI
from utils.general import check_img_size, check_requirements, non_max_suppression, \
    scale_coords, increment_path, apply_classifier
from utils.torch_utils import select_device, time_synchronized, load_classifier


def detect():
    source, weights, imgsz = opt.source, opt.weights, opt.img_size
    # Directories
    save_dir = Path(opt.export_dir)  # export dir
    save_dir.mkdir(parents=True, exist_ok=True)  # make dir
    # Initialize
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA
    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16
    # Second-stage classifier
    classify = opt.second_stage
    if classify:
        print("Enabled second stage classifier")
        modelc = load_classifier(name='mobilenet_v2', n=3)  # initialize
        modelc.load_state_dict(torch.load(opt.second_stage, map_location=device))
        modelc = modelc.to(device).eval()
        if half:
            modelc.half()
    # cudnn.benchmark = True  # set True to speed up consant image size inference
    if opt.evalai:
        dataset = LoadImagesEvalAI(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    result = []
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        # Inference
        # t1 = time_synchronized()
        pred = model(img)[0]
        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, agnostic=opt.agnostic_nms)
        # t2 = time_synchronized()

        # Apply Classifier
        if classify and len(pred):
            pred_second_stage = apply_classifier(pred, modelc, img, im0s, half)
            aesthetics = pred_second_stage
            # aesthetics = np.zeros((len(pred_second_stage), len(pred_second_stage[0]), 3))
            # for i in range(len(pred_second_stage)):  # loop over images in batch
            #     aesthetics.append(np.zeros((len(pred_second_stage[i]), 3)))  # zero array of [dets, 3]
            #     for j in range(len(pred_second_stage[i])):  # loop over dets
            #         a_label = pred_second_stage[i][j]  # det label
            #         if a_label < 3:
            #             aesthetics[i][j][a_label] = 1

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape)#.round()
                # Write results
                for j, (*xyxy, conf, cls) in enumerate(det):
                    result.append({
                        "image_id": int(p.name[:-4]),
                        "category_id": int(cls.item()) + 1,
                        "aesthetic": aesthetics[i][j].cpu().numpy().astype(int).tolist() if classify else [0, 0, 0],
                        "bbox": [xyxy[0].item(), xyxy[1].item(),
                                 xyxy[2].item(), xyxy[1].item(),
                                 xyxy[2].item(), xyxy[3].item(),
                                 xyxy[0].item(), xyxy[3].item()],
                        "score": conf.item()
                    })
            # Print time (inference + NMS)
            # print(f'{s}Done. ({t2 - t1:.3f}s)')
    print(f'Done. ({time.time() - t0:.3f}s)')
    with open(save_dir / 'result.json', 'w') as outfile:
        json.dump(result, outfile)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--export-dir', default='inference', help='save results to dir')
    parser.add_argument('--exist-ok', action='store_true', help='existing dir ok, do not increment')
    parser.add_argument('--evalai', action='store_true', help='eval ai use images.json')
    parser.add_argument('--second-stage', type=str, default='',
                        help='second stage model ckpt')  # file/folder, 0 for webcam
    # python inference.py --source ../ictext/valtest --weights ../../Downloads/best.pt --exist-ok
    opt = parser.parse_args()
    print(opt)
    check_requirements()

    with torch.no_grad():
        detect()
