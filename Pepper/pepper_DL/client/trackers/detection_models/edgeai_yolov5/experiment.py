#import torch
#import os
#import numpy as np
from models.experimental import attempt_load
#from utils.torch_utils import select_device, load_classifier, time_synchronized
#from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
#    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
#from utils.datasets import LoadImages
#from pathlib import Path
#from yolo import YoloManager

#manager = YoloManager()


"""
# params
device = "cpu"
device = select_device(device)
save_txt_tidl = True
imgsz = [640,640]
augment = True
conf_thres = 0.25
iou_thres = 0.45
classes = 0
agnostic_nms = True
kpt_label = True
weights = 'yoloposes_640_lite.pt'
webcam = False


half = device.type != 'cpu' and not save_txt_tidl  # half precision only supported on CUDA
weights_dir = os.path.join("weights", weights)

# Load model
model = attempt_load(weights_dir, map_location=device)  # load model
stride = int(model.stride.max())  # model stride

names = model.module.names if hasattr(model, 'module') else model.names  # get class names

imgsz[0] = check_img_size(imgsz[0], s=stride)
imgsz[1] = check_img_size(imgsz[1], s=stride)

dataset = LoadImages(os.path.join("data", "images"), img_size=imgsz, stride=stride)

for path, img, im0s, vid_cap in dataset:
    print("path type: ", type(path))
    print("path: ", path)
    print("img type: ", type(img))
    print("img: ", img)
    print("img shape: ", img.shape)
    print("im0s type: ", type(im0s))
    print("im0s: ", im0s)
    print("im0s shape: ", im0s.shape)
    print("vid_cap type: ", type(vid_cap))
    print("vid_cap: ", vid_cap)

    #img = img
    #im0s = im0s

    #break
    img = torch.from_numpy(img).to(device)

    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    t1 = time_synchronized()
    img = img
    break
    pred = model(img, augment=augment)[0]
    print(pred[...,4].max())
    # Apply NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms, kpt_label=kpt_label)
    t2 = time_synchronized()
"""
"""
    # Process detections
    for i, det in enumerate(pred):  # detections per image

        if webcam:  # batch_size >= 1
            p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
        else:
            p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

        p = Path(p)  # to Path
        #save_path = str(save_dir / p.name)  # img.jpg
        #txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
        s += '%gx%g ' % img.shape[2:]  # print string

        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if len(det):
            # Rescale boxes from img_size to im0 size
            scale_coords(img.shape[2:], det[:, :4], im0.shape, kpt_label=False)
            scale_coords(img.shape[2:], det[:, 6:], im0.shape, kpt_label=kpt_label, step=3)

            # Print results
            for c in det[:, 5].unique():
                n = (det[:, 5] == c).sum()  # detections per class
                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

            # Write results
            for det_index, (*xyxy, conf, cls) in enumerate(reversed(det[:,:6])):
                if save_txt:  # Write to file
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                    with open(txt_path + '.txt', 'a') as f:
                        f.write(('%g ' * len(line)).rstrip() % line + '\n')

                if save_img or opt.save_crop or view_img:  # Add bbox to image
                    c = int(cls)  # integer class
                    label = None if opt.hide_labels else (names[c] if opt.hide_conf else f'{names[c]} {conf:.2f}')
                    kpts = det[det_index, 6:]
                    plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=opt.line_thickness, kpt_label=kpt_label, kpts=kpts, steps=3, orig_shape=im0.shape[:2])
                    if opt.save_crop:
                        save_one_box(xyxy, im0s, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)


            if save_txt_tidl:  # Write to file in tidl dump format
                for *xyxy, conf, cls in det_tidl:
                    xyxy = torch.tensor(xyxy).view(-1).tolist()
                    line = (conf, cls,  *xyxy) if opt.save_conf else (cls, *xyxy)  # label format
                    with open(txt_path + '.txt', 'a') as f:
                        f.write(('%g ' * len(line)).rstrip() % line + '\n')

        # Print time (inference + NMS)
        print(f'{s}Done. ({t2 - t1:.3f}s)')

"""