
import sys
#sys.path.append(r"C:\Users\U734813\Documents\GitHub\mask_scanner")
import argparse
import json     
import torch.backends.cudnn as cudnn
import time
from utils import google_utils
from utils.datasets import *
from utils.utils import *


def detect_json(options,save_img=False):
    """
    Returns a json file which includes the results of the inference:
    Form of the json
    Video: {0: [{},{}]}:

    Image: 
    """
    opt = options
    out, source, weights, view_img, save_txt, imgsz = \
        opt["output"], opt["source"], opt["weights"], opt["view_img"], opt["save_txt"], opt["img_size"]
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    # Initialize
    device = torch_utils.select_device(opt["device"])
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    google_utils.attempt_download(weights)
    model = torch.load(weights, map_location=device)['model'].float()  # load to FP32
    # torch.save(torch.load(weights, map_location=device), weights)  # update model if SourceChangeWarning
    # model.fuse()
    model.to(device).eval()
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = torch_utils.load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.names if hasattr(model, 'names') else model.modules.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
    colors[0] = (255,0,0)
    colors[1] = (0,255,0)

    # Run inference

    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once

    result_dic = dict()
    img_counter = 0
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = torch_utils.time_synchronized()
        pred = model(img, augment=opt["augment"])[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt["conf_thres"], opt["iou_thres"], classes=opt["classes"], agnostic=opt["agnostic_nms"])
        t2 = torch_utils.time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  #  normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                #No we create the result dict
                result_list_one_frame = []
                for *xyxy, conf, cls in det:
                    single_inference_dic = {}
                    if int(cls)==0:
                        label = "mask"
                    elif int(cls)==1:
                        label = "no mask"
                    else:
                        sys.exit("Label not known")
                    single_inference_dic["xywh"] =  (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    single_inference_dic["xywh"] = [float(i) for i in single_inference_dic["xywh"] ]
                    single_inference_dic["xyxy"] = ((float(xyxy[0]),float(xyxy[1])),(float(xyxy[2]),float(xyxy[3])))
                    single_inference_dic["map"] = float(conf)
                    single_inference_dic["label"] = str(label)
                    result_list_one_frame.append(single_inference_dic)

            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))
        img_counter += 1
        result_dic[img_counter] = result_list_one_frame 

    print(result_dic)
    print('Done. (%.3fs)' % (time.time() - t0))
    return result_dic

def detect(options,save_img=False):
    opt = options
    out, source, weights, view_img, save_txt, imgsz = \
        opt["output"], opt["source"], opt["weights"], opt["view_img"], opt["save_txt"], opt["img_size"]
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    # Initialize
    device = torch_utils.select_device(opt["device"])
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    google_utils.attempt_download(weights)
    model = torch.load(weights, map_location=device)['model'].float()  # load to FP32
    # torch.save(torch.load(weights, map_location=device), weights)  # update model if SourceChangeWarning
    # model.fuse()
    model.to(device).eval()
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = torch_utils.load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.names if hasattr(model, 'names') else model.modules.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
    colors[0] = (255,0,0)
    colors[1] = (0,255,0)

    # Run inference

    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = torch_utils.time_synchronized()
        pred = model(img, augment=opt["augment"])[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt["conf_thres"], opt["iou_thres"], classes=opt["classes"], agnostic=opt["agnostic_nms"])
        t2 = torch_utils.time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  #  normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                for *xyxy, conf, cls in det:
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        with open(save_path[:save_path.rfind('.')] + '.txt', 'a') as file:
                            file.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format

                    if save_img or view_img:  # Add bbox to image
                        if int(cls)==0:
                            label = "mask"
                        elif int(cls)==1:
                            label = "no mask"
                        else:
                            sys.exit("Label not known")

                        label = '%s %.2f' % (label, conf)
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Stream results
            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*opt["fourcc"]), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))
    return 1

def detect_demo(options,save_img=False):
    opt = options
    out, source, weights, view_img, save_txt, imgsz = \
        opt["output"], opt["source"], opt["weights"], opt["view_img"], opt["save_txt"], opt["img_size"]
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    # Initialize
    device = torch_utils.select_device(opt["device"])
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    google_utils.attempt_download(weights)
    model = torch.load(weights, map_location=device)['model'].float()  # load to FP32
    # torch.save(torch.load(weights, map_location=device), weights)  # update model if SourceChangeWarning
    # model.fuse()
    model.to(device).eval()
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = torch_utils.load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.names if hasattr(model, 'names') else model.modules.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once


    keep_overlay = False
    overlay_to_use = None
    overlay_to_use_old = None
    iteration_subcounter = 0
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = torch_utils.time_synchronized()
        pred = model(img, augment=opt["augment"])[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt["conf_thres"], opt["iou_thres"], classes=opt["classes"], agnostic=opt["agnostic_nms"])
        t2 = torch_utils.time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)



        #Initial load of the diffent overlays
        path_to_ok = "demo_files/ok_sign_final.png"
        overlay_ok = cv2.imread(path_to_ok)
        overlay_ok = cv2.resize(overlay_ok,(1920,1080))

        path_to_stop = "demo_files/stop_sign_final.png"
        overlay = cv2.imread(path_to_stop)
        trans_mask = overlay[:,:,2] == 0
        overlay[trans_mask] = [255, 255, 255]
        overlay_not_ok = cv2.cvtColor(overlay, cv2.COLOR_BGRA2BGR)
        overlay_not_ok = cv2.resize(overlay_not_ok,(1920,1080))

        path_to_attention = "demo_files/attention_sign_final.png"
        overlay_attention = cv2.imread(path_to_attention)
        overlay_attention = cv2.resize(overlay_attention,(1920,1080))


        # Process detections
        for i, det in enumerate(pred):  # detections per image
            print("tetes")
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  #  normalization gain whwh


            no_mask_true = False
            mask_true = False

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                for *xyxy, conf, cls in det:
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        with open(save_path[:save_path.rfind('.')] + '.txt', 'a') as file:
                            file.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format

                    if int(cls)==0:
                        mask_true = True
                    elif int(cls)==1:
                        no_mask_true =True
                    else:
                        sys.exit("Label not known")


                    if save_img or view_img:  # Add bbox to image
                        label = None
                        if int(cls)==0:
                            label = "mask"
                            color = (0,255,0)
                        elif int(cls)==1:
                            label = "No mask"
                            color = (0,0,255)
                        else:
                            sys.exit("Label not known")

                        plot_one_box(xyxy, im0, label=label, color=color, line_thickness=3)


            print(keep_overlay,iteration_subcounter,overlay_to_use,overlay_to_use_old)
            frame_to_wait = 100
            if (keep_overlay==True) and (iteration_subcounter<frame_to_wait):
                overlay_to_use = overlay_to_use_old
                iteration_subcounter += 1
            elif (keep_overlay==True) and (iteration_subcounter>=frame_to_wait):
                iteration_subcounter = 0
                keep_overlay = False


            if (mask_true==True) and (no_mask_true==False) and (keep_overlay==False):
                overlay_to_use = "ok"
            elif (no_mask_true==True) and (keep_overlay==False):
                overlay_to_use = "not_ok"
            elif (mask_true==False) and (no_mask_true==False) and (keep_overlay==False):
                overlay_to_use = "attention"
            else:
                pass


            #
            if overlay_to_use != overlay_to_use_old:
                keep_overlay = True
                iteration_subcounter = 0
                overlay_to_use_old = overlay_to_use
            else:
                overlay_to_use_old = overlay_to_use

            if overlay_to_use=="ok":
                overlay = overlay_ok
            elif overlay_to_use=="not_ok":
                overlay = overlay_not_ok
            elif overlay_to_use=="attention":
                overlay = overlay_attention
            else:
                print(f"Overlay not known {overlay_to_use}")

            #Including the current images
            im0 = cv2.resize(im0,(1920,1080))

            #include_small_image = True
            #info_screen_small = False
            if options["info_screen_small"]==False:
                s_img = im0.copy()
                s_img = cv2.resize(s_img,(1080,1920))
                print(s_img.shape,overlay.shape)
                s_img = cv2.resize(s_img,(480,360))
                print(s_img.shape,overlay.shape)
                x_offset=10
                y_offset=10
                if options["include_small_image"]==True:
                    overlay[y_offset:y_offset+s_img.shape[0], x_offset:x_offset+s_img.shape[1]] = s_img
                im0 = overlay
            else:
                s_img = overlay.copy()
                s_img = cv2.resize(s_img,(1080,1920))
                print(s_img.shape,overlay.shape)
                s_img = cv2.resize(s_img,(480,360))
                print(s_img.shape,im0   .shape)
                x_offset=10
                y_offset=10
                if options["include_small_image"]==True:
                    im0[y_offset:y_offset+s_img.shape[0], x_offset:x_offset+s_img.shape[1]] = s_img
                im0 = im0



            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Stream results
            if view_img:
                cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
                cv2.setWindowProperty("window",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
                #im0 = cv2.resize(im0,(1920,1080))
                cv2.imshow("window", im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            save_img = True
            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*opt["fourcc"]), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        print('Results saved to %s' % save_path)
        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    options = {}
    options["weights"] = "weights/yolov5l_fm_opt.pt"
    options["source"] = "0" #r"C:\Users\U734813\Documents\GitHub\mask_scanner\inference\images\with_mask_group_of_people.jpg"
    options["output"] = "."
    options["img_size"] = 480
    options["conf_thres"] = 0.4
    options["iou_thres"] = 0.5
    options["fourcc"] = "mp4v"
    options["device"] =""
    options["view_img"] = False
    options["save_txt"] = None
    options["classes"] = None
    options["agnostic_nms"] = None
    options["augment"] = None

    #with torch.no_grad():
    detect(options)
