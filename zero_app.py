# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from PIL import Image, ImageDraw, ImageFont
import numpy as np
import time
import io
import cv2
import json
import traceback
import os
from datetime import datetime
import shutil
from distutils.util import strtobool
#import requests
from zero_mask import detect,detect_demo

# Imports for the REST API
from flask import Flask, request, jsonify, Response,send_file

#Setting up logger
import logging
handler = logging.StreamHandler()
root = logging.getLogger()
root.setLevel(os.environ.get("LOGLEVEL", "INFO"))
root.addHandler(handler)



session = None
tags = []
output_dir = 'images'

# Called when the deployed service starts
def init():
    if (os.path.exists(output_dir)):
        print(output_dir + " already exits")
    else:
        os.mkdir(output_dir)


def gen():
   """Video streaming generator function."""
   vc = cv2.VideoCapture(0)
   while True:
       rval, frame = vc.read()
       cv2.imwrite('pic.jpg', frame)
       yield (b'--frame\r\n' 
              b'Content-Type: image/jpeg\r\n\r\n' + open('pic.jpg', 'rb').read() + b'\r\n')


app = Flask(__name__)

# / routes to the default function which returns 'Hello World'
@app.route('/', methods=['GET'])
def defaultPage():
    return Response(response='Hello from Yolov3 inferencing based on ONNX', status=200)


@app.route('/post_video_stream')
def post_stream():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/get_video_stream')
def get_stream():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')





#
@app.route('/annotate_image', methods=['POST'])
def annotate_image():
    try:
        options = dict()
        options["weights"] = "weights/yolov5l_fm_opt.pt"
        options["source"] = r"dummy.jpg"
        options["output"] = r"/mask_scanner/inference/output"
        #options["output"] = r"inference/output"
        options["img_size"] = 480
        options["conf_thres"] = 0.4
        options["iou_thres"] = 0.5
        options["fourcc"] = "mp4v"
        options["device"] = ""
        options["view_img"] = False
        options["save_txt"] = None
        options["classes"] = None
        options["agnostic_nms"] = None
        options["augment"] = None
        options["include_small_image"] = True
        options["info_screen_small"] = True

        for entry in request.args:
            convert_to_bool = []
            try:
                if entry in convert_to_bool:
                    options[entry] = strtobool(request.args[entry])
            except:
                print(f"The following term could not be converted {entry}:{request.args[entry]}")

        imageData = io.BytesIO(request.get_data())
        img = Image.open(imageData)
        img.save("dummy.jpg")
        detect(options)

        img = Image.open(f"{options['output']}/dummy.jpg")
        imgByteArr = io.BytesIO()
        img.save(imgByteArr, format = 'JPEG')
        imgByteArr = imgByteArr.getvalue()

        return Response(response = imgByteArr, status = 200, mimetype = "image/jpeg")
    except Exception as e:
        logging.exception(f"Exception in annotate(): {str(e)}")
        print('EXCEPTION:', str(e))
        return Response(response='Error processing image', status= 500)

#
@app.route('/annotate_image_demo', methods=['POST'])
def annotate_image_demo():
    try:
        options = dict()
        options["weights"] = "weights/yolov5l_fm_opt.pt"
        options["source"] = r"dummy.jpg"
        options["output"] = r"/mask_scanner/inference/output"
        #options["output"] = r"inference/output"
        options["img_size"] = 480
        options["conf_thres"] = 0.4
        options["iou_thres"] = 0.5
        options["fourcc"] = "mp4v"
        options["device"] = ""
        options["view_img"] = False
        options["save_txt"] = None
        options["classes"] = None
        options["agnostic_nms"] = None
        options["augment"] = None
        options["include_small_image"] = True
        options["info_screen_small"] = True

        for entry in request.args:
            convert_to_bool = ["include_small_image","info_screen_small"]
            try:
                if entry in convert_to_bool:
                    options[entry] = strtobool(request.args[entry])
            except:
                print(f"The following term could not be converted {entry}:{request.args[entry]}")

        imageData = io.BytesIO(request.get_data())
        img = Image.open(imageData)
        img.save("dummy.jpg")
        detect_demo(options)

        img = Image.open(f"{options['output']}/dummy.jpg")
        imgByteArr = io.BytesIO()
        img.save(imgByteArr, format = 'JPEG')
        imgByteArr = imgByteArr.getvalue()                

        return Response(response = imgByteArr, status = 200, mimetype = "image/jpeg")
    except Exception as e:
        logging.exception(f"Exception in annotate(): {str(e)}")
        print('EXCEPTION:', str(e))
        return Response(response='Error processing image', status= 500)



@app.route("/annotate_video", methods=['POST'])
def annotate_video():
    pass

@app.route("/annotate_video_demo", methods=['POST'])
def annotate_video_demo():
    try:
        options = dict()
        options["weights"] = "weights/yolov5l_fm_opt.pt"
        options["source"] = r"dummy.mp4"
        options["output"] = r"/mask_scanner/inference/output"
        options["output"] = r"inference/output"
        options["img_size"] = 480
        options["conf_thres"] = 0.4
        options["iou_thres"] = 0.5
        options["fourcc"] = "mp4v"
        options["device"] = ""
        options["view_img"] = False
        options["save_txt"] = None
        options["classes"] = None
        options["agnostic_nms"] = None
        options["augment"] = None

        for entry in request.args:
            convert_to_bool = ["include_small_image","info_screen_small"]
            try:
                if entry in convert_to_bool:
                    options[entry] = strtobool(request.args[entry])
            except:
                print(f"The following term could not be converted {entry}:{request.args[entry]}")

        imageData = io.BytesIO(request.get_data())
        imageData.seek(0)
        print(type(imageData))
        with open('dummy.mp4', 'wb') as f:
            shutil.copyfileobj(imageData, f)

        detect_demo(options)

        file_to_send = f"{options['output']}/dummy.mp4"
        print("File to send:", {file_to_send})
        return send_file(file_to_send)

    except Exception as e:
        traceback.print_exc()
        logging.exception(f"Exception in annotate(): {str(e)}")
        return Response(response='Error processing image ' + str(e), status= 500)


# Load and initialize the model
init()

if __name__ == '__main__':
    # Run the server
    app.run(host='0.0.0.0', port=80,debug=True)
