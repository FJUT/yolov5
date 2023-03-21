# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run a Flask REST API exposing one or more YOLOv5s models
"""

import argparse
import io

import torch
from flask import Flask, request, jsonify
from PIL import Image
# from models.common import DetectMultiBackend
import datetime

app = Flask(__name__)
models = {}

DETECTION_URL = "/v1/object-detection/<model>"

@app.route('/')
def index():
    return 'heyo'

@app.route(DETECTION_URL, methods=["POST"])
def predict(model):
    if request.method != "POST":
        return

    if request.files.get("image") or request.form.get("path"):
        # Method 1
        # with request.files["image"] as f:
        #     im = Image.open(io.BytesIO(f.read()))

        # Method 2
        if request.files.get("image"):
            imgFrom = 'remote'
            im_file = request.files["image"]
            im_bytes = io.BytesIO(im_file.read())
        else:
            imgFrom = 'local'
            im_bytes = '/www/node/zaniliazhao/' + request.form.get("path") # upload/ScreenShot/名称_1677343150621.jpg
            
        im = Image.open(im_bytes)

        # if model in models:
        #     results = models[model](im, size=640)  # reduce size=320 for faster inference
        #     return results.pandas().xyxy[0].to_json(orient="records")
        # model = torch.hub.load(r'C:\Users\Milan\Projects\yolov5', 'custom', path=r'C:\Users\Milan\Projects\yolov5\models\yolov5s.pt', source='local')

        # modelY = torch.hub.load('/www/python/yolov5/yolov5-master/', 'custom', path='/www/python/yolov5/weights/best.pt', source='local', force_reload=False)
        modelY = torch.hub.load('/Users/guotao071/dev/github/qt/yolov5/', 'custom', path='weights/best.pt', source='local', force_reload=False)

        modelY.conf = 0.25  # NMS confidence threshold
        results = modelY(im, size=640)  # reduce size=320 for faster inference
        # results = torch.hub.load('/Users/guotao071/dev/github/qt/yolov5/', 'custom', path='/Users/guotao071/dev/github/qt/yolov5/weights/yolov5n6.pt', source='local')(im, size=640) 
        # results = DetectMultiBackend('/Users/guotao071/dev/github/qt/yolov5/weights/yolov5n6.pt')(im, size=640)
        if imgFrom == 'remote': # 上传的图片才保存，通过 path 读取的无需重复保存
            nowTime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            # im.save('/www/node/zaniliazhao/upload/ScreenShot/' + nowTime + '.' + im.format)
            # im.save('./' + nowTime + '.' + im.format)
        return jsonify(
            code = 0,
            result = results.pandas().xyxy[0].to_json(orient="records"),
            # filePath = ""
        )
        
    else:
        return 'no image attached'


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask API exposing YOLOv5 model")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    parser.add_argument('--model', nargs='+', default=['yolov5s'], help='model(s) to run, i.e. --model yolov5n yolov5s')
    opt = parser.parse_args()

    # for m in opt.model:
    #     # models[m] = torch.hub.load("ultralytics/yolov5", m, force_reload=True, skip_validation=True)
    #     models[m] = torch.hub.load("/Users/guotao071/dev/github/qt/yolov5/", m, force_reload=True, skip_validation=True, source='local')

    app.run(host="0.0.0.0", port=opt.port)  # debug=True causes Restarting with stat
