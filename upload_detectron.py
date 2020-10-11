# coding:utf-8
 
from flask import Flask, render_template, request, redirect, url_for, make_response,jsonify
from werkzeug.utils import secure_filename
import time
from PIL import Image 
from datetime import timedelta
import os, json, cv2, random, io
import numpy as np
import torch
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.data import DatasetCatalog
MetadataCatalog.get("dla_train").thing_classes = ['caption', 'figure', 'page', 'table', 'title', 'text']



# 输入是一个图片的地址，输出为一张图片，可以直接把输出通过imwrite保存。
def inference(input_path,model,model_weight):

    im = cv2.imread(input_path)
    #im = input
    #这里的im需要是一张图片，因此如果是图片路径就需要先通过imread变成图片，如果是url就需要通过load方法。

    cfg = get_cfg()
    cfg.merge_from_file(model)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 # set threshold for this mode
    cfg.MODEL.WEIGHTS = model_weight
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 6

    if torch.cuda.is_available():
        print('we use cuda!')
        cfg.MODEL.DEVICE='cuda'
    else:
        print('running on cpu')
        cfg.MODEL.DEVICE='cpu'

    predictor = DefaultPredictor(cfg)
    outputs = predictor(im)
    outputs["instances"].pred_classes
    outputs["instances"].pred_boxes

    v = Visualizer(im[:,:,::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    outs = out.get_image()[:, :, ::-1]

    if os.path.exists(input_path):
        os.remove(input_path)
        print('remove image')
        
    if os.path.splitext(input_path)[-1] == ".jpg":
        cv2.imwrite(input_path,outs)
        print('input is a jpg file:',outs)
        return input_path
    else:
        image_name = os.path.splitext(os.path.split(input_path)[1])[0]
        print('image name:',image_name)
        jpg_name = os.path.join(os.path.split(input_path)[0],image_name+'.jpg')
        cv2.imwrite(jpg_name,outs)
        print('convert input to jpg:',jpg_name)
        return jpg_name


#input = "demo/input1.jpg"
model = "configs/DLA_mask_rcnn_X_101_32x8d_FPN_3x.yaml"
model_weight = 'model_weight/pub_model_final.pth'


#设置允许的文件格式
ALLOWED_EXTENSIONS = set(['PNG', 'JPG', 'JPGE', 'PBM','PDF'])

 
def allowed_file(filename):
    ext = filename.rsplit('.', 1)[1]
    return '.' in filename and ext.upper() in ALLOWED_EXTENSIONS
 
app = Flask(__name__)

# 设置静态文件缓存过期时间
#app.send_file_max_age_default = timedelta(seconds=1)
 
def resize_image(files):
    im = Image.open(files)
    (w,h) = im.size
    n_w = 500
    n_h = int(h/w*n_w)
    return n_w,n_h

 
@app.route('/upload', methods=['POST', 'GET'])  # 添加路由
def upload():
    if request.method == 'POST':
        file = request.files['file']
        
        # 如果非法的拓展名，或者为空，或者没有.那么返回error。
        if not (file and allowed_file(file.filename)):
            #return jsonify({"error": "please check the input form, only accept image file and PDF."})
            return render_template('upload_start2.html',warning = "Illegal input, please choose again.")
        # 根据当前文件所在路径，创建一个储存image的文件夹
        basepath = os.path.dirname(__file__) 
        file_path = os.path.join(basepath, 'static/result')
        print('file path:',file_path)
        if not os.path.exists(file_path):
            os.makedirs(file_path, 755)
        
        # 保存图片
        file_name = secure_filename(file.filename)
        upload_path = os.path.join(file_path, file_name)
        file.save(upload_path)
        print('file path:',file_path,'file name:',file_name,'upload path:',upload_path)

        # 推断结果，并保存
        infer_path = inference(upload_path,model,model_weight)
        infer_name = os.path.split(infer_path)[1]
        
        # 重新调整图片大小，使得适合屏幕        
        n_w,n_h = resize_image(infer_path)
        print('new size is:',n_w,n_h,'file name is:',file_name)

        return render_template('upload_done2.html', input_name = infer_name, new_weight = n_w, new_height = n_h)
 
    return render_template('upload_start2.html')
 
 
if __name__ == '__main__':
    # app.debug = True
    app.run(host='0.0.0.0', port=5000)