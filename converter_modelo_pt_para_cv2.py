from models import *

model_cfg = './weights/yolov3-tiny-80-classes/yolov3.cfg'
model_weights = './weights/yolov3-tiny-80-classes/yolov3-tiny.weights'

convert(model_cfg, model_weights)