from models import *

model_cfg = './cfg/yolov3-spp_best.cfg'
model_weights = './weights/best.pt'

convert(model_cfg, model_weights)