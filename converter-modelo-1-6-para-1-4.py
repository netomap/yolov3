import torch
import torchvision
from models import *

print ('versão torch: ', torch.__version__)

tipo_modelo = 'yolov3-tiny-1-classe'

# iniciando o modelo
imgsz = 416
model_cfg = './weights/' + tipo_modelo + '/yolov3.cfg'
model_weights = './weights/' + tipo_modelo + '/yolov3.pt'
model = Darknet(model_cfg, imgsz)

# salvando sem usar zipfile
#torch.save(model, 'yolov3-tiny-1-classe-v-1-4', _use_new_zipfile_serialization=False)

# apenas um teste de importação do modelo v 1.4
model2 = torch.load('./yolov3-tiny-1-classe-v-1-4', map_location='cpu')
print (model2)