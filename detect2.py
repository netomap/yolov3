import torch
import torchvision
import numpy as np
import cv2
import time
import pathlib
import os
import sys
from models import *
from PIL import Image, ImageFont, ImageDraw
import matplotlib.pyplot as plt
# descobrindo o device
#device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
device = torch.device('cpu')

# iniciando o modelo
imgsz = 416
model_cfg = '/content/yolov3/cfg/yolov3-tiny.cfg'
model_weights = '/content/yolov3/weights/best.pt'
model = Darknet(model_cfg, imgsz)

model.load_state_dict(torch.load(model_weights, map_location=device)['model'])

# configurando modelo para device e fazer somente inferências
model.to(device).eval()

# carregando nomes das classes
class_file = open('/content/yolov3/data/classes.names', 'r')
classes = class_file.readlines()
class_file.close()
classes = [nome.strip() for nome in classes]
print (classes)

img_path = '/content/yolov3/data/samples/teste4.jpg'
img = cv2.imread(img_path)
print ('tamanho original da imagem: {}'.format(img.shape))

new_w = 674 #512 para spp ou 674 para tiny
new_h = 416 #316 para spp ou 416 para tiny

img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

# Convert
img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
img = np.ascontiguousarray(img)

# transformando em tensor
img = torch.from_numpy(img).to(device)
img = img.float()
img = img/255.0

# adicionando uma dimenção
img.unsqueeze_(0)
print ('tensor shape: ', img.shape)

# realizando a inferência
inicio = time.time()
pred = model(img)[0]
tempo = round(time.time()-inicio, 2)
print ('tempo de inferencia %02f segundos com yolov3-spp.cfg - 2 classes.' % tempo)

# retirando os boxes repetidos
pred = non_max_suppression(pred, 0.3, 0.6, multi_label=False, classes=None)

# pegando de fato das detecções
pred = pred[0]



def imprime_deteccoes(img_path, new_w, new_h, detections, classes, texto):
    img_pil = Image.open(img_path)
    img_w = img_pil.width
    img_h = img_pil.height
    draw = ImageDraw.Draw(img_pil)
    for d in detections:
        cor = (0,0,0) if d[-1] == 0 else (255, 255, 255)
        x0, y0, x1, y1 = d[0]*img_w/new_w, d[1]*img_h/new_h, d[2]*img_w/new_w, d[3]*img_h/new_h, 
        draw.rectangle([x0, y0, x1, y1], fill=None, outline=cor, width=2)
    
    return img_pil

if (pred): # se o vetor pred tem alguma predição, se não é None, então

  # e transformando em um vetor numpy
  detections = pred.detach().numpy()
  print ('detections: ', detections)

  texto = 'Detecção usando {} em {} segundos com {} detecções.'.format(model_cfg, tempo, len(detections))
  nova_imagem = imprime_deteccoes(img_path, new_w, new_h, detections, classes, texto)
  plt.imshow(nova_imagem)
  plt.xticks([])
  plt.yticks([])
  plt.xlabel(texto)
  plt.show()
else:
  print ('não houve detecções de pedestres na foto.')