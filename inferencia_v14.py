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
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

tipo_modelo = 'yolov3-tiny-1-classe'
# carregando nomes das classes
class_file = open('./weights/' + tipo_modelo + '/classes.names', 'r')
classes = class_file.readlines()
class_file.close()
classes = [nome.strip() for nome in classes]

new_w = 674 #512 para spp ou 674 para tiny
new_h = 416 #316 para spp ou 416 para tiny
img_path = './data/samples/teste2.jpg'


# iniciando o modelo
imgsz = 416
model = torch.load('yolov3-tiny-1-classe-v-1-4', map_location=device)

model.to(device).eval()

img = cv2.imread(img_path)

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

# realizando a inferência
inicio = time.time()
pred = model(img)[0]
tempo = round(time.time()-inicio, 2)

conf_thres = 0.1
iou_thres = 0.85
# retirando os boxes repetidos
pred = non_max_suppression(pred, conf_thres, iou_thres, multi_label=False, classes=None)

# pegando de fato das detecções
pred = pred[0]
print (pred)

img_pil = Image.open(img_path)

if (pred != None): # se o vetor pred tem alguma predição, se não é None, então

    # e transformando em um vetor numpy
    detections = pred.detach().numpy()
    
    img_w = img_pil.width
    img_h = img_pil.height
    draw = ImageDraw.Draw(img_pil)
    for d in detections:
        x0, y0, x1, y1 = d[0]*img_w/new_w, d[1]*img_h/new_h, d[2]*img_w/new_w, d[3]*img_h/new_h, 
        draw.rectangle([x0, y0, x1, y1], fill=None, outline=0, width=2)

plt.imshow(img_pil)
plt.xticks([])
plt.yticks([])

#plt.savefig('./resultado.jpg')
plt.show()