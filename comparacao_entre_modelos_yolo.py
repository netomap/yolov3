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

branco = (255, 255, 255)
preto = (0, 0, 0)
vermelho = (255, 0, 0)
verde = (0, 255, 0)
azul = (0, 0, 255)

# descobrindo o device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def detecta_objetos(tipo_modelo, img_path, img_pil, new_w, new_h, cor_box, conf_thres, iou_thres):

    # carregando nomes das classes
    class_file = open('./weights/' + tipo_modelo + '/classes.names', 'r')
    classes = class_file.readlines()
    class_file.close()
    classes = [nome.strip() for nome in classes]
    
    # iniciando o modelo
    imgsz = 416
    model_cfg = './weights/' + tipo_modelo + '/yolov3.cfg'
    model_weights = './weights/' + tipo_modelo + '/yolov3.pt'
    model = Darknet(model_cfg, imgsz)

    model.load_state_dict(torch.load(model_weights, map_location=device)['model'])

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

    # retirando os boxes repetidos
    pred = non_max_suppression(pred, conf_thres, iou_thres, multi_label=False, classes=None)

    # pegando de fato das detecções
    pred = pred[0]  

    if (pred != None): # se o vetor pred tem alguma predição, se não é None, então

        # e transformando em um vetor numpy
        detections = pred.detach().numpy()

        texto = 'Modelo: {}, tempo: {} s, detecções: {}, conf_thres: {}%, iou_thres: {}'.format(tipo_modelo, tempo, len(detections), round(100*conf_thres), iou_thres)
        
        img_w = img_pil.width
        img_h = img_pil.height
        draw = ImageDraw.Draw(img_pil)
        for d in detections:
            if (d[-1] == 0):
                x0, y0, x1, y1 = d[0]*img_w/new_w, d[1]*img_h/new_h, d[2]*img_w/new_w, d[3]*img_h/new_h, 
                draw.rectangle([x0, y0, x1, y1], fill=None, outline=cor_box, width=2)
    
    else:
        texto = 'Modelo: {}, nenhum objeto identificado.'.format(tipo_modelo)
    
    return img_pil, texto


#tipo_modelo = 'yolov3-tiny-1-classe'
#tipo_modelo = 'yolov3-tiny-80-classes'
tipo_modelo = 'yolov3-spp-80-classes'
new_w = 512 #512 para spp ou 674 para tiny
new_h = 316 #316 para spp ou 416 para tiny
img_path = './data/samples/teste8.jpg'
img_pil = Image.open(img_path)
conf_thres = 0.1
iou_thres = 0.1

img1, texto1 = detecta_objetos('yolov3-tiny-80-classes', img_path, img_pil, 674, 416, vermelho, 0.5, 0.1)

img2, texto2 = detecta_objetos('yolov3-tiny-1-classe', img_path, img1, 674, 416, verde, 0.1, 0.1)

img3, texto3 = detecta_objetos('yolov3-spp-1-classe', img_path, img1, 512, 316, azul, 0.1, 0.1)

draw = ImageDraw.Draw(img3)
draw.rectangle([0, 0, img3.width, 75], fill=branco)

plt.imshow(img3)
plt.text(0, 25, texto1, {'color': 'red'})
plt.text(0, 50, texto2, {'color': 'green'})
plt.text(0, 75, texto3, {'color': 'blue'})
plt.xticks([])
plt.yticks([])
plt.show()