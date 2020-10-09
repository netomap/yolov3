import cv2
import numpy as np
import pathlib
from time import time

# preparação do vetor com as classes para o modelo
classes_file = open('./data/dataset.names', 'r')
linhas = classes_file.readlines()
classes = [linha.strip() for linha in linhas]

# instanciando o modelo
pesos = './weights/best.weights'
configuracao = './cfg/yolov3-spp_best.cfg'

model = cv2.dnn.readNet(pesos, configuracao)

# teste incial com apenas uma imagem
img = cv2.imread('./data/samples/img7.png')
width = img.shape[1]  # no cv2 width aparece na segunda posição
height = img.shape[0] # height aparece na primeira posição
#scale = 0.00392
scale = 0.004

# preparando a imagem para passar pelo modelo
blob = cv2.dnn.blobFromImage(img, scalefactor=scale, size=(416, 416), mean=(0,0,0), swapRB=True, crop=False)
model.setInput(blob)

num_camdas_saida = model.getUnconnectedOutLayers()
nomes_camadas = model.getLayerNames()
camadas_saida = []
for numero in num_camdas_saida:
    numero = numero[0] - 1   # no caso cada linha ainda retorna dentro de um outro vetor, por isso [0]
    camadas_saida.append(nomes_camadas[numero])  # pega o nome das camadas de saída

outputs = model.forward(camadas_saida)

boxes = []
confidences = []
classes_ids = []

for output in outputs:
    for detection in output:
        scores = detection[5:]
        classe_id = np.argmax(scores)  # retorna o índice de maior valor

        confidence = scores[classe_id] # valor de confiança da predição

        if (confidence > 0.5):
            center_x = int(detection[0]*width)
            center_y = int(detection[1]*height)
            w = int(detection[2]*width)
            h = int(detection[3]*height)
            x = center_x - w/2  # posição inicial do box em x
            y = center_y - h/2  # posição inicial do box em y

            boxes.append([x, y, w, h])
            classes_ids.append(classe_id)
            confidences.append(float(confidence))

conf_threshold = 0.5
nms_threshold = 0.4
indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

for indice in indices:
    ind = indice[0]
    box = boxes[ind]
    conf = confidences[ind]
    class_name = classes[classes_ids[ind]]
    texto = '{}, {}%'.format(class_name, round(100*conf))
    x, y, w, h = round(box[0]), round(box[1]), round(box[2]), round(box[3]) # é necessário fazer round nas variáveis
    cv2.rectangle(img, (x, y), (x+w, y+h), 0, 2)  # desenha retângulo
    cv2.putText(img, texto, (x-10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0, 2)

cv2.imshow('deteccao', img)
cv2.waitKey()
cv2.destroyAllWindows()