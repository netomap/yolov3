import torch
import os

device = torch.device('cpu')

for k in range(5):

    checkpoint = torch.load('/content/drive/My Drive/Colab Notebooks/YOLO/YOLO_PERSON_COCO_DATASET/last.pt', map_location=device)
    proxima_epoca = checkpoint['epoch'] + 2
    comando = "python3 train.py --cfg './cfg/yolov3-tiny.cfg' " \
              "--wdir '/content/drive/My Drive/Colab Notebooks/YOLO/YOLO_PERSON_COCO_DATASET/' " \
              "--weights '/content/drive/My Drive/Colab Notebooks/YOLO/YOLO_PERSON_COCO_DATASET/last.pt' " \
              "--data '/content/dataset.data' " \
              "--epochs {} " \
              "--batch-size 1".format(proxima_epoca)

    os.system(comando)