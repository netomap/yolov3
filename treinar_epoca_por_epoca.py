import torch
import os

device = torch.device('cpu')
caminho_pesos = '/home/manuel/PycharmProjects/yolov3/last.pt'
wdir = ''

for k in range(5):

    checkpoint = torch.load('./last.pt', map_location=device)
    proxima_epoca = checkpoint['epoch'] + 2
    comando = "python3 train.py --cfg './cfg/yolov3-tiny.cfg' " \
              "--wdir {} " \
              "--weights {} " \
              "--data 'dataset.data' " \
              "--epochs {} " \
              "--batch-size 1".format(wdir, caminho_pesos, proxima_epoca)

    os.system(comando)