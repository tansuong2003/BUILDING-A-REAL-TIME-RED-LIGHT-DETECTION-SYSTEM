!pip install ultralytics --upgrade -q
from ultralytics import YOLO
import torch

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn', force=True) 

model = YOLO('/kaggle/input/yolo11-dengiaothongv1/other/default/1/yolo11x.pt')

results = model.train(
    data='/kaggle/input/config-den/mydata_den.yaml',  # đường dẫn file yaml của bạn
    epochs=200,
    batch=16,
    imgsz=416,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    workers=8,
    amp=True,
    project='/kaggle/working/',
    name='dendo_traffic_light',
)