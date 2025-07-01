# Cài ultralytics nếu chưa có
!pip install -q ultralytics

from ultralytics import YOLO
import torch
import os

def create_yaml():
    # Đường dẫn tới file YAML
    yaml_path = "/kaggle/working/mydata_biendo.yaml"

    # Nội dung YAML chuẩn theo nhãn bạn dùng
    yaml_content = """
path: /kaggle/input/datakytu/datakytu

train: images/train
val: images/val
test:

names:
  0: '0'
  1: '1'
  2: '2'
  3: '3'
  4: '4'
  5: '5'
  6: '6'
  7: '7'
  8: '8'
  9: '9'
  10: 'A'
  11: 'B'
  12: 'C'
  13: 'D'
  14: 'E'
  15: 'F'
  16: 'G'
  17: 'H'
  18: 'K'
  19: 'L'
  20: 'M'
  21: 'N'
  22: 'P'
  23: 'R'
  24: 'S'
  25: 'T'
  26: 'U'
  27: 'V'
  28: 'X'
  29: 'Y'
  30: 'Z'
"""

    # Ghi file YAML
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    print(f"✅ YAML file created at: {yaml_path}")
    return yaml_path

def train_model(yaml_path):
    torch.multiprocessing.set_start_method('spawn', force=True) 
    model = YOLO("/kaggle/input/yolo11-dengiaothongv1/other/default/1/yolo11x.pt")  
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    results = model.train(
        data=yaml_path,
        epochs=200,
        batch=16,
        device='0',
        workers=4,
        cache=True,
        amp=True,
        imgsz=640,
        project="/kaggle/working/",
        name="train_label_bienso",
        verbose=True
    )

if __name__ == '__main__':
    yaml_path = create_yaml()
    train_model(yaml_path)