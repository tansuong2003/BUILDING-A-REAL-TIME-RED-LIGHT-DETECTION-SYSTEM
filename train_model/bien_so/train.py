from ultralytics import YOLO
import torch

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn', force=True) 
    
    model = YOLO("/kaggle/input/yolo11/pytorch/default/1/yolo11x.pt")  
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = model.train(
        data="/kaggle/input/mydata-biensoxe/mydata_biensoxe.yaml",
        epochs=200,
        batch=16,              
        device='0,1',
        workers=8,
        cache=True,
        amp=True,
        imgsz=640,      
        project="/kaggle/working/",
        name="license_plate",
        verbose=True
    )