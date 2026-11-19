from ultralytics import YOLO
import multiprocessing

# yaml会自动下载
def main():
    model = YOLO("/home/wuxi/CAF-YOLO/ultralytics/cfg/models/v8/yolov8-CAFMAttention.yaml")
# model = YOLO("d:/Data/yolov8s.pt")  # load a pretrained model (recommended for training)

        # Train the model
    # 指定保存位置：project + name -> {project}/{name}/weights/{best.pt,last.pt}
    results = model.train(
        data="/home/wuxi/yolo_fdi/data.yaml",
        batch=16,
        epochs=300,
        imgsz=640,
        amp=False,
        device=[0, 1],
        flipud=0.0,  # 🔥 牙科图像不要上下翻转
        fliplr=0.0,  # ✅ 你已经设置了
        project="/home/wuxi/CAF-YOLO/runs",
        name="exp0",
        exist_ok=True,
    )

    # ultralytics 会在运行目录下创建 {project}/{name}/weights/{best.pt, last.pt}
    print(f"Weights saved to: /home/wuxi/CAF-YOLO/runs/exp0/weights/")
    

if __name__=='__main__':
    multiprocessing.freeze_support()
    main()