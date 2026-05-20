import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import ultralytics
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
from multiprocessing import freeze_support

trained_model_path = "masks/train4/weights/best.pt"
MODEL_PATH = 'yolov8s.pt'
input_dir = "cats"
output_dir = "result"


def main():
    model = YOLO(trained_model_path)
    #model.train(data="masked.yaml", model="yolov8n.pt", epochs=30, batch=8,
    #             project='masks', val=True, verbose=True, workers=0)
    os.makedirs(output_dir, exist_ok=True)

    test_images = [f for f in os.listdir(input_dir) if f.lower().endswith(".jpg")]

    for img_name in test_images:
        img_path = os.path.join(input_dir, img_name)

        results = model(img_path, conf=0.7, iou=0.45)

        result_img = results[0].plot()

        save_path = os.path.join(output_dir, img_name)
        cv2.imwrite(save_path, result_img)


if __name__ == '__main__':
    freeze_support()
    main()