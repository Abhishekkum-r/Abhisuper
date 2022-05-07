from imageai.Detection import ObjectDetection
from tensorflow.keras.layers import BatchNormalization

import os

execution_path = os.getcwd()              

detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath( os.path.join(execution_path , "D:\Git\Ai\yolo-tiny.h5"))
detector.loadModel()
detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , "D:\Git\Ai\tr.jpg"), output_image_path=os.path.join(execution_path , "D:\Git\Ai\imagenew.jpg"))

for eachObject in detections:
    print(eachObject["name"] , " : " , eachObject["percentage_probability"] )