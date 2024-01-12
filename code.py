import cv2
import mediapipe as mp
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
##
def predict(chosen_model, img, classes=[], conf=0.5):
    if classes:
        results = chosen_model.predict(img, classes=classes, conf=conf)
    else:
        results = chosen_model.predict(img, conf=conf)
    return results


def predict_and_detect(chosen_model, img, classes=[], conf=0.5):
    results = predict(chosen_model, img, classes, conf=conf)
    for result in results:
        for box in result.boxes:
            cv2.rectangle(img, (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                          (int(box.xyxy[0][2]), int(box.xyxy[0][3])), (255, 0, 0), 2)
            cv2.putText(img, f"{result.names[int(box.cls[0])]}",
                        (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
                        cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)
    return img
###
def main():
    cap=cv2.VideoCapture("cars.mp4")
    while True: 
        success,img=cap.read()
        if success==False:
            break
        result_img = predict_and_detect(model, img, classes=[1,2,3,5,7], conf=0.5)
        cv2.imshow("sss",result_img)
        if cv2.waitKey(1) == ord('q'):
            break
if __name__=="__main__":
    main()