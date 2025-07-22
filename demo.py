# -*- coding: utf-8 -*-
from ultralytics import YOLO
from tracker import update_tracker
import cv2
import imutils
import warnings
import collections

warnings.filterwarnings("ignore", category=UserWarning, message="torch.meshgrid: in an upcoming release")


class TargetDetector:
    def __init__(self, weights_path):
        self.model = YOLO(weights_path)
        self.names = self.model.names
        self.faceTracker = collections.defaultdict(lambda: 0)

    def detect(self, image):
        results = self.model.predict(
            source=image,
            conf=0.25,
            imgsz=640,
            verbose=False
        )
        bboxes = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            conf = float(box.conf[0].cpu().numpy())
            cls_id = int(box.cls[0].cpu().numpy())
            lbl = self.names[cls_id]

            if lbl in ['person', 'car', 'truck']:
                bboxes.append((x1, y1, x2, y2, lbl, conf))

        return results, bboxes

    def feedcap(self, im):
        retdict = {
            'frame': None,
            'vehicle_crops': None,
            'list_of_ids': None,
            'vehicle_bboxes': []
        }
        im, vehicle_crops, vehicle_bboxes, list_of_ids = update_tracker(self, im)

        retdict['frame'] = im
        retdict['vehicle_crops'] = vehicle_crops
        retdict['vehicle_bboxes'] = vehicle_bboxes
        retdict['list_of_ids'] = list_of_ids

        return retdict

def main():
    name = 'YOLO12 + DeepSORT Tracking Demo'
    detector = TargetDetector('weights/yolov12x.pt')

    cap = cv2.VideoCapture('traffic_car.mp4')
    fps = cap.get(cv2.CAP_PROP_FPS)
    t = int(1000 / fps)
    print(f'视频帧率 (FPS): {fps}')

    videoWriter = None 

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = detector.feedcap(frame)
        result_frame = imutils.resize(result['frame'], height=600)
        
        cv2.putText(result_frame, f"Vehicles: {len(result['list_of_ids'])}", (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        if videoWriter is None:
            fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
            videoWriter = cv2.VideoWriter('result_yolov12_deepsort.mp4', fourcc, fps,
                                          (result_frame.shape[1], result_frame.shape[0]))
        videoWriter.write(result_frame)

        print(f"[INFO] Current Frame IDs: {result['list_of_ids']}, Total: {len(result['list_of_ids'])}")

        cv2.imshow(name, result_frame)
        if cv2.waitKey(t) & 0xFF == 27:
            break

    cap.release()
    if videoWriter:
        videoWriter.release()
    cv2.destroyAllWindows()
    print("已保存视频 result_yolov12_deepsort.mp4")

if __name__ == '__main__':
    main()
