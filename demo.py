# -*- coding: utf-8 -*-
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2
import imutils
import warnings
import collections

warnings.filterwarnings("ignore", category=UserWarning, message="torch.meshgrid: in an upcoming release")

def main():
    name = 'YOLO + DeepSORT Tracking Demo'

    # 加载 YOLO 模型
    detector = YOLO('weights/yolov12x.pt')
    class_names = detector.names

    # 初始化 DeepSORT
    tracker = DeepSort(max_age=30, n_init=3)

    cap = cv2.VideoCapture('traffic_car.mp4')
    fps = cap.get(cv2.CAP_PROP_FPS)
    t = int(1000 / fps)
    print(f'视频帧率 (FPS): {fps}')

    videoWriter = None 

    # 🔥 追踪 ID 出现计数器
    trackCounter = collections.defaultdict(lambda: 0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # -----------------
        # 🔎 YOLO 检测
        # -----------------
        results = detector.predict(
            source=frame,
            conf=0.25,
            imgsz=640,
            verbose=False
        )

        detections = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            conf = float(box.conf[0].cpu().numpy())
            cls_id = int(box.cls[0].cpu().numpy())
            lbl = class_names[cls_id]
            
            # if conf < 0.5:
            #     continue
            if lbl in ['person', 'car', 'truck']:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f'{lbl} {conf:.2f}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                detections.append(([x1, y1, x2 - x1, y2 - y1], conf, lbl))

        # -----------------
        # 🟢 DeepSORT 跟踪
        # -----------------
        tracks = tracker.update_tracks(detections, frame=frame)
        frame_h, frame_w = frame.shape[:2]

        current_ids = set()
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            current_ids.add(track_id)

            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)
            w, h = x2 - x1, y2 - y1
            if w * h < 150 or w * h > frame_w * frame_h * 0.13:
                continue
            if x1 < 0 or y1 < 0 or x2 > frame_w or y2 > frame_h:
                continue

            # 🔵 绿色画正常跟踪框
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'ID: {track_id}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # 🔥 打印当前帧的所有目标 id 和数量
        print(f"[INFO] Current Frame IDs: {list(current_ids)}, Total: {len(current_ids)}")

        # -----------------
        # 🚀 删除跟踪 ID 逻辑
        # -----------------
        ids2delete = []
        for history_id in list(trackCounter.keys()):
            if history_id not in current_ids:
                trackCounter[history_id] -= 1
            else:
                trackCounter[history_id] = max(trackCounter[history_id], 0) + 1

            if trackCounter[history_id] < -5:
                ids2delete.append(history_id)

        for del_id in ids2delete:
            trackCounter.pop(del_id)
            print(f"-[INFO] Delete track id: {del_id}")

        # -----------------
        # resize & 写视频
        # -----------------
        result_frame = imutils.resize(frame, height=600)

        if videoWriter is None:
            fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
            videoWriter = cv2.VideoWriter('result_yolov12_deepsort.mp4', fourcc, fps,
                                          (result_frame.shape[1], result_frame.shape[0]))
        videoWriter.write(result_frame)

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

