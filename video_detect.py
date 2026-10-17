from ultralytics import YOLO
import cv2

# 模型路径，用train-9
model = YOLO("runs/detect/train-9/weights/best.pt")
video_path = "evaluation/basketball_dribble_evaluation.mp4"

# 读取视频
cap = cv2.VideoCapture(video_path)
# 获取视频参数
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 输出视频保存位置
out = cv2.VideoWriter("output_result.mp4",
                      cv2.VideoWriter_fourcc(*'mp4v'),
                      fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # 推理
    results = model(frame, conf=0.5)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
            # 画框
            cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)
            # 类别+置信度文字
            conf = float(box.conf)
            cv2.putText(frame,f"basketball {conf:.2f}",(x1,y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,0),2)
            # 中心坐标、红点（加分项）
            cx = int((x1+x2)/2)
            cy = int((y1+y2)/2)
            cv2.circle(frame,(cx,cy),4,(0,0,255),-1)
            cv2.putText(frame,f"({cx},{cy})",(cx+6,cy),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1)
    out.write(frame)

cap.release()
out.release()
print("完整视频生成完毕，文件：output_result.mp4")