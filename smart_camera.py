import streamlit as st
import cv2
from ultralytics import YOLO
import datetime

st.set_page_config(page_title="Интеллектуальное видеонаблюдение", layout="wide")
st.title("🎥 Логические правила и нейросетевые методы в интеллектуальных системах видеонаблюдения: живой пример определения объектов")

model = YOLO("yolov8n.pt")
start = st.button("▶️ Запустить камеру")

if start:
    cap = cv2.VideoCapture(0)
    stframe = st.empty()
    events = []

    line_y = 350 

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)
        annotated_frame = results[0].plot()
        boxes = results[0].boxes.data.cpu().numpy()

        for box in boxes:
            cls = int(box[5])
            if cls == 0:  
                x1, y1, x2, y2 = map(int, box[:4])
                mid_x = (x1 + x2) // 2
                mid_y = y2 

                cv2.circle(annotated_frame, (mid_x, mid_y), 5, (0, 0, 255), -1)
                if mid_y > line_y:
                    event = f"{datetime.datetime.now().strftime('%H:%M:%S')} — Пересечение линии!"
                    if event not in events:
                        events.append(event)
                    cv2.putText(annotated_frame, "IVT-21-24", (20, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        stframe.image(annotated_frame, channels="BGR")
    cap.release()
    st.subheader("📋 Журнал событий:")
    for e in events[-10:]:
        st.write(e)