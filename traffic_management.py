import streamlit as st
import cv2
from ultralytics import YOLO
import time
import os
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
# ---------------- SETUP ----------------
st.set_page_config(layout="wide")
st.title("🚦 Intelligent Traffic Management System (Smart Cyclic Signals)")

# ---------------- CSS ----------------
st.markdown("""
<style>
.signal-card {
    border-radius: 16px;
    padding: 16px;
    color: white;
    text-align: center;
    font-weight: bold;
    font-size: 20px;
}
.big { font-size: 38px; }
.green { background:#00c853; box-shadow:0 0 25px #00e676; }
.yellow { background:#ffc400; box-shadow:0 0 20px #ffd740; color:black; }
.red { background:#d50000; }
.timer { font-size:15px; margin-top:6px; }
</style>
""", unsafe_allow_html=True)

# ---------------- STATE ----------------
st.session_state.setdefault("run", False)
st.session_state.setdefault("order", None)
st.session_state.setdefault("index", 0)
st.session_state.setdefault("start_time", time.time())

# ---------------- YOLO ----------------
model = YOLO("yolov8n.pt")
vehicle_classes = ["car", "bus", "truck", "motorcycle"]

# ---------------- VIDEOS ----------------
lanes = {
    "Lane 1": "traffic-videos/lane1.mp4",
    "Lane 2": "traffic-videos/lane2.mp4",
    "Lane 3": "traffic-videos/lane3.mp4",
    "Lane 4": "traffic-videos/lane4.mp4",
}
caps = {l: cv2.VideoCapture(p) for l, p in lanes.items()}

# ---------------- CONTROLS ----------------
c1, c2 = st.columns(2)
with c1:
    if st.button("▶ Start"):
        st.session_state.run = True
        st.session_state.order = None
        st.session_state.index = 0
        st.session_state.start_time = time.time()
with c2:
    if st.button("⏸ Pause"):
        st.session_state.run = False

# ---------------- UI ------------------------
vcols = st.columns(4)
vbox = {l: vcols[i].empty() for i, l in enumerate(lanes)}
scols = st.columns(4)
sbox = {l: scols[i].empty() for i, l in enumerate(lanes)}

# ---------------- MAIN LOOP ----------------
if st.session_state.run:
    while st.session_state.run:
        counts = {}

        # ---- READ FRAMES ----
        for lane, cap in caps.items():
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()

            frame = cv2.resize(frame, (400, 250))
            res = model(frame, verbose=False)[0]

            c = 0
            for box in res.boxes:
                if model.names[int(box.cls[0])] in vehicle_classes:
                    c += 1
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

            counts[lane] = c
            vbox[lane].image(frame, channels="BGR")

        # ---- SET ORDER ONCE PER FULL CYCLE ----
        if st.session_state.order is None:
            st.session_state.order = sorted(counts, key=counts.get, reverse=True)

        order = st.session_state.order
        idx = st.session_state.index

        green_lane = order[idx]
        yellow_lane = order[(idx + 1) % len(order)]

        # ---- TIMER ----
        elapsed = int(time.time() - st.session_state.start_time)
        remaining = 30 - elapsed

        if remaining <= 0:
            st.session_state.index = (idx + 1) % len(order)
            st.session_state.start_time = time.time()

            if st.session_state.index == 0:
                st.session_state.order = None  # 🔥 recompute counts after full round

        # ---- DISPLAY ----
        for lane in lanes:
            c = counts[lane]

            if lane == green_lane:
                sbox[lane].markdown(
                    f"<div class='signal-card green'>🟢 GREEN<div class='big'>{c}</div><div class='timer'>{remaining}s</div></div>",
                    unsafe_allow_html=True)

            elif lane == yellow_lane:
                sbox[lane].markdown(
                    f"<div class='signal-card yellow'>🟡 YELLOW<div class='big'>{c}</div></div>",
                    unsafe_allow_html=True)

            else:
                sbox[lane].markdown(
                    f"<div class='signal-card red'>🔴 RED<div class='big'>{c}</div></div>",
                    unsafe_allow_html=True)

        time.sleep(0.3)

# ---------------- CLEANUP ----------------
if not st.session_state.run:
    for cap in caps.values():
        cap.release()


