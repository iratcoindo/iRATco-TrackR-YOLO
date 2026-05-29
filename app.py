import streamlit as st
import cv2
import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt
import tempfile
from streamlit_image_coordinates import streamlit_image_coordinates
import pandas as pd

st.set_page_config(layout="wide")
st.title("🧬 CASA Sperm Velocity Tracker")

# =========================
# CONFIG
# =========================
MIN_AREA = 5
MAX_AREA = 200
PIXEL_TO_UM = 0.5
MAX_DISTANCE = 20

# =========================
# UPLOAD VIDEO
# =========================
uploaded_file = st.file_uploader("Upload sperm video", type=["mp4", "avi", "mov"])

if uploaded_file is None:
    st.info("👉 Upload video terlebih dahulu")
    st.stop()

# tampilkan video asli
st.video(uploaded_file)

# simpan sementara
tfile = tempfile.NamedTemporaryFile(delete=False)
tfile.write(uploaded_file.read())

cap = cv2.VideoCapture(tfile.name)

if not cap.isOpened():
    st.error("❌ Video tidak bisa dibuka")
    st.stop()

fps = cap.get(cv2.CAP_PROP_FPS)
st.write(f"FPS: {fps}")

# =========================
# TRACKING STORAGE
# =========================
tracks = {}
next_id = 0
prev_points = []

# display frame
frame_placeholder = st.empty()

# =========================
# HELPER
# =========================
def get_centroid(cnt):
    M = cv2.moments(cnt)
    if M["m00"] == 0:
        return None
    cx = int(M["m10"]/M["m00"])
    cy = int(M["m01"]/M["m00"])
    return (cx, cy)

def calc_velocity(track, fps):
    if len(track) < 2:
        return 0

    dist = 0
    for i in range(1, len(track)):
        x1, y1 = track[i-1]
        x2, y2 = track[i]
        dist += np.sqrt((x2-x1)**2 + (y2-y1)**2)

    dist_um = dist * PIXEL_TO_UM
    time = len(track) / fps

    return dist_um / time if time > 0 else 0

st.markdown("### 🎯 Klik sperma yang ingin dianalisis")

# =========================
# AMBIL FRAME
# =========================
cap_preview = cv2.VideoCapture(tfile.name)
ret, preview_frame = cap_preview.read()
cap_preview.release()

# =========================
# INIT SESSION
# =========================
if "selected_points" not in st.session_state:
    st.session_state.selected_points = []

# =========================
# PREVIEW + RESIZE (FIX SIZE)
# =========================
preview_rgb = cv2.cvtColor(preview_frame, cv2.COLOR_BGR2RGB)

TARGET_WIDTH = 600
TARGET_HEIGHT = 400

preview_rgb = cv2.resize(preview_rgb, (TARGET_WIDTH, TARGET_HEIGHT))

# =========================
# COLOR FUNCTION
# =========================
def get_color(i):
    colors = [
        (255,0,0), (0,255,0), (0,0,255),
        (255,255,0), (255,0,255), (0,255,255)
    ]
    return colors[i % len(colors)]
    
# =========================
# RESET BUTTON
# =========================
col_reset1, col_reset2 = st.columns([1,5])

with col_reset1:
    if st.button("🔄 Reset"):
        st.session_state.selected_points = []
        st.rerun()
        
# =========================
# LAYOUT
# =========================
col1, col2 = st.columns(2)

# =========================
# LEFT: CANVAS CLICK
# =========================
with col1:

    # ❌ JANGAN gambar anotasi di canvas
    preview_draw = preview_rgb.copy()

    clicked = streamlit_image_coordinates(
        preview_draw,
        key=f"click_{len(st.session_state.selected_points)}",
        width=TARGET_WIDTH
    )

    if clicked:
        x = clicked["x"]
        y = clicked["y"]

        st.session_state.selected_points.append((x, y))
        st.success(f"Selected: ({x},{y})")

# =========================
# RIGHT: PREVIEW RESULT
# =========================
with col2:

    preview_result = preview_rgb.copy()

    for i, (px, py) in enumerate(st.session_state.selected_points):
        cv2.circle(preview_result, (px, py), 3, get_color(i), -1)
        cv2.putText(preview_result, str(i+1), (px+5, py-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, get_color(i), 1)

    st.image(preview_result, width=TARGET_WIDTH)

# =========================
# RUN TRACKING FROM CLICK
# =========================
if st.button("▶️ Run Tracking Analysis"):

    if len(st.session_state.selected_points) == 0:
        st.warning("Pilih minimal 1 sperma dulu")
        st.stop()

    # reload video
    cap = cv2.VideoCapture(tfile.name)

    # ambil ukuran asli video
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # scaling dari preview ke original
    scale_x = orig_w / TARGET_WIDTH
    scale_y = orig_h / TARGET_HEIGHT

    # =========================
    # INIT TRACKS (FROM CLICK)
    # =========================
    tracks = {}

    for i, (px, py) in enumerate(st.session_state.selected_points):
        orig_x = int(px * scale_x)
        orig_y = int(py * scale_y)
        tracks[i] = [(orig_x, orig_y)]

    prev_points = [tracks[i][0] for i in tracks]

    fps = cap.get(cv2.CAP_PROP_FPS)

    # =========================
    # TRACKING LOOP
    # =========================
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)

        _, thresh = cv2.threshold(blur, 120, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        current_points = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if MIN_AREA < area < MAX_AREA:
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"]/M["m00"])
                    cy = int(M["m01"]/M["m00"])
                    current_points.append((cx, cy))

        # =========================
        # MATCHING (TRACK BY NEAREST)
        # =========================
        for tid in tracks:
            last_point = tracks[tid][-1]

            if len(current_points) == 0:
                continue

            dists = distance.cdist([last_point], current_points)[0]
            min_idx = np.argmin(dists)

            if dists[min_idx] < MAX_DISTANCE:
                tracks[tid].append(current_points[min_idx])

    cap.release()

    # =========================
    # ANALYSIS FUNCTION
    # =========================
    def analyze_track(track):

        if len(track) < 2:
            return None

        distances = []
        angles = []

        for i in range(1, len(track)):
            x1, y1 = track[i-1]
            x2, y2 = track[i]

            dx = x2 - x1
            dy = y2 - y1

            dist = np.sqrt(dx**2 + dy**2)
            distances.append(dist)

            angle = np.degrees(np.arctan2(dy, dx))
            angles.append(angle)

        # cumulative distance
        cum_dist = np.sum(distances) * PIXEL_TO_UM

        # velocity
        time = len(track) / fps
        velocity = cum_dist / time if time > 0 else 0

        # turn direction (change antar step)
        turn_angles = np.diff(angles)
        mean_turn = np.mean(np.abs(turn_angles)) if len(turn_angles) > 0 else 0

        # absolute bearing (start → end)
        x0, y0 = track[0]
        x_end, y_end = track[-1]
        abs_bearing = np.degrees(np.arctan2(y_end - y0, x_end - x0))

        return {
            "Cumulative Distance (µm)": cum_dist,
            "Velocity (µm/s)": velocity,
            "Turn Direction (°)": mean_turn,
            "Absolute Bearing (°)": abs_bearing
        }

    # =========================
    # RUN ANALYSIS
    # =========================
    results = []

    for tid, track in tracks.items():
        res = analyze_track(track)
        if res:
            res["ID"] = tid + 1
            results.append(res)

    df_result = pd.DataFrame(results)

    # =========================
    # OUTPUT
    # =========================
    st.markdown("## 📊 CASA Analysis Result")
    st.dataframe(df_result, use_container_width=True)

    # =========================
    # TRAJECTORY VISUAL (SIDE BY SIDE)
    # =========================
    st.markdown("## 📍 Trajectory Visualization")
    
    col_traj1, col_traj2 = st.columns(2)
    
    # =========================
    # LEFT: OVERLAY ON IMAGE
    # =========================
    with col_traj1:
    
        overlay_img = preview_rgb.copy()
    
        for tid, track in tracks.items():
    
            color = get_color(tid)
    
            # scale ke preview size
            scaled_track = [
                (int(x/scale_x), int(y/scale_y))
                for (x, y) in track
            ]
    
            # gambar trajectory
            for i in range(1, len(scaled_track)):
                cv2.line(
                    overlay_img,
                    scaled_track[i-1],
                    scaled_track[i],
                    color,
                    1,
                    lineType=cv2.LINE_AA
                )
    
            # label ID di titik akhir
            if len(scaled_track) > 0:
                x_end, y_end = scaled_track[-1]
                cv2.putText(
                    overlay_img,
                    str(tid+1),
                    (x_end+5, y_end),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1
                )
    
        st.image(overlay_img, width=TARGET_WIDTH)
    
    
    # =========================
    # RIGHT: CLEAN PLOT
    # =========================
    with col_traj2:
    
        fig, ax = plt.subplots()
    
        for tid, track in tracks.items():
    
            xs = [p[0] for p in track]
            ys = [p[1] for p in track]
    
            color = np.array(get_color(tid))/255.0
    
            ax.plot(
                xs,
                ys,
                linewidth=1.5,
                color=color,
                label=f"ID {tid+1}"
            )
    
            # titik akhir
            ax.scatter(xs[-1], ys[-1], color=color, s=20)
    
        ax.invert_yaxis()
        ax.set_title("Trajectory (Clean)")
        ax.legend()
    
        st.pyplot(fig)
