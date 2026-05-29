import streamlit as st
import cv2
import numpy as np
import pandas as pd
import tempfile
import matplotlib.pyplot as plt
from scipy.spatial import distance
from streamlit_image_coordinates import streamlit_image_coordinates

st.set_page_config(layout="wide")
st.title("CASA MOG2 Tracker")

MIN_AREA = 5
MAX_AREA = 200
PIXEL_TO_UM = 0.5
MAX_DISTANCE = 25

uploaded_file = st.file_uploader("Upload sperm video", type=["mp4","avi","mov"])

if uploaded_file is None:
    st.stop()

tfile = tempfile.NamedTemporaryFile(delete=False)
tfile.write(uploaded_file.read())

cap_preview = cv2.VideoCapture(tfile.name)
ret, preview = cap_preview.read()
cap_preview.release()

if not ret:
    st.error("Cannot read video")
    st.stop()

TARGET_WIDTH = 700
TARGET_HEIGHT = int(preview.shape[0] * TARGET_WIDTH / preview.shape[1])

preview_rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
preview_rgb = cv2.resize(preview_rgb, (TARGET_WIDTH, TARGET_HEIGHT))

if "selected_points" not in st.session_state:
    st.session_state.selected_points = []

col1, col2 = st.columns(2)

with col1:
    clicked = streamlit_image_coordinates(
        preview_rgb,
        key=f"click_{len(st.session_state.selected_points)}",
        width=TARGET_WIDTH
    )

    if clicked:
        st.session_state.selected_points.append(
            (clicked["x"], clicked["y"])
        )
        st.rerun()

with col2:
    vis = preview_rgb.copy()
    for i, (x, y) in enumerate(st.session_state.selected_points):
        cv2.circle(vis, (x, y), 4, (255, 0, 0), -1)
        cv2.putText(vis, str(i+1), (x+5, y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255,0,0), 1)
    st.image(vis, width=TARGET_WIDTH)

if st.button("Reset"):
    st.session_state.selected_points = []
    st.rerun()

if st.button("Run CASA MOG2 Analysis"):

    cap = cv2.VideoCapture(tfile.name)

    fps = cap.get(cv2.CAP_PROP_FPS)

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    scale_x = orig_w / TARGET_WIDTH
    scale_y = orig_h / TARGET_HEIGHT

    tracks = {}

    for i, (px, py) in enumerate(st.session_state.selected_points):
        tracks[i] = [(int(px*scale_x), int(py*scale_y))]

    bg_sub = cv2.createBackgroundSubtractorMOG2(
        history=500,
        varThreshold=16,
        detectShadows=False
    )

    frame_holder = st.empty()

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        fgmask = bg_sub.apply(frame)

        kernel = np.ones((3,3), np.uint8)

        fgmask = cv2.morphologyEx(
            fgmask,
            cv2.MORPH_OPEN,
            kernel
        )

        fgmask = cv2.morphologyEx(
            fgmask,
            cv2.MORPH_CLOSE,
            kernel
        )

        contours, _ = cv2.findContours(
            fgmask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        current_points = []
        display = cv2.cvtColor(fgmask, cv2.COLOR_GRAY2BGR)

        for cnt in contours:

            area = cv2.contourArea(cnt)

            if MIN_AREA < area < MAX_AREA:

                M = cv2.moments(cnt)

                if M["m00"] > 0:

                    cx = int(M["m10"]/M["m00"])
                    cy = int(M["m01"]/M["m00"])

                    current_points.append((cx, cy))

                    cv2.circle(display, (cx,cy), 2, (0,0,255), -1)

        for tid in tracks:

            if len(current_points) == 0:
                continue

            last_point = tracks[tid][-1]

            dists = distance.cdist(
                [last_point],
                current_points
            )[0]

            idx = np.argmin(dists)

            if dists[idx] < MAX_DISTANCE:
                tracks[tid].append(current_points[idx])
            else:
                tracks[tid].append(last_point)

        frame_holder.image(display, channels="BGR")

    cap.release()

    results = []

    def analyze(track):

        if len(track) < 2:
            return None

        d = []

        for i in range(1, len(track)):
            x1,y1 = track[i-1]
            x2,y2 = track[i]

            d.append(np.sqrt((x2-x1)**2 + (y2-y1)**2))

        cum_dist = np.sum(d) * PIXEL_TO_UM

        time_sec = len(track)/fps

        vcl = cum_dist / time_sec if time_sec > 0 else 0

        x0,y0 = track[0]
        xn,yn = track[-1]

        straight = np.sqrt(
            (xn-x0)**2 +
            (yn-y0)**2
        ) * PIXEL_TO_UM

        vsl = straight / time_sec if time_sec > 0 else 0

        lin = (vsl/vcl*100) if vcl > 0 else 0

        return {
            "VCL (um/s)": round(vcl,2),
            "VSL (um/s)": round(vsl,2),
            "LIN (%)": round(lin,2),
            "Distance (um)": round(cum_dist,2)
        }

    for tid, tr in tracks.items():
        res = analyze(tr)
        if res:
            res["ID"] = tid + 1
            results.append(res)

    df = pd.DataFrame(results)

    st.subheader("Results")
    st.dataframe(df, use_container_width=True)

    fig, ax = plt.subplots(figsize=(6,6))

    for tid, tr in tracks.items():
        xs = [p[0] for p in tr]
        ys = [p[1] for p in tr]

        ax.plot(xs, ys, linewidth=1)

    ax.invert_yaxis()
    ax.set_title("Trajectory")

    st.pyplot(fig)

    csv = df.to_csv(index=False)

    st.download_button(
        "Download CSV",
        csv,
        "casa_results.csv",
        "text/csv"
    )
