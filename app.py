from streamlit_image_coordinates import streamlit_image_coordinates
import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tempfile
import zipfile
import io
import time
from matplotlib.colors import LinearSegmentedColormap

bg_sub = cv2.createBackgroundSubtractorMOG2(
    history=2000,
    varThreshold=25,
    detectShadows=False
)

# PAGE CONFIG
st.set_page_config(
    page_title="iRATco TrackR: Automated Rodent Behavior Analysis System",
    page_icon="logo.png",
    layout="wide"
)

# =========================
# DATA USER (hardcoded)
# =========================
USERS = {
    "admin": "iratcolab1",
    "lab": "iratcolab5"
}

# =========================
# SESSION INIT
# =========================
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

# =========================
# LOGIN
# =========================
if not st.session_state.authenticated:

    # 🔥 STYLE TAMBAHAN (INI SAJA YANG BARU)
    st.markdown("""
    <style>
    .stTextInput input {
        height: 45px;
        border-radius: 10px;
        border: 1px solid #e5e7eb;
        font-size: 15px;
    }

    .stButton button {
        height: 45px;
        width: 140px;
        border-radius: 10px;
        background-color: #2563eb;
        color: white;
        font-weight: 600;
        border: none;
    }

    .stButton button:hover {
        background-color: #1e40af;
    }

    /* subtle scientific background */
    body {
        background-color: #f9fafb;
    }
    </style>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([7,2])  # sedikit diperlebar

    with col1:
        st.markdown("""
        <h1 style="
            color:#4b5563;
            font-weight:600;
            white-space:nowrap;
            letter-spacing:0.3px;
        ">
        🔬 Login to iRATco Software
        </h1>

        <div style="
            color:#9ca3af;
            font-size:16px;
            margin-top:-10px;
            margin-bottom:20px;
        ">
        Advanced Laboratory Analysis Platform
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.image("logo_iratco.png", width=230)

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username in USERS and USERS[username] == password:
            st.session_state.authenticated = True
            st.session_state.username = username
            st.success(f"Welcome, {username} 👋")
        else:
            st.error("Invalid username or password. Contact us at: office-indo@iratco.co.id")

    st.stop()

# HEADER
col1, col2 = st.columns([6, 2])
with col1:
    st.title("iRATco TrackR: Automated Rodent Behavior Analysis System")
    st.markdown("<span style='font-size:16px;color:gray;'>**version 1.2.1**</span>", unsafe_allow_html=True)
with col2:
    st.image("logo_iratco.png", width=250)

uploaded_video = st.file_uploader("Upload your video")

if uploaded_video:
    if st.button("Reset ROI"):
        for key in ["roi", "roi_points", "pixel_to_mm", "real_roi_width_mm"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

# reset ROI if new video uploaded
if uploaded_video is not None:
    if "last_video" not in st.session_state:
        st.session_state.last_video = uploaded_video.name

    if uploaded_video.name != st.session_state.last_video:
        st.session_state.last_video = uploaded_video.name

        for key in ["roi", "roi_points", "pixel_to_mm", "real_roi_width_mm"]:
            if key in st.session_state:
                del st.session_state[key]

##### selector
roi = None

if uploaded_video:
    # simpan video ke temp file sekali saja
    video_bytes = uploaded_video.getvalue()
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_bytes)
    st.session_state.video_path = tfile.name

    cap = cv2.VideoCapture(st.session_state.video_path)
    ret, frame = cap.read()
    cap.release()

    if ret:
        display_width = 300
        scale = display_width / frame.shape[1]
        display_height = int(frame.shape[0] * scale)

        display_frame = cv2.resize(frame, (display_width, display_height))

        st.subheader("Select ROI (click TOP LEFT then BOTTOM RIGHT)")

        if "roi_points" not in st.session_state:
            st.session_state.roi_points = []

        roi_display = display_frame.copy()

        if "roi" in st.session_state:
            x, y, w, h = st.session_state.roi
            x_disp = int(x * scale)
            y_disp = int(y * scale)
            w_disp = int(w * scale)
            h_disp = int(h * scale)

            cv2.rectangle(
                roi_display,
                (x_disp, y_disp),
                (x_disp + w_disp, y_disp + h_disp),
                (0, 255, 0),
                3
            )

        point = streamlit_image_coordinates(roi_display)

        if point is not None and len(st.session_state.roi_points) < 2:
            real_x = int(point["x"] / scale)
            real_y = int(point["y"] / scale)
            st.session_state.roi_points.append((real_x, real_y))

        if len(st.session_state.roi_points) == 1 and "roi" not in st.session_state:
            st.info("Click BOTTOM RIGHT corner")

        if len(st.session_state.roi_points) == 2 and "roi" not in st.session_state:
            (x1, y1), (x2, y2) = st.session_state.roi_points

            x = min(x1, x2)
            y = min(y1, y2)
            w = abs(x2 - x1)
            h = abs(y2 - y1)

            st.session_state.roi = (x, y, w, h)
            st.rerun()

        # Calibration input
        if "roi" in st.session_state:
            x, y, w, h = st.session_state.roi

            st.subheader("Spatial Calibration")

            cal_col1, cal_col2 = st.columns(2)

            with cal_col1:
                st.write(f"ROI width (pixels): {w}")

            with cal_col2:
                real_roi_width_mm = st.number_input(
                    "Real length of ROI width (mm)",
                    min_value=0.0,
                    value=float(st.session_state.get("real_roi_width_mm", 500.0)),
                    step=10.0
                )

            st.session_state.real_roi_width_mm = real_roi_width_mm

            if w > 0 and real_roi_width_mm > 0:
                st.session_state.pixel_to_mm = real_roi_width_mm / w
                st.write(f"Scale: {st.session_state.pixel_to_mm:.4f} mm/pixel")

analysis_speed = st.selectbox(
    "Analysis Speed",
    ["1X", "2X", "4X", "8X", "20X"]
)

speed_map = {
    "1X": 1,
    "2X": 2,
    "4X": 4,
    "8X": 8,
    "20X": 20
}

skip = speed_map[analysis_speed]

# object contrast selection
contrast_mode = st.radio(
    "Object Type",
    ["Bright object", "Dark object"],
    horizontal=True
)

# SESSION STATE
if "running" not in st.session_state:
    st.session_state.running = False

# CONTROL BUTTONS
c1, c2 = st.columns(2)

with c1:
    if st.button("▶ Run Analysis"):
        if "roi" not in st.session_state:
            st.warning("Please select ROI first.")
        elif "pixel_to_mm" not in st.session_state:
            st.warning("Please enter real ROI width first.")
        else:
            st.session_state.running = True
            st.session_state.paused = False

with c2:
    if st.button("⏹ Stop Analysis"):
        st.session_state.running = False
        st.session_state.paused = False

def negative_mouse_view(frame):

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    if contrast_mode == "Bright object":

        neg = cv2.bitwise_not(gray)

    else:

        neg = gray.copy()

    # UNTUK KEDUA MODE
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8,8)
    )

    neg = clahe.apply(neg)

    neg = cv2.applyColorMap(
        neg,
        cv2.COLORMAP_BONE
    )

    # =========================
    # FOREGROUND MASK
    # =========================
    fgmask = bg_sub.apply(frame)

    kernel = np.ones((5,5), np.uint8)

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

    # =========================
    # AREA FILTER
    # =========================
    contours, _ = cv2.findContours(
        fgmask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) > 0:

        largest = max(
            contours,
            key=cv2.contourArea
        )

        area = cv2.contourArea(largest)

        if area > 50:

            # buat mask hanya untuk mouse
            mouse_mask = np.zeros_like(
                fgmask,
                dtype=np.uint8
            )

            cv2.drawContours(
                mouse_mask,
                [largest],
                -1,
                255,
                -1
            )

            # =========================
            # TRANSPARENT RED OVERLAY
            # =========================
            overlay = neg.copy()

            overlay[mouse_mask > 0] = [
                0,    # B
                0,    # G
                255   # R
            ]

            alpha = 0.60

            neg = cv2.addWeighted(
                overlay,
                alpha,
                neg,
                1 - alpha,
                0
            )

    return neg

def detect_mouse(frame):

    fgmask = bg_sub.apply(frame)

    kernel = np.ones((5,5), np.uint8)

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

    if len(contours) == 0:
        return None, None

    largest = max(
        contours,
        key=cv2.contourArea
    )

    area = cv2.contourArea(largest)

    if area < 30:
        return None, None

    M = cv2.moments(largest)

    if M["m00"] == 0:
        return None, None

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    return cx, cy

if uploaded_video and st.session_state.running:
    cap = cv2.VideoCapture(st.session_state.video_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    if "roi" in st.session_state:
        width = st.session_state.roi[2]
        height = st.session_state.roi[3]

    fps = cap.get(cv2.CAP_PROP_FPS)

    X = []
    Y = []

    video_col1, video_col2 = st.columns(2)

    with video_col1:
        st.markdown("**Raw Video**")
        raw_video = st.empty()

    with video_col2:
        st.markdown("**Tracking View**")
        neg_video = st.empty()

    progress = st.progress(0)

    st.markdown(
        "<h2 style='text-align:center;'>Movement Analysis</h2>",
        unsafe_allow_html=True
    )
    col1, col2, col3 = st.columns(3)

    traj_plot = col1.empty()
    dist_plot = col2.empty()
    vel_plot = col3.empty()

    st.markdown(
        "<h2 style='text-align:center;'>Spatial Behaviour</h2>",
        unsafe_allow_html=True
    )

    spatial_col1, spatial_col2 = st.columns(2)

    heat_plot = spatial_col1.empty()
    zone_plot = spatial_col2.empty()

    st.markdown(
        "<h2 style='text-align:center;'>Directional Analysis</h2>",
        unsafe_allow_html=True
    )

    dir_col1, dir_col2 = st.columns(2)

    bearing_plot = dir_col1.empty()
    turn_plot = dir_col2.empty()

    frame_id = 0
    saved_plots = []

    st.markdown(
        "<h2 style='text-align:center;'>Behavior Metrics</h2>",
        unsafe_allow_html=True
    )
    metrics_table = st.empty()

    st.markdown("""
    <style>
    .metrics-table {
        font-size:22px;
        text-align:center;
        margin:auto;
        margin-top:5px;
        margin-bottom:5px;
        border-collapse:collapse;
    }

    .metrics-table th {
        font-size:24px;
        font-weight:bold;
        text-align:center;
        padding:8px;
    }

    .metrics-table td {
        font-size:22px;
        padding:8px;
    }
    </style>
    """, unsafe_allow_html=True)

    while True:
        if not st.session_state.running:
            break

        ret, frame = cap.read()

        if not ret or frame is None:
            break

        if "roi" not in st.session_state:
            st.error("Please select ROI first.")
            st.stop()

        x, y, w, h = st.session_state.roi
        fh, fw = frame.shape[:2]

        x = max(0, min(int(x), fw - 1))
        y = max(0, min(int(y), fh - 1))
        w = max(1, min(int(w), fw - x))
        h = max(1, min(int(h), fh - y))

        if w <= 0 or h <= 0:
            st.error("ROI is invalid.")
            st.stop()

        frame = frame[y:y+h, x:x+w]

        if frame_id % skip != 0:
            frame_id += 1
            continue

        x, y = detect_mouse(frame)

        if x is not None and y is not None:
            X.append(x)
            Y.append(y)
        else:
            X.append(np.nan)
            Y.append(np.nan)

        neg_frame = negative_mouse_view(frame)

        if x is not None:
            cv2.circle(frame, (x, y), 6, (0, 0, 255), -1)
            cv2.circle(neg_frame, (x, y), 6, (255, 0, 0), -1)

        raw_video.image(frame, channels="BGR")
        neg_video.image(neg_frame, channels="BGR")

        track = pd.DataFrame({"X": X, "Y": Y})

        track["Y"] = height - track["Y"]

        track["Xs"] = track["X"].rolling(9, center=True).mean()
        track["Ys"] = track["Y"].rolling(9, center=True).mean()

        alpha = 0.2
        track["Xs"] = track["X"].ewm(alpha=alpha).mean()
        track["Ys"] = track["Y"].ewm(alpha=alpha).mean()

        track["Xs"].fillna(track["X"], inplace=True)
        track["Ys"].fillna(track["Y"], inplace=True)

        if len(track) > 2:
            track["dx"] = track["Xs"].diff()
            track["dy"] = track["Ys"].diff()

            track["step_distance"] = np.sqrt(track["dx"]**2 + track["dy"]**2)

            if "pixel_to_mm" in st.session_state:
                track["step_distance"] = track["step_distance"] * st.session_state.pixel_to_mm

            movement_threshold = 0.3
            track.loc[track["step_distance"] < movement_threshold, "Xs"] = np.nan
            track.loc[track["step_distance"] < movement_threshold, "Ys"] = np.nan

            track["Xs"] = track["Xs"].ffill()
            track["Ys"] = track["Ys"].ffill()
            
            dt = skip / fps
            track["velocity"] = track["step_distance"] / dt
            track["cumulative_distance"] = track["step_distance"].fillna(0).cumsum()

            track["bearing"] = np.arctan2(track["dy"], track["dx"])
            track["bearing_deg"] = np.degrees(track["bearing"])

            track["turn_angle"] = track["bearing_deg"].diff()
            track["turn_angle"] = (track["turn_angle"] + 180) % 360 - 180

            moving_velocity = track.loc[
                track["velocity"] > movement_threshold,
                "velocity"
            ]

            if len(moving_velocity) > 0:
                mean_velocity = moving_velocity.mean()
            else:
                mean_velocity = 0

            # minimal 5 frame berturut-turut bergerak
            track["moving"] = (
                track["velocity"] > movement_threshold
            ).rolling(5).sum() >= 5
            
            track.loc[
                ~track["moving"],
                "velocity"
            ] = np.nan

            # freezing
            freezing_threshold = 0.5
            track["freezing"] = track["velocity"] < freezing_threshold
            freezing_time = track["freezing"].sum() * dt

            # zone analysis
            cx = width / 2
            cy = height / 2
            center_radius = min(width, height) * 0.25

            dist_center = np.sqrt((track["Xs"] - cx)**2 + (track["Ys"] - cy)**2)
            track["zone"] = np.where(dist_center < center_radius, "center", "wall")

            center_time = (track["zone"] == "center").sum() * dt
            wall_time = (track["zone"] == "wall").sum() * dt

            anxiety_index = wall_time / (center_time + wall_time) if (center_time + wall_time) > 0 else np.nan

            # exploration
            grid_size = 5
            x_valid = track["Xs"].dropna()
            y_valid = track["Ys"].dropna()

            if len(x_valid) > 1 and len(y_valid) > 1:
                xbins = np.linspace(x_valid.min(), x_valid.max(), grid_size)
                ybins = np.linspace(y_valid.min(), y_valid.max(), grid_size)
                grid_counts, _, _ = np.histogram2d(x_valid, y_valid, bins=[xbins, ybins])
                visited_cells = np.sum(grid_counts > 0)
                total_cells = (grid_size - 1) * (grid_size - 1)
                exploration_index = visited_cells / total_cells if total_cells > 0 else np.nan
            else:
                exploration_index = np.nan

            total_distance = track["cumulative_distance"].iloc[-1]
            total_time = len(track) * dt

            if frame_id % 20 == 0:
                # Movement Trajectory = dwell time movement trajectory

                roi_width_mm = width * st.session_state.pixel_to_mm
                roi_height_mm = height * st.session_state.pixel_to_mm
                
                fig1, ax1 = plt.subplots(figsize=(6, 5))
                traj_data = track[["Xs", "Ys"]].dropna()

                if len(traj_data) > 0:
                    dwell_weights = np.full(len(traj_data), dt)

                    n_bins = 60
                    x_edges = np.linspace(0, width, n_bins + 1)
                    y_edges = np.linspace(0, height, n_bins + 1)

                    heatmap, _, _ = np.histogram2d(
                        traj_data["Xs"],
                        traj_data["Ys"],
                        bins=[x_edges, y_edges],
                        weights=dwell_weights
                    )

                    heatmap_smooth = cv2.GaussianBlur(heatmap, (0, 0), sigmaX=2.5, sigmaY=2.5)

                    dwell_cmap = LinearSegmentedColormap.from_list(
                        "dwell_cmap",
                        ["#ffffff", "#00ff00", "#ffff00", "#ff0000"]
                    )

                    im1 = ax1.imshow(
                        heatmap_smooth.T,
                        origin="lower",
                        extent=[
                            0,
                            roi_width_mm,
                            0,
                            roi_height_mm
                        ],
                        aspect="equal",
                        cmap=dwell_cmap,
                        interpolation="bilinear"
                    )

                    ax1.plot(
                        track["Xs"] * st.session_state.pixel_to_mm,
                        track["Ys"] * st.session_state.pixel_to_mm,
                        color="black",
                        alpha=0.35,
                        linewidth=1
                    )

                    cbar1 = fig1.colorbar(im1, ax=ax1, shrink=0.8)
                    cbar1.set_label("Dwell Time (s)")

                ax1.set_title("Movement Trajectory")
                ax1.set_xlabel("X (mm)")
                ax1.set_ylabel("Y (mm)")
                traj_plot.pyplot(fig1)
                plt.close(fig1)

                fig2, ax2 = plt.subplots(
                    figsize=(6,4)
                )
                
                ax2.plot(
                    track["cumulative_distance"],
                    linewidth=1.5
                )
                
                current_distance = (
                    track["cumulative_distance"]
                    .iloc[-1]
                )
                
                current_time = (
                    len(track) * dt
                )
                
                ax2.text(
                    0.02,
                    0.95,
                    f"Distance = {current_distance:.1f} mm",
                    transform=ax2.transAxes,
                    ha="left",
                    va="top",
                    fontsize=10,
                    fontweight="bold",
                    bbox=dict(
                        facecolor="white",
                        alpha=0.8
                    )
                )
                
                ax2.text(
                    0.02,
                    0.82,
                    f"Time = {current_time:.1f} s",
                    transform=ax2.transAxes,
                    ha="left",
                    va="top",
                    fontsize=10,
                    bbox=dict(
                        facecolor="white",
                        alpha=0.8
                    )
                )
                
                ax2.set_title(
                    "Cumulative Distance (mm)"
                )
                
                ax2.set_xlabel(
                    "Frame"
                )
                
                ax2.set_ylabel(
                    "Distance (mm)"
                )
                
                dist_plot.pyplot(fig2)
                plt.close(fig2)

                fig3, ax3 = plt.subplots(figsize=(6,4))

                ax3.plot(
                    track["velocity"],
                    linewidth=1.5
                )
                
                # garis mean
                ax3.axhline(
                    mean_velocity,
                    color="red",
                    linestyle="--",
                    linewidth=2,
                    alpha=0.8
                )
                
                ax3.set_title(
                    "Velocity (mm/s)"
                )

                ax3.text(
                    0.02,
                    0.95,
                    f"Mean = {mean_velocity:.2f} mm/s",
                    transform=ax3.transAxes,
                    ha="left",
                    va="top",
                    fontsize=10,
                    fontweight="bold",
                    bbox=dict(
                        facecolor="white",
                        alpha=0.8,
                        edgecolor="gray"
                    )
                )

                current_velocity = (
                    track["velocity"]
                    .dropna()
                    .iloc[-1]
                    if track["velocity"].notna().sum() > 0
                    else 0
                )
                ax3.text(
                    0.02,
                    0.82,
                    f"Current = {current_velocity:.2f} mm/s",
                    transform=ax3.transAxes,
                    ha="left",
                    va="top",
                    fontsize=10,
                    bbox=dict(
                        facecolor="white",
                        alpha=0.8,
                        edgecolor="gray"
                    )
                )

                vel_plot.pyplot(fig3)
                plt.close(fig3)

                # Visit Frequency Heatmap (tanpa trajectory)
                if len(track) > 20:
                    fig4, ax4 = plt.subplots()
                    heat_data = track[["Xs", "Ys"]].dropna()

                    if len(heat_data) > 20 and heat_data["Xs"].nunique() > 1 and heat_data["Ys"].nunique() > 1:
                        try:
                            sns.kdeplot(
                                x=heat_data["Xs"],
                                y=heat_data["Ys"],
                                fill=True,
                                cmap="RdYlGn_r",
                                ax=ax4
                            )
                        except Exception:
                            ax4.scatter(heat_data["Xs"], heat_data["Ys"], s=5, color="red")

                    ax4.set_aspect("equal")
                    ax4.set_title("Visit Frequency Heatmap")
                    heat_plot.pyplot(fig4)
                    plt.close(fig4)

                # absolute bearing
                bins = np.linspace(-180, 180, 24)

                fig5 = plt.figure(figsize=(4, 4))
                hist, _ = np.histogram(track["bearing_deg"].dropna(), bins=bins)
                theta = np.deg2rad((bins[:-1] + bins[1:]) / 2)

                ax5 = fig5.add_subplot(111, polar=True)
                ax5.bar(theta, hist, width=np.deg2rad(15))
                ax5.set_title("Absolute Bearing")
                bearing_plot.pyplot(fig5)
                plt.close(fig5)

                # turn direction
                fig6 = plt.figure(figsize=(4, 4))
                hist, _ = np.histogram(track["turn_angle"].dropna(), bins=bins)
                theta = np.deg2rad((bins[:-1] + bins[1:]) / 2)

                ax6 = fig6.add_subplot(111, polar=True)
                ax6.bar(theta, hist, width=np.deg2rad(15))
                ax6.set_title("Turn Direction")
                turn_plot.pyplot(fig6)
                plt.close(fig6)

                # zone occupancy
                fig7, ax7 = plt.subplots()
                zone_counts = track["zone"].value_counts()
                ax7.bar(zone_counts.index, zone_counts.values)
                ax7.set_title("Zone Occupancy")
                zone_plot.pyplot(fig7)
                plt.close(fig7)

                metrics_df = pd.DataFrame([{
                    "Mean velocity (mm/s)": round(mean_velocity, 2) if pd.notna(mean_velocity) else np.nan,
                    "Anxiety index": round(anxiety_index, 2) if pd.notna(anxiety_index) else np.nan,
                    "Freezing time (s)": round(freezing_time, 2) if pd.notna(freezing_time) else np.nan,
                    "Exploration index": round(exploration_index, 2) if pd.notna(exploration_index) else np.nan,
                    "Total Distance (mm)": round(total_distance, 2) if pd.notna(total_distance) else np.nan,
                    "Total Time (s)": round(total_time, 2) if pd.notna(total_time) else np.nan
                }])

                metrics_table.markdown(
                    metrics_df.to_html(classes="metrics-table", index=False),
                    unsafe_allow_html=True
                )

        frame_id += 1
        progress.progress(frame_id / total_frames)

    cap.release()
    st.success("Analysis complete")

    if "track" in locals() and not track.empty:
        csv = track.to_csv(index=False)

        st.download_button(
            label="Download Tracking Data (CSV)",
            data=csv,
            file_name="tracking_data.csv",
            mime="text/csv"
        )

st.markdown("---")

st.markdown("""
<div style="
    text-align:left;
    color:#6b7280;
    font-size:13px;
    padding-top:10px;
    padding-bottom:10px;
    border-top:1px solid #e5e7eb;
    margin-top:20px;
">
© 2026 Mawar Subangkit<br>
<b>Automated Rodent Behavior Analysis System Software</b><br><br>

If you use this software, please cite:<br>

<b>Subangkit</b>, MAWAR (2026)<br>
<i>iRATco TrackR: Automated Rodent Behavior Analysis System</i><br>

<a href="available at: https://iratco-trackr.streamlit.app/" target="_blank" style="color:#6b7280;">
available at: https://iratco-trackr.streamlit.app/
</a>
</div>
""", unsafe_allow_html=True)
