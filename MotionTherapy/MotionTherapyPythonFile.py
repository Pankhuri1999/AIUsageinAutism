#!/usr/bin/env python
# coding: utf-8

# In[ ]:




import time
import numpy as np
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
from fastdtw import fastdtw



mp_pose = mp.solutions.pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False
)



def extract_pose_sequence_with_stats(video_path):
    """
    Returns:
      pose_sequence: list of (33,2) arrays (only frames where pose detected)
      stats: dict with frames scanned, pose frames, latency etc.
    """
    cap = cv2.VideoCapture(video_path)

    pose_sequence = []
    total_frames = 0
    pose_frames = 0
    frame_times = []  

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        total_frames += 1

        t0 = time.perf_counter()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = mp_pose.process(rgb)
        t1 = time.perf_counter()

        frame_times.append(t1 - t0)

        if res.pose_landmarks:
            pose_frames += 1
            keypoints = np.array([[lm.x, lm.y] for lm in res.pose_landmarks.landmark], dtype=np.float32)
            pose_sequence.append(keypoints)

    cap.release()

    avg_ms = 1000 * (np.mean(frame_times) if frame_times else float("nan"))
    p95_ms = 1000 * (np.percentile(frame_times, 95) if frame_times else float("nan"))

    stats = {
        "video": video_path,
        "total_frames_scanned": total_frames,
        "frames_with_pose": pose_frames,
        "pose_detection_rate_%": 100 * pose_frames / max(total_frames, 1),
        "avg_processing_ms_per_frame": avg_ms,
        "p95_processing_ms_per_frame": p95_ms,
    }
    return pose_sequence, stats



def pose_to_motion_series(pose_seq):
    """
    1D motion signal: mean joint displacement between consecutive pose frames.
    Output length = len(pose_seq)-1
    """
    if len(pose_seq) < 2:
        return np.array([], dtype=np.float32)

    motion = []
    for i in range(1, len(pose_seq)):
        prev = pose_seq[i - 1]
        curr = pose_seq[i]
        disp = np.linalg.norm(curr - prev, axis=1)  # (33,)
        motion.append(float(np.mean(disp)))
    return np.array(motion, dtype=np.float32)


def z_normalize(x):
    x = np.asarray(x, dtype=np.float32)
    if len(x) == 0:
        return x
    return (x - x.mean()) / (x.std() + 1e-9)



def compare_two_videos(video1, video2, normalize_series=True):
    """
    Returns a dict with all outcomes + motion series + dtw path.
    """
    pose1, stats1 = extract_pose_sequence_with_stats(video1)
    pose2, stats2 = extract_pose_sequence_with_stats(video2)

    motion1 = pose_to_motion_series(pose1)
    motion2 = pose_to_motion_series(pose2)

    if normalize_series:
        motion1 = z_normalize(motion1)
        motion2 = z_normalize(motion2)

    if len(motion1) == 0 or len(motion2) == 0:
        return {
            "error": "Not enough pose frames to compute motion series for one or both videos.",
            "stats1": stats1,
            "stats2": stats2,
            "motion1_len": len(motion1),
            "motion2_len": len(motion2),
        }, motion1, motion2, []


    dtw_distance, path = fastdtw(motion1, motion2, dist=lambda a, b: abs(a - b))

    path_len = len(path)  
    unique_v1 = len(set(i for i, j in path))
    unique_v2 = len(set(j for i, j in path))

    cov_v1 = 100 * unique_v1 / max(len(motion1), 1)
    cov_v2 = 100 * unique_v2 / max(len(motion2), 1)

    dtw_cost = float(dtw_distance) / (path_len + 1e-9) 

    result = {
        "stats_video1": stats1,
        "stats_video2": stats2,
        "motion1_len": int(len(motion1)),
        "motion2_len": int(len(motion2)),
        "fastdtw_distance": float(dtw_distance),
        "alignment_path_len_total_pairs": int(path_len),
        "unique_frames_compared_video1": int(unique_v1),
        "unique_frames_compared_video2": int(unique_v2),
        "coverage_%_video1": float(cov_v1),
        "coverage_%_video2": float(cov_v2),
        "dtw_cost_distance_per_step": float(dtw_cost),
 
    }
    return result, motion1, motion2, path



def distance_to_similarity_single_pair(d, alpha=1.0):
    """
    For a SINGLE pair (no dataset min/max), use a smooth inverse mapping:
      sim01 = exp(-alpha * d)
      sim_1_10 = 1 + 9*sim01
    Lower d => higher similarity.
    alpha controls how fast similarity drops with distance.
    """
    d = float(d)
    sim01 = np.exp(-alpha * d)
    return float(1 + 9 * sim01)



def plot_motion_overlap(m1, m2, title="Motion overlap (normalized)"):
    plt.figure(figsize=(10, 4))
    plt.plot(m1, label="Video 1 motion")
    plt.plot(m2, label="Video 2 motion")
    plt.xlabel("Time index")
    plt.ylabel("Motion (normalized)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_dtw_alignment_path(path, title="FastDTW alignment path"):
    if not path:
        print("No DTW path to plot.")
        return
    xs = [p[0] for p in path]
    ys = [p[1] for p in path]
    plt.figure(figsize=(6, 6))
    plt.plot(xs, ys, linewidth=1)
    plt.xlabel("Video 1 time index")
    plt.ylabel("Video 2 time index")
    plt.title(title)
    plt.grid(True)
    plt.show()



if __name__ == "__main__":
    VIDEO1_PATH = "degree1.mp4"
    VIDEO2_PATH = "degree2.mp4"

    result, motion1, motion2, path = compare_two_videos(VIDEO1_PATH, VIDEO2_PATH, normalize_series=True)

    if "error" in result:
        print("ERROR:", result["error"])
        print("Video1 stats:", result["stats1"])
        print("Video2 stats:", result["stats2"])
    else:
        # Single-pair similarity score (1..10) using exp mapping
        sim_1_10 = distance_to_similarity_single_pair(result["dtw_cost_distance_per_step"], alpha=3.0)
        result["similarity_1_to_10_single_pair"] = sim_1_10

        print("\n==================== FINAL OUTCOME ====================")
        print("Video 1:", result["stats_video1"]["video"])
        print("  Frames scanned:", result["stats_video1"]["total_frames_scanned"])
        print("  Pose frames:", result["stats_video1"]["frames_with_pose"])
        print("  Pose detection rate (%):", f'{result["stats_video1"]["pose_detection_rate_%"]:.2f}')
        print("  Avg latency (ms/frame):", f'{result["stats_video1"]["avg_processing_ms_per_frame"]:.2f}')
        print("  P95 latency (ms/frame):", f'{result["stats_video1"]["p95_processing_ms_per_frame"]:.2f}')

        print("\nVideo 2:", result["stats_video2"]["video"])
        print("  Frames scanned:", result["stats_video2"]["total_frames_scanned"])
        print("  Pose frames:", result["stats_video2"]["frames_with_pose"])
        print("  Pose detection rate (%):", f'{result["stats_video2"]["pose_detection_rate_%"]:.2f}')
        print("  Avg latency (ms/frame):", f'{result["stats_video2"]["avg_processing_ms_per_frame"]:.2f}')
        print("  P95 latency (ms/frame):", f'{result["stats_video2"]["p95_processing_ms_per_frame"]:.2f}')

        print("\nMotion series lengths:")
        print("  motion1_len:", result["motion1_len"])
        print("  motion2_len:", result["motion2_len"])

        print("\nFastDTW:")
        print("  distance (lower=more similar):", f'{result["fastdtw_distance"]:.6f}')
        print("  alignment path len (total frame-pairs compared):", result["alignment_path_len_total_pairs"])
        print("  unique frames compared (video1):", result["unique_frames_compared_video1"],
              f'({result["coverage_%_video1"]:.1f}%)')
        print("  unique frames compared (video2):", result["unique_frames_compared_video2"],
              f'({result["coverage_%_video2"]:.1f}%)')
        print("  normalized DTW cost (distance/path_len):", f'{result["dtw_cost_distance_per_step"]:.8f}')

        print("\nSimilarity (single pair, 1..10):", f'{result["similarity_1_to_10_single_pair"]:.3f}')
        print("=======================================================\n")


        plot_motion_overlap(motion1, motion2, title="Motion overlap (z-normalized)")
        plot_dtw_alignment_path(path, title="FastDTW alignment path")


# In[ ]:




