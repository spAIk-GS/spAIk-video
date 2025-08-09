import cv2
import mediapipe as mp
import numpy as np
from face_detection import FaceDetector
from mark_detection import MarkDetector
from pose_estimation import PoseEstimator
from utils import refine
import math
from tqdm import tqdm

def run(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    face_detector = FaceDetector("assets/face_detector.onnx")
    mark_detector = MarkDetector("assets/face_landmarks.onnx")
    pose_estimator = PoseEstimator(frame_width, frame_height)

    picked_frame = 0
    head_down = 0

    mp_pose = mp.solutions.pose
    pose_a = mp_pose.Pose()

    THRESHOLD = 0.3
    prev_landmarks = None
    frame_count = 0
    movement_detected = 0
    total_checked = 0

    # 전체 프레임 수(일부 코덱/스트림은 -1일 수 있음)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # 너가 15프레임마다만 처리하니, 총 스텝도 그에 맞춰 계산
    total_steps = (total_frames // 15) if total_frames > 0 else None
    progress = tqdm(total=total_steps, desc="분석", unit="step", leave=True)

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % 15 != 0:   # 샘플링: 15프레임당 1스텝
                continue

            # 진행률 1스텝 업데이트
            if total_steps is not None:
                progress.update(1)

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose_a.process(image_rgb)

            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                lw = np.array([lm[mp_pose.PoseLandmark.LEFT_WRIST].x,
                               lm[mp_pose.PoseLandmark.LEFT_WRIST].y,
                               lm[mp_pose.PoseLandmark.LEFT_WRIST].z])
                rw = np.array([lm[mp_pose.PoseLandmark.RIGHT_WRIST].x,
                               lm[mp_pose.PoseLandmark.RIGHT_WRIST].y,
                               lm[mp_pose.PoseLandmark.RIGHT_WRIST].z])
                ls = np.array([lm[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                               lm[mp_pose.PoseLandmark.LEFT_SHOULDER].y,
                               lm[mp_pose.PoseLandmark.LEFT_SHOULDER].z])
                rs = np.array([lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                               lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].y,
                               lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].z])

                rel_lw = lw - ls
                rel_rw = rw - rs

                shoulder_dist = np.linalg.norm(ls - rs)
                if shoulder_dist < 1e-6:
                    shoulder_dist = 1e-6

                faces, _ = face_detector.detect(frame, 0.6)
                # 얼굴
                if len(faces) > 0:
                    picked_frame += 1
                    face = refine(faces, frame_width, frame_height, 0.15)[0]
                    x1, y1, x2, y2 = face[:4].astype(int)
                    patch = frame[y1:y2, x1:x2]
                    marks = mark_detector.detect([patch])[0].reshape([68, 2])
                    marks *= (x2 - x1)
                    marks[:, 0] += x1
                    marks[:, 1] += y1
                    pose_f = pose_estimator.solve(marks)
                    rotation_matrix, _ = cv2.Rodrigues(pose_f[0])
                    pitch_rad = math.atan2(rotation_matrix[2,1], rotation_matrix[2,2])
                    pitch_deg = np.degrees(pitch_rad)
                    if pitch_deg < -18:
                        head_down += 1

                # 팔
                if prev_landmarks is not None:
                    prev_rel_lw, prev_rel_rw = prev_landmarks
                    left_movement = np.linalg.norm(rel_lw - prev_rel_lw) / shoulder_dist
                    right_movement = np.linalg.norm(rel_rw - prev_rel_rw) / shoulder_dist
                    avg_movement = (left_movement + right_movement) / 2
                    total_checked += 1
                    if avg_movement > THRESHOLD:
                        movement_detected += 1

                prev_landmarks = [rel_lw, rel_rw]

        # total_frames을 못 읽은 경우, 마지막에 대략 완료 표시
        if total_steps is None:
            progress.set_description("분석(완료)")
        else:
            # 혹시 남은 스텝이 있다면 마저 채움
            remaining = total_steps - progress.n
            if remaining > 0:
                progress.update(remaining)

        head_down_ratio = head_down / picked_frame if picked_frame > 0 else 0.0
        arm_move_ratio = movement_detected / total_checked if total_checked > 0 else 0.0

        from videoFG import generate_posture_feedback
        gaze_feedback, gaze_level, gesture_feedback, gesture_level, summary = generate_posture_feedback(
            head_down_ratio, arm_move_ratio
        )

        report = {
            "body_movement": {"gestureFeedback": gesture_feedback, "value": gesture_level},
            "gaze": {"gazeFeedback": gaze_feedback, "value": gaze_level},
            "content_summary": summary
        }
        return report

    finally:
        progress.close()
        cap.release()
        try:
            pose_a.close()
        except Exception:
            pass
