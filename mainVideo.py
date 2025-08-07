import cv2
import mediapipe as mp
import numpy as np

import json

import cv2

# 필요한 모듈 임포트
from face_detection import FaceDetector
from mark_detection import MarkDetector
from pose_estimation import PoseEstimator
from utils import refine
import math

video_path = "sample_input/testK.mp4"  # 비디오 파일 경로
video_id = video_path.split("/")[-1].split(".")[0]  # 비디오 ID 생성


# 초기 설정
def run(video_path):
    cap = cv2.VideoCapture(video_path)
    video_id = video_path.split("/")[-1].split(".")[0]  # 비디오 ID 생성
    # Get the frame size. This will be used by the following detectors.
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Setup a face detector to detect human faces.
    face_detector = FaceDetector("assets/face_detector.onnx")

    # Setup a mark detector to detect landmarks.
    mark_detector = MarkDetector("assets/face_landmarks.onnx")

    # Setup a pose estimator to solve pose.
    pose_estimator = PoseEstimator(frame_width, frame_height)


    frame_count = 0
    picked_frame = 0
    head_down = 0

    mp_pose = mp.solutions.pose

    pose_a = mp_pose.Pose()

    THRESHOLD = 0.3  # 움직임 감지 임계값 (조정 가능)
    prev_landmarks = None
    frame_count = 0
    movement_detected = 0
    total_checked = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 15 != 0:
            continue

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose_a.process(image_rgb)

        if results.pose_landmarks:
            # 현재 프레임에서 좌표 추출
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
            # 상대 위치 계산
            rel_lw = lw - ls
            rel_rw = rw - rs

            # 기준 스케일 (양 어깨 거리)
            shoulder_dist = np.linalg.norm(ls - rs)
            if shoulder_dist < 1e-6:
                shoulder_dist = 1e-6

            # 얼굴 감지
            faces, _ = face_detector.detect(frame, 0.6) # threshold 0.6
            
            if len(faces) > 0:
                picked_frame += 1

                # Detect landmarks. Crop and feed the face area into the
                # mark detector. Note only the first face will be used for
                # demonstration.
                face = refine(faces, frame_width, frame_height, 0.15)[0]
                x1, y1, x2, y2 = face[:4].astype(int)
                patch = frame[y1:y2, x1:x2]

                # Run the mark detection.
                marks = mark_detector.detect([patch])[0].reshape([68, 2])

                # Convert the locations from local face area to the global image.
                marks *= (x2 - x1)
                marks[:, 0] += x1
                marks[:, 1] += y1

                # pose estimation with 68 points.
                pose_f = pose_estimator.solve(marks)
                
                rotation_matrix, _ = cv2.Rodrigues(pose_f[0])
                pitch_rad = math.atan2(rotation_matrix[2,1], rotation_matrix[2,2])
                pitch_deg = np.degrees(pitch_rad)
                if pitch_deg < -18:
                    head_down += 1

            if prev_landmarks is not None:
                prev_rel_lw, prev_rel_rw = prev_landmarks

                left_movement = np.linalg.norm(rel_lw - prev_rel_lw) / shoulder_dist
                right_movement = np.linalg.norm(rel_rw - prev_rel_rw) / shoulder_dist
                avg_movement = (left_movement + right_movement) / 2

                total_checked += 1
                if avg_movement > THRESHOLD:
                    movement_detected += 1

            prev_landmarks = [rel_lw, rel_rw]

    head_down_ratio = head_down / picked_frame

    cap.release()
    pose_a.close()

    # Id 생성
    from datetime import datetime

    if total_checked > 0:
        arm_move_ratio = movement_detected / total_checked

    from videoFG import generate_posture_feedback

    gaze_feedback, gaze_level, gesture_feedback, gesture_level, summary = generate_posture_feedback(head_down_ratio, arm_move_ratio)

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    analysis_id = f"{video_id}_VideoFeedback_{timestamp}"
    
    report = {
        "analysisId": analysis_id,
        "videoId": video_id,
        "result": {
            "body_movement": {
                "gestureFeedback": gesture_feedback,
                "value": gesture_level,
            },
            "gaze": {
                "gazeFeedback": gaze_feedback,
                "value": gaze_level,
            },
            "content_summary": summary
        }
    }

    with open("videoAnalysis.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4, ensure_ascii=False)
    
    print(f" 팔 활동성 비율: {arm_move_ratio*100:.2f}%")
    print(f" 시선 오프스크린 비율: {head_down_ratio*100:.2f}%")



if __name__ == '__main__':
    run()