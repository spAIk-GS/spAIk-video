def generate_posture_feedback(head_down_ratio, arm_movement_ratio):
    # 시선 피드백
    if head_down_ratio > 0.37:
        gaze_feedback = "시선이 자주 불안정합니다. 발표 중에는 청중을 바라보는 자세를 유지하는 것이 좋습니다."
        gaze_level = "불안"
    elif head_down_ratio > 0.18:
        gaze_feedback = "가끔 시선이 아래로 향했지만, 전반적으로 괜찮은 편입니다. 조금 더 정면을 바라보면 좋겠습니다."
        gaze_level = "부족"
    else:
        gaze_feedback = "시선이 안정적이고, 청중과의 아이 컨택이 잘 유지되었습니다."
        gaze_level = "좋음"

    # 제스처 피드백
    if arm_movement_ratio < 0.28:
        gesture_feedback = "발표 중 움직임이 거의 없습니다. 너무 경직되어 보일 수 있으니 자연스럽게 제스처를 섞어보세요."
        gesture_level = "경직"
    elif arm_movement_ratio > 0.75:
        gesture_feedback = "팔을 자주 움직였습니다. 너무 과한 제스처는 집중을 방해할 수 있으니 주의하세요."
        gesture_level = "과함"
    else:
        gesture_feedback = "팔의 움직임이 적절하게 활용되었습니다. 자연스러운 제스처는 발표에 도움이 됩니다."
        gesture_level = "적정"

    # 종합 요약
    if gaze_level == "good" and gesture_level == "good":
        summary = "전반적으로 시선 처리와 자세가 안정적이었습니다."
    elif gaze_level == "good" and gesture_level == "high":
        summary = "시선은 안정적이었지만, 제스처가 다소 과했습니다."
    elif gaze_level == "bad" and gesture_level == "good":
        summary = "제스처는 자연스러웠지만, 시선이 자주 불안정했습니다."
    elif gaze_level == "bad" and gesture_level == "low":
        summary = "시선과 제스처 모두 개선이 필요해 보입니다."
    else:
        summary = "시선과 제스처에서 개선할 부분이 일부 보입니다."

    return gaze_feedback, gaze_level, gesture_feedback, gesture_level, summary
