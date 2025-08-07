from flask import Flask, request, jsonify
import uuid
import os
from datetime import datetime
import requests
import mainVideo
import json


app = Flask(__name__)

# 메모리 기반 상태 저장소
analysis_status_map = {}

@app.route('/analysis/video', methods=['POST'])
def analyze_video():
    try:
        data = request.get_json()
        presentation_id = data.get('presentationId')
        s3_url = data.get('s3Url')

        if not presentation_id or not s3_url:
            return jsonify({"error": "Missing presentationId or s3Url"}), 400

        # 고유 analysis ID 생성
        analysis_id = f"video-analysis-uuid-{str(uuid.uuid4())}"

        # 상태: PENDING
        analysis_status_map[analysis_id] = "PENDING"

        # presigned URL에서 영상 다운로드
        os.makedirs("downloads", exist_ok=True)
        filename = s3_url.split("?")[0].split("/")[-1]
        video_path = os.path.join("downloads", f"{analysis_id}_{filename}")

        # 상태: IN_PROGRESS
        analysis_status_map[analysis_id] = "IN_PROGRESS"

        download_success = download_video(s3_url, video_path)
        if not download_success:
            analysis_status_map[analysis_id] = "FAILED"
            return jsonify({"error": "영상 다운로드 실패"}), 500

        # 분석 실행
        try:
            mainVideo.run(video_path)
            analysis_status_map[analysis_id] = "COMPLETED"
        except Exception as e:
            analysis_status_map[analysis_id] = "FAILED"
            return jsonify({"error": f"분석 실패: {str(e)}"}), 500

        # 최종 응답
        return jsonify({
            "analysisId": analysis_id,
            "status": analysis_status_map[analysis_id]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/analysis/video/<analysis_id>', methods=['GET'])
def get_analysis_status(analysis_id):
    # 상태 조회
    status = analysis_status_map.get(analysis_id, "UNKNOWN")
    if not status:
        return jsonify({"error": "Analysis ID not found"}), 404
    
    if status != "COMPLETED":
        return jsonify({
            "analysisId": analysis_id,
            "status": status
        })
    
    # 결과 파일 경로
    result_path = os.path.join("results", f"{analysis_id}.json")

    if not os.path.exists(result_path):
        return jsonify({"error": "결과 파일을 찾을 수 없습니다."}), 500

    # 결과 JSON 불러와서 반환
    with open(result_path, "r", encoding="utf-8") as f:
        result = json.load(f)

    return jsonify(result)



def download_video(s3_url, output_path):
    import shutil
    """presigned S3 URL로 영상 다운로드"""
    try:
        if s3_url.startswith("file://"):
            # 윈도우 전용: /C:/ → C:/로 변환
            local_path = s3_url.replace("file://", "")
            if os.name == "nt" and local_path.startswith("/") and ":" in local_path:
                local_path = local_path[1:]  # 맨 앞 슬래시 제거

            print(f"[복사 경로] {local_path} → {output_path}")
            shutil.copy(local_path, output_path)
            return True
        response = requests.get(s3_url, stream=True, timeout=10)
        response.raise_for_status()

        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        return True

    except requests.exceptions.RequestException as e:
        print(f"[다운로드 실패] {e}")
        return False


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
