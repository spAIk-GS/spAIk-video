from flask import Flask, request, jsonify
import uuid
import os
import requests
import mainVideo
import threading
import time
from tqdm import tqdm
import tempfile
from urllib.parse import urlparse


app = Flask(__name__)

# 메모리 기반 상태 저장소 (+ 락)
analysis_status_map = {}
status_lock = threading.Lock()


def set_status(analysis_id, status):
    with status_lock:
        analysis_status_map[analysis_id] = status


def get_status(analysis_id):
    with status_lock:
        return analysis_status_map.get(analysis_id)


def notify_status(callback_url, payload, retries=3):
    """
    콜백 POST (payload는 이미 완성된 dict)
    """
    delay = 1.0
    for attempt in range(1, retries + 1):
        try:
            res = requests.post(callback_url, json=payload, timeout=10)
            print(f"[POST] to {callback_url} -> {res.status_code}")
            if 200 <= res.status_code < 300:
                return True
        except Exception as e:
            print(f"[실패] POST (attempt {attempt}/{retries}) -> {e}")
        if attempt < retries:
            time.sleep(delay)
            delay *= 2
    return False



def download_video(s3_url, output_path):
    """presigned S3 URL 또는 file:// 로 영상 다운로드 (tqdm로 진행률 표시)"""
    try:
        chunk_size = 1024 * 256  # 256KB

        if s3_url.startswith("file://"):
            # 로컬 파일 복사도 진행률 표시
            local_path = s3_url.replace("file://", "")
            if os.name == "nt" and local_path.startswith("/") and ":" in local_path:
                local_path = local_path[1:]

            total_size = os.path.getsize(local_path)
            print(f"[복사] {local_path} -> {output_path}")
            with open(local_path, "rb") as src, open(output_path, "wb") as dst, tqdm(
                total=total_size,
                unit="B",
                unit_scale=True,
                desc="다운로드(로컬 복사)",
                leave=True,
            ) as pbar:
                while True:
                    buf = src.read(chunk_size)
                    if not buf:
                        break
                    dst.write(buf)
                    pbar.update(len(buf))
            print("[다운로드 완료] (로컬 복사)")
            return True

        # 원격 다운로드
        with requests.get(s3_url, stream=True, timeout=10) as response:
            response.raise_for_status()
            total_size = int(response.headers.get("content-length", 0))
            with open(output_path, 'wb') as f, tqdm(
                total=total_size if total_size > 0 else None,
                unit="B",
                unit_scale=True,
                desc="다운로드",
                leave=True,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if not chunk:
                        continue
                    f.write(chunk)
                    pbar.update(len(chunk))
        print("[다운로드 완료]")
        return True

    except requests.exceptions.RequestException as e:
        print(f"[다운로드 실패] {e}")
        return False
    except Exception as e:
        print(f"[다운로드 예외] {e}")
        return False

def process_video(s3_url, analysis_id, presentation_id, callback_url):
    set_status(analysis_id, "IN_PROGRESS")

    # s3_url에서 미리 video_id 추출 (다운로드 실패시에도 사용)
    url_path = urlparse(s3_url).path
    url_name = os.path.basename(url_path)
    video_id_from_url = os.path.splitext(url_name)[0] if url_name else analysis_id

    with tempfile.TemporaryDirectory(prefix="dl_") as tmpdir:
        video_path = os.path.join(tmpdir, f"{analysis_id}.mp4")

        # 1) 다운로드
        if not download_video(s3_url, video_path):
            set_status(analysis_id, "FAILED")
            fail_payload = {
                "analysisId": analysis_id,
                "videoId": video_id_from_url,  # URL에서 추출한 id 사용
                # result 없음
            }
            notify_status(callback_url, fail_payload)
            return

        try:
            # 2) 분석 실행 (mainVideo.run이 dict 반환)
            result_data = mainVideo.run(video_path)
            if not isinstance(result_data, dict):
                raise ValueError("mainVideo.run 결과 형식이 dict가 아닙니다.")

            # 파일 기준 video_id 재계산(확실히)
            video_id = os.path.splitext(os.path.basename(video_path))[0]

            # 3) 완료 통지: 최종 payload 조립 (presentationId/status는 빼는 조건)
            final_payload = {
                "analysisId": analysis_id,
                "videoId": video_id,
                "result": result_data
            }

            set_status(analysis_id, "COMPLETED")
            notify_status(callback_url, final_payload)

        except Exception as e:
            print(f"[분석 실패] {e}")
            set_status(analysis_id, "FAILED")
            fail_payload = {
                "analysisId": analysis_id,
                "videoId": video_id_from_url,
            }
            notify_status(callback_url, fail_payload)
            return


@app.route('/analysis/video', methods=['POST'])
def analyze_video():
    data = request.get_json()
    presentation_id = data.get("presentationId")
    s3_url = data.get("s3Url")
    callback_url = data.get("callbackUrl")

    if not all([presentation_id, s3_url, callback_url]):
        return jsonify({"error": "presentationId, s3Url, callbackUrl은 필수입니다."}), 400

    analysis_id = f"video-analysis-uuid-{uuid.uuid4()}"

    # 초기 상태: PENDING
    set_status(analysis_id, "PENDING")

    # 백그라운드 작업 시작 
    thread = threading.Thread(
        target=process_video,
        args=(s3_url, analysis_id, presentation_id, callback_url),
        daemon=False
    )
    thread.start()

    # 즉시 PENDING 응답
    return jsonify({
        "analysisId": analysis_id,
        "status": "PENDING"
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)