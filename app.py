import os
import json
import tempfile
import traceback
import cv2
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)

# Allow requests from Creator OS frontend
CORS(app, origins=[
    'https://creator-os-frontend-production.up.railway.app',
    'http://localhost:8080',
    'http://127.0.0.1:8080',
    '*'  # open for testing — restrict later
])

# ── Load face detector once at startup ──────────────────────────────────────
# OpenCV's DNN-based face detector — fast and accurate
MODEL_DIR  = os.path.join(os.path.dirname(__file__), 'models')
PROTO_PATH = os.path.join(MODEL_DIR, 'deploy.prototxt')
MODEL_PATH = os.path.join(MODEL_DIR, 'res10_300x300_ssd_iter_140000.caffemodel')

face_net = None

def load_model():
    global face_net
    if not os.path.exists(PROTO_PATH) or not os.path.exists(MODEL_PATH):
        print('[FaceTracker] Downloading face detection model...')
        os.makedirs(MODEL_DIR, exist_ok=True)
        import urllib.request
        urllib.request.urlretrieve(
            'https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt',
            PROTO_PATH
        )
        urllib.request.urlretrieve(
            'https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel',
            MODEL_PATH
        )
    face_net = cv2.dnn.readNetFromCaffe(PROTO_PATH, MODEL_PATH)
    print('[FaceTracker] Model loaded OK')

# ── Health check ─────────────────────────────────────────────────────────────
@app.route('/', methods=['GET'])
def health():
    return jsonify({
        'status'  : 'ok',
        'service' : 'Creator OS Face Tracker',
        'model'   : 'loaded' if face_net else 'not loaded'
    })

# ── Detect face in a single frame ────────────────────────────────────────────
def detect_face(frame):
    """
    Returns (cx, cy) as normalised 0-1 floats, or None if no face found.
    cx = horizontal center, cy = vertical center of the best detected face.
    """
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)),
        1.0,
        (300, 300),
        (104.0, 177.0, 123.0)
    )
    face_net.setInput(blob)
    detections = face_net.forward()

    best_conf = 0
    best_box  = None

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence < 0.5:  # confidence threshold
            continue
        if confidence > best_conf:
            best_conf = confidence
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            best_box = box.astype(int)

    if best_box is None:
        return None

    x1, y1, x2, y2 = best_box
    cx = ((x1 + x2) / 2) / w
    cy = ((y1 + y2) / 2) / h
    # Bias slightly upward so full head is in frame
    cy = max(0.0, cy - 0.04)
    return { 'x': round(float(cx), 4), 'y': round(float(cy), 4) }


# ── Main face tracking endpoint ───────────────────────────────────────────────
@app.route('/track', methods=['POST'])
def track_faces():
    """
    Accepts a video file upload.
    Returns JSON array of face positions per frame:
    [
      { "frame": 0,  "time": 0.0,   "x": 0.48, "y": 0.42 },
      { "frame": 1,  "time": 0.033, "x": 0.49, "y": 0.41 },
      ...
    ]
    Frames where no face is detected return null for x/y.
    """
    if face_net is None:
        return jsonify({ 'error': 'Face detection model not loaded yet. Try again in a moment.' }), 503

    if 'video' not in request.files:
        return jsonify({ 'error': 'No video file provided. Send as multipart/form-data with key "video".' }), 400

    video_file = request.files['video']
    if not video_file.filename:
        return jsonify({ 'error': 'Empty filename.' }), 400

    # Check file size — max 200MB
    video_file.seek(0, 2)
    size_mb = video_file.tell() / 1048576
    video_file.seek(0)
    if size_mb > 200:
        return jsonify({ 'error': f'File too large ({size_mb:.0f}MB). Max is 200MB.' }), 413

    # Analyse every Nth frame for speed (sample_rate = 1 means every frame)
    sample_rate = int(request.form.get('sample_rate', 2))  # default: every 2nd frame
    sample_rate = max(1, min(sample_rate, 10))

    tmp_path = None
    try:
        # Save to temp file (OpenCV needs a file path)
        suffix = os.path.splitext(video_file.filename)[1] or '.mp4'
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            video_file.save(tmp)
            tmp_path = tmp.name

        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            return jsonify({ 'error': 'Could not open video file. Make sure it is a valid MP4 or WebM.' }), 422

        fps        = cap.get(cv2.CAP_PROP_FPS) or 30
        total_fr   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration   = total_fr / fps if fps > 0 else 0

        print(f'[FaceTracker] Video: {total_fr} frames @ {fps}fps, {duration:.1f}s, {size_mb:.1f}MB')

        results      = []
        frame_idx    = 0
        last_face    = None  # carry forward last known face position
        SMOOTH       = 0.25  # lerp smoothing

        smoothed_x   = None
        smoothed_y   = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            t = frame_idx / fps

            if frame_idx % sample_rate == 0:
                # Resize frame for faster detection
                small = cv2.resize(frame, (640, int(frame.shape[0] * 640 / frame.shape[1])))
                face  = detect_face(small)

                if face:
                    if smoothed_x is None:
                        smoothed_x = face['x']
                        smoothed_y = face['y']
                    else:
                        # Lerp smooth
                        smoothed_x += (face['x'] - smoothed_x) * SMOOTH
                        smoothed_y += (face['y'] - smoothed_y) * SMOOTH
                    last_face = { 'x': round(smoothed_x, 4), 'y': round(smoothed_y, 4) }
                else:
                    # Drift back toward center
                    if smoothed_x is not None:
                        smoothed_x += (0.5 - smoothed_x) * SMOOTH * 0.3
                        smoothed_y += (0.5 - smoothed_y) * SMOOTH * 0.3
                        last_face = { 'x': round(smoothed_x, 4), 'y': round(smoothed_y, 4) }

            results.append({
                'frame' : frame_idx,
                'time'  : round(t, 4),
                'x'     : last_face['x'] if last_face else None,
                'y'     : last_face['y'] if last_face else None,
            })

            frame_idx += 1

        cap.release()
        print(f'[FaceTracker] Processed {frame_idx} frames, {len([r for r in results if r["x"]])} with face detected')

        return jsonify({
            'frames'      : results,
            'fps'         : fps,
            'total_frames': frame_idx,
            'duration'    : round(duration, 2),
            'face_frames' : len([r for r in results if r['x'] is not None]),
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({ 'error': 'Processing failed: ' + str(e) }), 500

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


if __name__ == '__main__':
    load_model()
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
