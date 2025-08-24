import cv2
import numpy as np
import insightface

IMAGE_SIZE = (224, 224)

detector = insightface.app.FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
detector.prepare(ctx_id=0, det_size=(640, 640))

def extract_frames(video_path, num_frames=8, target_size=IMAGE_SIZE):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        cap.release()
        return np.zeros((num_frames, *target_size, 3), dtype=np.uint8)

    idxs = np.linspace(0, total - 1, num_frames, dtype=int)
    frames = []

    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        faces = detector.get(frame)
        if len(faces) > 0:
            best = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
            x1, y1, x2, y2 = best.bbox.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            face = frame[y1:y2, x1:x2] if (x2 > x1 and y2 > y1) else frame
        else:
            face = frame

        face = cv2.resize(face, target_size)
        frames.append(face)

    cap.release()

    while len(frames) < num_frames:
        frames.append(frames[-1] if frames else np.zeros((*target_size, 3), dtype=np.uint8))

    return np.array(frames, dtype=np.uint8)
