import cv2
import numpy as np
from mtcnn import MTCNN

detector = MTCNN()
IMAGE_SIZE = (224,224)

def extract_frames(video_path, num_frames=8, target_size=IMAGE_SIZE):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        cap.release()
        return np.zeros((num_frames, *target_size, 3), dtype=np.uint8)
    
    idxs = np.linspace(0, total-1, num_frames, dtype=int)
    frames = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Crop mặt bằng MTCNN
        faces = detector.detect_faces(frame)
        if faces:
            best = max(faces, key=lambda d: d['confidence'])
            x,y,w,h = best['box']
            x,y = max(0,x), max(0,y)
            x2,y2 = min(frame.shape[1],x+w), min(frame.shape[0],y+h)
            face = frame[y:y2, x:x2] if (x2>x and y2>y) else frame
        else:
            face = frame

        face = cv2.resize(face, target_size)
        frames.append(face)

    cap.release()
    while len(frames) < num_frames:
        frames.append(frames[-1] if frames else np.zeros((*target_size,3),dtype=np.uint8))
    return np.array(frames, dtype=np.uint8)
