import cv2
import numpy as np

class VideoFeatureExtractor:
    def __init__(self, frame_samples: int = 16):
        self.frame_samples = frame_samples

    def _frame_hist(self, frame_bgr):
        # convert to HSV
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        # histogram bins: H=16, S=4, V=4 -> 256 dims
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [16, 4, 4], [0, 180, 0, 256, 0, 256])
        hist = cv2.normalize(hist, None).flatten()
        # ensure length 256
        if hist.shape[0] != 256:
            hist = cv2.resize(hist.reshape(-1, 1), (1, 256), interpolation=cv2.INTER_AREA).flatten()
        return hist.astype('float32')

    def extract(self, path, out_dim=256):
        try:
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                raise RuntimeError("Video open failed")
            length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            idxs = np.linspace(0, max(length - 1, 0), num=self.frame_samples, dtype=int)
            feats = []
            for idx in idxs:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue
                feats.append(self._frame_hist(frame))
            cap.release()
            if not feats:
                raise RuntimeError("No frames read")
            feat = np.stack(feats, axis=0).mean(axis=0)
        except Exception:
            feat = np.zeros(out_dim, dtype=np.float32)
        # ensure exact out_dim
        if feat.shape[0] > out_dim:
            feat = feat[:out_dim]
        elif feat.shape[0] < out_dim:
            feat = np.pad(feat, (0, out_dim - feat.shape[0]))
        return feat.astype('float32')
