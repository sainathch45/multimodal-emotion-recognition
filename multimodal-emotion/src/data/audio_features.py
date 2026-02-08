import torch, librosa
import numpy as np

class AudioFeatureExtractor:
    def __init__(self, sample_rate=16000, n_mels=64, mfcc_dim=20):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.mfcc_dim = mfcc_dim

    def _load(self, path):
        y, sr = librosa.load(path, sr=self.sample_rate)
        if len(y) == 0:
            return np.zeros(self.sample_rate)  # 1s silence fallback
        return y

    def extract(self, path, out_dim=256):
        try:
            y = self._load(path)
            mel = librosa.feature.melspectrogram(y=y, sr=self.sample_rate, n_mels=self.n_mels)
            mel_db = librosa.power_to_db(mel, ref=np.max)
            mfcc = librosa.feature.mfcc(y=y, sr=self.sample_rate, n_mfcc=self.mfcc_dim)
            # aggregate
            mel_mean = mel_db.mean(axis=1)  # [n_mels]
            mfcc_mean = mfcc.mean(axis=1)  # [mfcc_dim]
            feat = np.concatenate([mel_mean, mfcc_mean], axis=0)  # length n_mels+mfcc_dim
        except Exception:
            feat = np.zeros(self.n_mels + self.mfcc_dim, dtype=np.float32)
        feat = feat.astype('float32')
        # project to out_dim deterministically via repeat/truncate
        if feat.shape[0] >= out_dim:
            return feat[:out_dim]
        else:
            reps = int(np.ceil(out_dim / feat.shape[0]))
            tiled = np.tile(feat, reps)[:out_dim]
            return tiled
