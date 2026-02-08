import torch
import torch.nn as nn
from .encoders import TextEncoder, AudioEncoder, VideoEncoder
from .fusion import LinearFusion

class StudentModel(nn.Module):
    def __init__(self, text_dim=312, audio_dim=256, video_dim=256, fuse_dim=384, num_classes=6, modality_dropout_p=0.2):
        super().__init__()
        self.text_dim, self.audio_dim, self.video_dim = text_dim, audio_dim, video_dim
        self.fuse_dim = fuse_dim
        self.modality_dropout_p = modality_dropout_p

        self.text = TextEncoder(text_dim)
        self.audio = AudioEncoder(audio_dim)
        self.video = VideoEncoder(video_dim)
        self.align_text = nn.Linear(text_dim, fuse_dim)
        self.align_audio = nn.Linear(audio_dim, fuse_dim)
        self.align_video = nn.Linear(video_dim, fuse_dim)
        # Projection heads (for contrastive learning)
        self.proj_text = nn.Sequential(nn.Linear(fuse_dim, fuse_dim), nn.LayerNorm(fuse_dim))
        self.proj_audio = nn.Sequential(nn.Linear(fuse_dim, fuse_dim), nn.LayerNorm(fuse_dim))
        self.proj_video = nn.Sequential(nn.Linear(fuse_dim, fuse_dim), nn.LayerNorm(fuse_dim))
        # Fusion and classifier
        self.fusion = LinearFusion(embed_dim=fuse_dim)
        self.classifier = nn.Linear(fuse_dim, num_classes)
        # Reconstruction heads
        self.recon_text = nn.Linear(fuse_dim, text_dim)
        self.recon_audio = nn.Linear(fuse_dim, audio_dim)
        self.recon_video = nn.Linear(fuse_dim, video_dim)

    def _modality_dropout(self, mt, ma, mv):
        if not self.training or self.modality_dropout_p <= 0.0:
            return mt, ma, mv, None
        B = mt.size(0)
        device = mt.device
        keep = (torch.rand(B, 3, device=device) > self.modality_dropout_p).float()
        none_kept = (keep.sum(dim=1) == 0)
        if none_kept.any():
            idx = torch.randint(low=0, high=3, size=(int(none_kept.sum().item()),), device=device)
            keep[none_kept, idx] = 1.0
        mask_t = keep[:, 0].view(B, 1)
        mask_a = keep[:, 1].view(B, 1)
        mask_v = keep[:, 2].view(B, 1)
        mt = mt * mask_t
        ma = ma * mask_a
        mv = mv * mask_v
        return mt, ma, mv, keep

    def forward(self, sample):
        t = self.align_text(self.text(sample['text']))
        a = self.align_audio(self.audio(sample['audio']))
        v = self.align_video(self.video(sample['video']))
        t, a, v, keep = self._modality_dropout(t, a, v)
        fused = self.fusion([t, a, v])
        logits = self.classifier(fused)
        proj = {'text': self.proj_text(t), 'audio': self.proj_audio(a), 'video': self.proj_video(v)}
        recon = {'text': self.recon_text(fused), 'audio': self.recon_audio(fused), 'video': self.recon_video(fused)}
        return {'logits': logits, 'fused': fused, 'proj': proj, 'recon': recon, 'mask': keep}
