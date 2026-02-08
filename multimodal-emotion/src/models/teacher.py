import torch
import torch.nn as nn
from .encoders import TextEncoder, AudioEncoder, VideoEncoder, ReliabilityGate
from .fusion import MultiHeadCrossModal

class TeacherModel(nn.Module):
    def __init__(self, text_dim=312, audio_dim=256, video_dim=256, fuse_dim=512, num_classes=6, modality_dropout_p=0.2):
        """
        Multimodal Teacher Model for Emotion Recognition
        
        Args:
            text_dim: Text embedding dimension (default: 312 for hash-based)
            audio_dim: Audio feature dimension (default: 256 for MFCC)
            video_dim: Video feature dimension (default: 256 for basic CNN)
            fuse_dim: Fusion hidden dimension (default: 512)
            num_classes: Number of emotion classes (default: 6)
            modality_dropout_p: Probability of dropping modalities during training (default: 0.2)
        """
        super().__init__()
        self.text_dim, self.audio_dim, self.video_dim = text_dim, audio_dim, video_dim
        self.fuse_dim = fuse_dim
        self.modality_dropout_p = modality_dropout_p

        # Deep encoders with MUCH larger hidden dimensions (no bottleneck for pre-trained features)
        self.text = TextEncoder(text_dim, hidden_dim=512)
        self.audio = AudioEncoder(audio_dim, hidden_dim=768)
        self.video = VideoEncoder(video_dim, hidden_dim=768)
        self.txt_gate = ReliabilityGate(text_dim)
        self.aud_gate = ReliabilityGate(audio_dim)
        self.vid_gate = ReliabilityGate(video_dim)

        # Align to fusion dim
        self.align_text = nn.Sequential(
            nn.Linear(text_dim, fuse_dim),
            nn.LayerNorm(fuse_dim),
            nn.ReLU()
        )
        self.align_audio = nn.Sequential(
            nn.Linear(audio_dim, fuse_dim),
            nn.LayerNorm(fuse_dim),
            nn.ReLU()
        )
        self.align_video = nn.Sequential(
            nn.Linear(video_dim, fuse_dim),
            nn.LayerNorm(fuse_dim),
            nn.ReLU()
        )

        # Projection heads for contrastive learning
        self.proj_text = nn.Sequential(
            nn.Linear(fuse_dim, fuse_dim),
            nn.ReLU(),
            nn.Linear(fuse_dim, fuse_dim)
        )
        self.proj_audio = nn.Sequential(
            nn.Linear(fuse_dim, fuse_dim),
            nn.ReLU(),
            nn.Linear(fuse_dim, fuse_dim),
            nn.LayerNorm(fuse_dim)
        )
        self.proj_video = nn.Sequential(
            nn.Linear(fuse_dim, fuse_dim),
            nn.LayerNorm(fuse_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(fuse_dim, fuse_dim),
            nn.LayerNorm(fuse_dim),
            nn.GELU(),
            nn.Linear(fuse_dim, fuse_dim),
            nn.LayerNorm(fuse_dim)
        )

        # Cross-modal fusion
        self.fusion = MultiHeadCrossModal(embed_dim=fuse_dim, num_heads=8, num_layers=2, feedforward_dim=fuse_dim*4)

        # Reconstruction heads (from fused back to modality space)
        self.recon_text = nn.Sequential(
            nn.Linear(fuse_dim, fuse_dim // 2),
            nn.ReLU(),
            nn.Linear(fuse_dim // 2, text_dim)
        )
        self.recon_audio = nn.Sequential(
            nn.Linear(fuse_dim, fuse_dim // 2),
            nn.ReLU(),
            nn.Linear(fuse_dim // 2, audio_dim)
        )
        self.recon_video = nn.Sequential(
            nn.Linear(fuse_dim, fuse_dim // 2),
            nn.ReLU(),
            nn.Linear(fuse_dim // 2, video_dim)
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(fuse_dim, fuse_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(fuse_dim // 2, num_classes)
        )

    def _modality_dropout(self, mt, ma, mv):
        # Randomly drop modalities during training; ensure at least one stays
        if not self.training or self.modality_dropout_p <= 0.0:
            return mt, ma, mv, None
        B = mt.size(0)
        device = mt.device
        keep = (torch.rand(B, 3, device=device) > self.modality_dropout_p).float()
        # Ensure at least one modality
        none_kept = (keep.sum(dim=1) == 0)
        if none_kept.any():
            # randomly enable one modality for those rows
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
        # sample: dict with 'text','audio','video' -> tensors [B, dim]
        t = self.text(sample['text'])
        a = self.audio(sample['audio'])
        v = self.video(sample['video'])
        # reliability weights
        wt = self.txt_gate(t)
        wa = self.aud_gate(a)
        wv = self.vid_gate(v)
        # apply weights (broadcast)
        t = t * wt
        a = a * wa
        v = v * wv
        # align dims
        mt = self.align_text(t)
        ma = self.align_audio(a)
        mv = self.align_video(v)
        # modality dropout during training
        mt, ma, mv, keep = self._modality_dropout(mt, ma, mv)
        fused = self.fusion([mt, ma, mv])
        logits = self.classifier(fused)
        # projections and reconstructions
        proj = {
            'text': self.proj_text(mt),
            'audio': self.proj_audio(ma),
            'video': self.proj_video(mv)
        }
        recon = {
            'text': self.recon_text(fused),
            'audio': self.recon_audio(fused),
            'video': self.recon_video(fused)
        }
        return {
            'logits': logits,
            'fused': fused,
            'weights': {'text': wt, 'audio': wa, 'video': wv},
            'proj': proj,
            'recon': recon,
            'mask': keep
        }
