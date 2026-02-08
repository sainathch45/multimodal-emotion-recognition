from typing import Optional
import torch
from transformers import AutoTokenizer, AutoModel


class TextEmbedder:
    def __init__(self, model_name: str = "distilbert-base-uncased", device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode(self, text: str, out_dim: int = 312):
        if text is None:
            return torch.zeros(out_dim)
        enc = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        out = self.model(**enc)
        hidden = out.last_hidden_state  # [B, T, H]
        pooled = hidden.mean(dim=1).squeeze(0).detach().cpu()  # [H]
        # project/truncate/pad to desired dim
        h = pooled.shape[0]
        if h == out_dim:
            return pooled
        elif h > out_dim:
            return pooled[:out_dim]
        else:
            pad = torch.zeros(out_dim - h)
            return torch.cat([pooled, pad], dim=0)
