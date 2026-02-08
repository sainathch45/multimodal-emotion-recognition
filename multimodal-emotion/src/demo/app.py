import streamlit as st
import numpy as np
import torch
import sys
from pathlib import Path

# Add project root to path for absolute imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.teacher import TeacherModel
from src.models.student import StudentModel
from src.data.text_embed import TextEmbedder
from src.data.audio_features import AudioFeatureExtractor
from src.data.video_features import VideoFeatureExtractor


def softmax(x):
    x = torch.tensor(x)
    return torch.softmax(x, dim=-1).cpu().numpy()

@st.cache_resource
def load_model(ckpt_path: str):
    device = 'cpu'
    ckpt = torch.load(ckpt_path, map_location=device)
    if 'teacher' in Path(ckpt_path).name:
        model = TeacherModel()
    else:
        model = StudentModel()
    model.load_state_dict(ckpt['model'])
    model.eval(); model.to(device)
    return model

@st.cache_resource
def get_text_embedder():
    return TextEmbedder(device='cpu')

@st.cache_resource
def get_audio_extractor():
    return AudioFeatureExtractor()

@st.cache_resource
def get_video_extractor():
    return VideoFeatureExtractor()


def main():
    st.title("Multimodal Emotion Demo")
    
    # Find all available checkpoints
    experiments_dir = project_root / "experiments"
    checkpoint_files = []
    if experiments_dir.exists():
        for exp_dir in experiments_dir.iterdir():
            if exp_dir.is_dir():
                ckpt_file = exp_dir / "teacher_best.pt"
                if ckpt_file.exists():
                    # Try to load history to get performance metrics
                    history_file = exp_dir / "history.json"
                    metric_info = ""
                    if history_file.exists():
                        import json
                        try:
                            with open(history_file) as f:
                                history = json.load(f)
                            best_epoch = max(history, key=lambda x: x['val']['macro_f1'])
                            val_f1 = best_epoch['val']['macro_f1']
                            val_acc = best_epoch['val']['accuracy']
                            metric_info = f" (F1: {val_f1:.1%}, Acc: {val_acc:.1%})"
                        except:
                            pass
                    display_name = f"{exp_dir.name}{metric_info}"
                    checkpoint_files.append((display_name, str(ckpt_file)))
    
    if not checkpoint_files:
        st.sidebar.error("No checkpoints found in experiments/")
        st.sidebar.info("Train a model first using: python -m src.train.train_teacher")
        return
    
    # Dropdown selector
    selected_display = st.sidebar.selectbox(
        "Select Model Checkpoint",
        options=[name for name, _ in checkpoint_files],
        index=0
    )
    
    # Get actual path from selection
    ckpt_path = next(path for name, path in checkpoint_files if name == selected_display)
    
    model = None
    try:
        model = load_model(ckpt_path)
        st.sidebar.success(f"✓ Model loaded: {Path(ckpt_path).parent.name}")
    except Exception as e:
        st.sidebar.error(f"Failed to load model: {e}")

    text = st.text_area("Text", "Type something emotional...")
    audio_file = st.file_uploader("Audio file (.wav)", type=['wav'])
    video_file = st.file_uploader("Video file (.mp4)", type=['mp4'])

    if st.button("Predict"):
        if model is None:
            st.error("Load a valid checkpoint first")
            return
        txt = get_text_embedder(); afe = get_audio_extractor(); vfe = get_video_extractor()
        text_emb = txt.encode(text or "").cpu().numpy().astype('float32')
        if audio_file is not None:
            tmpa = Path("/tmp/audio_input.wav")
            with open(tmpa, 'wb') as f:
                f.write(audio_file.getbuffer())
            audio_emb = afe.extract(str(tmpa), out_dim=256)
        else:
            audio_emb = np.zeros(256, dtype='float32')
        if video_file is not None:
            tmpv = Path("/tmp/video_input.mp4")
            with open(tmpv, 'wb') as f:
                f.write(video_file.getbuffer())
            video_emb = vfe.extract(str(tmpv), out_dim=256)
        else:
            video_emb = np.zeros(256, dtype='float32')

        sample = {
            'text': torch.from_numpy(text_emb).unsqueeze(0),
            'audio': torch.from_numpy(audio_emb).unsqueeze(0),
            'video': torch.from_numpy(video_emb).unsqueeze(0),
        }
        with torch.no_grad():
            out = model(sample)
            logits = out['logits'].squeeze(0).cpu().numpy()
            probs = softmax(logits)
        st.write("Predicted probabilities:", probs)
        if 'weights' in out:
            w = {k: float(v.mean().item()) for k, v in out['weights'].items()}
            st.write("Reliability weights:", w)

if __name__ == '__main__':
    main()
