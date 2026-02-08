import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from ..models.teacher import TeacherModel
from ..models.student import StudentModel
from ..losses.distillation import DistillationLoss
from ..data.dataset import MultimodalEmotionDataset


def get_toy_loader(path, batch_size=8):
    ds = MultimodalEmotionDataset(path)
    return DataLoader(ds, batch_size=batch_size, shuffle=True)


def train_step_teacher(model, batch, optimizer, device):
    model.train()
    sample = {
        'text': batch['text_emb'].to(device),
        'audio': batch['audio_emb'].to(device),
        'video': batch['video_emb'].to(device)
    }
    out = model(sample)
    loss = nn.CrossEntropyLoss()(out['logits'], batch['label'].to(device))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def train_step_student(student, teacher, batch, optimizer, distill_loss, device):
    student.train(); teacher.eval()
    sample = {
        'text': batch['text_emb'].to(device),
        'audio': batch['audio_emb'].to(device),
        'video': batch['video_emb'].to(device)
    }
    with torch.no_grad():
        t_out = teacher(sample)
    s_out = student(sample)
    loss, parts = distill_loss(s_out['logits'], t_out['logits'], s_out['fused'], t_out['fused'], batch['label'].to(device))
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    parts['total'] = loss.item()
    return parts


def run_toy(path, epochs=2, device=None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    loader = get_toy_loader(path)
    teacher = TeacherModel().to(device)
    student = StudentModel().to(device)
    t_opt = torch.optim.AdamW(teacher.parameters(), lr=1e-3)
    s_opt = torch.optim.AdamW(student.parameters(), lr=1e-3)
    distill = DistillationLoss()
    for ep in range(epochs):
        for i, batch in enumerate(loader):
            t_loss = train_step_teacher(teacher, batch, t_opt, device)
            s_parts = train_step_student(student, teacher, batch, s_opt, distill, device)
            if i % 10 == 0:
                print(f"Epoch {ep} iter {i} teacher_ce={t_loss:.4f} student_total={s_parts['total']:.4f} kd={s_parts['kd']:.4f}")
    print('Finished toy run.')

if __name__ == '__main__':
    # Expect preprocessed .npz in data/processed/mosei
    run_toy('data/processed/mosei')
