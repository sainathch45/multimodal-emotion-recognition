cd C:\Vision\Education\college\stuff\projects\hons_project\multimodal-emotion
python -m src.train.train_emotion_model --data data/processed/combined_finetuning --batch_size 8 --accumulation_steps 4 --epochs 40 --freeze_epochs 0 --lr 2e-5 --lr_head 1e-4 --patience 15 --out experiments/emotion_pretrained_sota --save_every 5 --resume
