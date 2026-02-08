# 🚀 Launch Optimized SOTA Training
# This script starts the optimized training with encoder freezing for 10x speedup

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "🚀 LAUNCHING OPTIMIZED SOTA TRAINING" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Strategy:" -ForegroundColor Yellow
Write-Host "  - Phase 1: Freeze encoders (10 epochs) → ~10 min/epoch ⚡" -ForegroundColor Green
Write-Host "  - Phase 2: Unfreeze all (40 epochs) → ~3 hours/epoch" -ForegroundColor Green
Write-Host "  - Expected total time: ~40-50 hours" -ForegroundColor Yellow
Write-Host "  - Target F1: 75-85% (SOTA)" -ForegroundColor Green
Write-Host ""
Write-Host "Dataset: 10,253 samples (RAVDESS + CREMA-D + IEMOCAP)" -ForegroundColor Cyan
Write-Host "Output: experiments/finetuned_sota_optimized/" -ForegroundColor Cyan
Write-Host ""
Write-Host "Progress will be saved every 5 epochs + best model automatically saved!" -ForegroundColor Green
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host ""

# Confirm
$response = Read-Host "Ready to start? (y/n)"

if ($response -ne 'y') {
    Write-Host "Cancelled." -ForegroundColor Red
    exit
}

Write-Host ""
Write-Host "Starting training..." -ForegroundColor Green
Write-Host ""

# Navigate to correct directory
Set-Location C:\Vision\Education\college\stuff\projects\hons_project\multimodal-emotion

# Start training
python -m src.train.train_finetuned_optimized `
  --data data/processed/combined_finetuning `
  --batch_size 10 `
  --accumulation_steps 4 `
  --epochs 50 `
  --freeze_epochs 10 `
  --lr 2e-5 `
  --lr_head 2e-4 `
  --patience 15 `
  --out experiments/finetuned_sota_optimized `
  --save_every 5

Write-Host ""
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "Training completed or stopped!" -ForegroundColor Yellow
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Check results:" -ForegroundColor Cyan
Write-Host "  Best model: experiments/finetuned_sota_optimized/best_model.pt" -ForegroundColor Green
Write-Host "  Results: experiments/finetuned_sota_optimized/final_results.json" -ForegroundColor Green
Write-Host ""
