# Experiment Ledger - Guardrails Research

## Purpose
Track all experiments (successes and failures) for learning and iteration.

---

## 2026-03-21

### Experiment 1: Deterministic Filter Only
- **Time**: 10:09
- **Status**: ✅ Completed
- **Config**: Regex patterns for injection detection
- **Results**:
  - Accuracy: 0.507
  - F1: 0.124
  - Precision: 0.559
  - Recall: 0.070
  - Latency: <1ms/sample
- **Learning**:
  - Deterministic filters catch obvious patterns but miss semantic attacks
  - Very fast but low recall (7%)
  - Good for Tier 2 fast path, not sufficient alone

### Experiment 2: Neural Pre-Screen (Qwen2.5-0.5B-Instruct)
- **Time**: 10:16
- **Status**: ✅ Completed
- **Config**: Prompt-based yes/no classification, threshold=0.5
- **Results**:
  - Accuracy: 0.506
  - F1: 0.670
  - Precision: 0.507
  - Recall: 0.988
  - Latency: ~770ms/sample
- **Learning**:
  - Neural pre-screen over-predicts injection (threshold too low)
  - Almost perfect recall but many false positives
  - Need to calibrate threshold higher (try 0.7, 0.8)
  - Latency too high for production (<60ms target)

### Experiment 3: Combined Deterministic + Neural
- **Time**: 10:32
- **Status**: ✅ Completed
- **Config**: Deterministic first, then neural if not blocked
- **Results**:
  - Same as neural alone (F1=0.67, Recall=0.99)
  - Latency: ~662ms/sample
- **Learning**:
  - Neural dominates because it classifies almost everything as injection
  - No benefit from combination with current threshold
  - Need to fix threshold calibration first

### Experiment 4: LoRA Adapter Training (CPU)
- **Time**: 10:51
- **Status**: ❌ Failed (timeout)
- **Config**: 35K samples, Qwen2.5-0.5B, CPU only
- **Learning**:
  - CPU training too slow for practical iteration
  - Need GPU for efficient training

### Experiment 5: LoRA Adapter Training (GPU)
- **Time**: 12:35
- **Status**: ✅ Completed
- **Config**:
  - Model: Qwen2.5-0.5B-Instruct
  - 4-bit quantization
  - LoRA rank=16, alpha=32
  - Train: 35K samples, Val: 4.7K, Test: 7.1K
  - Batch size: 8, LR: 1e-3
  - Epochs: 3
- **Device**: NVIDIA RTX 4060 Laptop (8GB VRAM)
- **Speed**: ~6.8 it/s
- **Results**:
  - Test Accuracy: 0.937
  - Test F1: 0.938 ✅ (Target: >=0.90)
  - Test Precision: 0.922
  - Test Recall: 0.955
  - Best Val F1: 0.936
- **Learning**:
  - LoRA fine-tuning dramatically improves F1 (0.67 → 0.938)
  - Balanced precision/recall trade-off
  - Saved adapters for all 3 epochs in data/adapters/prompt_injection/
  - Ready for combined pipeline testing

### Experiment 6: Combined Deterministic + LoRA Pipeline
- **Time**: ~15:00
- **Status**: ✅ Completed
- **Config**:
  - Deterministic filter: Regex + keyword patterns
  - LoRA adapter: epoch_3 (best validation F1)
  - Pipeline: Deterministic first, then LoRA for non-blocked
- **Results**:
  - Test Accuracy: 0.915
  - Test F1: 0.918 ✅
  - Test Precision: 0.882
  - Test Recall: 0.958
  - Deterministic blocks: 6.2% (444/7137)
  - LoRA blocks: 48.1% (3434/7137)
- **Latency**:
  - Deterministic: p50=0.1ms, p95=0.4ms
  - LoRA: p50=127.7ms, p95=152.5ms
  - Combined: p50=119.9ms, p95=143.4ms
- **Learning**:
  - Combined pipeline maintains high F1 (0.918)
  - Deterministic provides fast early-exit for obvious cases
  - Latency dominated by LoRA (GPU inference on 0.5B model)
  - Target p95 < 200ms achieved ✓

### Experiment 7: Sample Efficiency
- **Time**: ~16:30
- **Status**: ✅ Completed
- **Question**: How few samples for F1 >= 0.87?
- **Results**:
  | Samples | F1    | Precision | Recall  | Time    |
  |---------|-------|-----------|--------|--------|
  | 50      | 0.25  | 0.95      | 0.14   | 3.4s   |
  | 100     | 0.78  | 0.72      | 0.86   | 5.7s   |
  | 200     | 0.80  | 0.78      | 0.83   | 11.9s  |
  | 500     | 0.84  | 0.83      | 0.85   | 31.8s  |
  | 1000    | 0.80  | 0.67      | 0.98   | 65.8s  |
  | 2000    | 0.88  | 0.95      | 0.81   | 134.4s  |
- **Learning**:
  - **2000 samples needed for F1 >= 0.87**
  - High variance with few samples (50-500)
  - 1000 samples showed overfitting (high recall, Low Precision)
  - Need few-Shot/Meta-Learning Techniques

### Experiment 7: Sample Efficiency (Adaptation Spectrum)
- **Time**: ~16:00
- **Status**: Completed
- **Config**:
  - Tested sample sizes: 50, 100, 200, 500, 1000, 2000
  - 3 epochs per training run
  - Same model/architecture as full training
- **Results**:
  | Samples | F1     | Time   | Target   |
  |---------|--------|--------|----------|
  | 50      | 0.25   | 3.4s   | FAIL     |
  | 100     | 0.78   | 5.7s   | FAIL     |
  | 200     | 0.80   | 11.9s  | FAIL     |
  | 500     | 0.84   | 31.8s  | FAIL     |
  | 1000    | 0.80   | 65.8s  | FAIL     |
  | 2000    | 0.88   | 134.4s | PASS     |
- **Learning**:
  - Minimum ~2000 samples needed for F1 >= 0.87
  - 1000 samples showed overfitting (high recall, low precision)
  - Need better few-shot techniques or adversarial data augmentation

### Experiment 8: Regularized Sample Efficiency
- **Time**: ~18:00
- **Status**: Completed
- **Question**: Can regularization fix the overfitting at 1000 samples?
- **Config**:
  - Dropout: 0.15
  - Weight decay: 0.05
  - Early stopping patience: 3
  - Max epochs: 10
- **Results**:
  | Samples | Val F1 | Test F1 | Precision | Recall | Epochs | Target |
  |---------|--------|---------|-----------|--------|--------|--------|
  | 100     | 0.7923 | 0.7747  | 0.6707    | 0.9168 | 10     | FAIL   |
  | 200     | 0.7896 | 0.7920  | 0.7906    | 0.7934 | 9      | FAIL   |
  | 300     | 0.8394 | 0.8247  | 0.8051    | 0.8453 | 9      | FAIL   |
  | 400     | 0.8462 | 0.8536  | 0.8401    | 0.8674 | 10     | FAIL   |
  | 500     | 0.8658 | 0.8537  | 0.7881    | 0.9311 | 10     | FAIL   |
  | 750     | 0.8749 | 0.8579  | 0.8080    | 0.9142 | 8      | FAIL   |
  | 1000    | 0.9019 | 0.8919  | 0.8819    | 0.9022 | 10     | PASS   |
- **Learning**:
  - **REGULARIZATION FIXED THE OVERFITTING!**
  - Baseline: 2000 samples needed for F1 >= 0.87
  - With regularization: 1000 samples achieves F1 = 0.8919
  - **50% reduction in samples** (2000 -> 1000)
  - Early stopping prevented overfitting (stopped at epoch 8 for 750 samples)
  - Still need synthetic augmentation to reach <500 samples

---

## Key Insights So Far

1. **Threshold calibration is critical** - Neural pre-screen needs higher threshold
2. **Deterministic alone insufficient** - Only catches ~7% of attacks
3. **GPU essential for training** - CPU too slow
4. **LoRA adapters should improve precision** - Training on actual task data

---

## Next Experiments to Try

**Goal: MAXIMIZE F1 score, not just >= 0.87** (Current best: 0.938 with full training)

1. [x] ~~Tune neural threshold to 0.7-0.8~~ (Bypassed - LoRA training solved the calibration issue)
2. [x] Complete LoRA adapter training (F1=0.938)
3. [x] Test LoRA adapter on test set (F1=0.938)
4. [x] Combine deterministic + calibrated LoRA (F1=0.918)
5. [x] Benchmark latency with LoRA (p50=119.9ms, p95=143.4ms)
6. [x] ~~Text-to-LoRA~~ (NOT RECOMMENDED - research shows impractical)
7. [x] **Sample efficiency baseline: 2000 samples for F1 >= 0.87**
8. [x] **Regularization fix: 1000 samples for F1 >= 0.87** (50% reduction!)
9. [x] **Synthetic data generation** (6500 samples, 8 attack categories)
10. [ ] **Synthetic + real data combination** (push toward 0.95+ F1)
11. [ ] **MeTA-LoRA with sub-task decomposition** (8-15 attack categories)
12. [ ] **Domain continued pre-training** (highest ROI for quality)
13. [ ] **Adversarial robustness testing** (maintain high F1 on attacks)

---

## Research Findings (2026-03-21)

### Key Finding: Sample efficiency bottleneck is NOT fundamental
The 2000-sample requirement is a compound effect of:
1. **Suboptimal LoRA initialization** → Fix with LoRA-GA (2-4x faster convergence)
2. **Only targeting Q/V layers** → Fix with all 7 linear layers
3. **No weight decomposition** → Fix with DoRA (+1-4% quality)
4. **Uniform learning rates** → Fix with LoRA+ (16x LR for B matrices)

### Recommended Stack (compounds to 4-6x sample reduction)
| Technique | Impact | Effort | Priority |
|-----------|--------|--------|----------|
| LoRA-GA initialization | 2-4x convergence | Low | Do first |
| DoRA (use_dora=True) | +1-4% quality | Low | Do first |
| LoRA+ (16x LR for B) | +1-2% accuracy | Low | Do first |
| All 7 linear layers | Better expressiveness | Low | Do first |
| Domain pre-training | Highest ROI | Medium | High |
| LLM synthetic augmentation | +3-26% F1 | Medium | High |

### Approaches NOT Recommended
- **Text-to-LoRA**: No Qwen2.5-0.5B checkpoint, needs 5 days H100, 7.6% gap
- **ReptiLoRA**: Modest gains (+0.6-0.9 ROUGE-L)

### Approaches Worth Reconsidering
- **MeTA-LoRA**: May work if we decompose "prompt injection" into sub-tasks:
  - Instruction override detection
  - Role-play manipulation detection
  - Encoding/obfuscation attack detection
  - Context window stuffing detection
  - Emotional manipulation detection
  - Authority invocation detection
  - Hypothetical bypass detection
  - Delimiter confusion detection

  With 8-15 such sub-categories, MeTA-LoRA's episodic training could learn shared representations across attack types and adapt to novel attack patterns with 50-100 examples. Key question: Do we have enough task diversity in existing labeled data?

### MeTA-LoRA Implementation Plan (if pursued)
1. **Data Preparation**: Label existing data by attack sub-category (8 categories)
2. **Episode Construction**: Create N-way K-shot episodes from attack families
3. **Meta-Training**: Inner-loop task-specific adaptation, outer-loop meta-updates
4. **Evaluation**: Test few-shot adaptation to new attack patterns
5. **Expected Outcome**: 50-100 examples per new attack type for F1 >= 0.87

---

## 2026-03-22

### Experiment 9: Regularized LoRA (Full Training)
- **Time**: ~12:45
- **Status**: ✅ Completed
- **Config**:
  - Model: Qwen2.5-0.5B-Instruct
  - DoRA (Weight Decomposition): True
  - LoRA+ (16x LR for B matrices): True
  - LoRA dropout: 0.15
  - Classifier dropout: 0.2
  - Weight decay: 0.05
  - Label smoothing: 0.1
  - Gradient clipping: 1.0
  - All 7 linear layers
  - Rank: 16, Alpha: 32
- **Training Progress**:
  | Epoch | Val F1 | Precision | Recall | Notes |
  |-------|--------|-----------|--------|-------|
  | 1 | 0.9085 | 0.8741 | 0.9458 | Good start |
  | 2 | 0.7446 | 0.9895 | 0.5969 | Over-regularized (extreme precision) |
  | 3 | 0.9142 | 0.9539 | 0.8777 | Recovery |
  | 4 | 0.8776 | 0.9788 | 0.7953 | High precision, lower recall |
  | 5 | 0.9194 | 0.9326 | 0.9067 | **Best balanced** |
- **Results**:
  - Test Accuracy: 0.9200
  - Test F1: **0.9191**
  - Test Precision: 0.9295
  - Test Recall: 0.9089
- **Learning**:
  - Full regularization stack achieves F1=0.9191
  - Epoch 2 showed extreme precision (0.99) but low recall (0.60)
  - Best model at epoch 5 with balanced precision/recall
  - Still ~3% gap to target of F1 >= 0.95

### Experiment 10: ARENA (L1 Regularized LoRA) - FAILED
- **Time**: ~23:20 (previous day)
- **Status**: ❌ Failed
- **Config**:
  - L1 weight: 0.01
  - Dynamic rank adjustment
  - DoRA: True
- **Results**:
  - Test F1: **0.026** (catastrophic failure)
  - Precision: 1.0
  - Recall: 0.013
- **Learning**:
  - L1=0.01 was too aggressive - killed almost all LoRA weights
  - Model collapsed to predicting everything as one class
  - Need much lower L1 weight (try 0.001 or 0.0001)

---

## TODO: Regularization Variations

See `data/TODO_REGULARIZATION.md` for:
1. Low dropout (0.05)
2. No label smoothing
3. Low weight decay (0.01)
4. Lower LoRA+ ratio (8x)

---

## Failure Analysis Template

When an experiment fails, document:
1. What was the hypothesis?
2. What went wrong?
3. What did we learn?
4. What to try next?
