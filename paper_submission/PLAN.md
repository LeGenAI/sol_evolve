이제 전체 내용을 정리하겠습니다.

---

# WBC Synthesis 실험 리뷰 및 계획

## 1. 실험 개요

**목표**: Peripheral Blood Cell(PBC) 데이터셋에서 SDXL + 클래스별 LoRA로 WBC 합성 이미지를 생성하고, 합성 데이터가 CNN 분류기의 견고성(robustness)을 향상시키는지 검증한다.

**평가 지표**: Test macro-F1 (primary), per-class F1, corruption robustness (5종 × 3 severity)

---

## 2. 완료된 작업

### 2.1 데이터 준비 (`01_prepare_data.py`) ✅

| 클래스 | 전체 | Train (70%) | Sharp Train | Val (15%) | Test (15%) |
|---|---|---|---|---|---|
| basophil | 1,218 | 852 | **596** | 183 | 183 |
| eosinophil | 3,117 | 2,181 | **1,527** | 468 | 468 |
| erythroblast | 1,551 | 1,085 | **759** | 233 | 233 |
| ig | 2,895 | 2,025 | **1,417** | 435 | 435 |
| lymphocyte | 1,214 | 849 | **594** | 182 | 183 |
| monocyte | 1,420 | 994 | **696** | 213 | 213 |
| neutrophil | 3,329 | 2,329 | **1,630** | 500 | 500 |
| platelet | 2,348 | 1,642 | **1,149** | 353 | 353 |
| **합계** | **17,092** | **11,957** | **8,368** | **2,567** | **2,568** |

- Stratified 70/15/15 split, seed=42
- Laplacian variance 상위 70% 이미지만 `train_sharp/`에 보존 → LoRA 학습 소스

### 2.2 Baseline CNN 학습 (`05_train_cnn.py --mode real_only`) ✅

**모델**: EfficientNet-B0 (ImageNet pretrained) | **디바이스**: MPS

| 지표 | 값 |
|---|---|
| Best Val macro-F1 | 0.9880 |
| **Test macro-F1** | **0.9884** |
| Test Accuracy | 0.9875 |
| Best Epoch | 25 |

**Per-class 결과**:

| 클래스 | F1 | 비고 |
|---|---|---|
| platelet | 1.0000 | 완벽 분류 |
| eosinophil | 0.9989 | |
| erythroblast | 0.9978 | |
| basophil | 0.9973 | |
| lymphocyte | 0.9864 | |
| neutrophil | 0.9780 | |
| monocyte | 0.9766 | |
| **ig** | **0.9724** | ← 최저 (immature granulocyte, 형태 다양) |

→ `models/baseline_cnn.pt` 저장 완료 (16MB)

### 2.3 환경 셋업 ✅

| 항목 | 내용 |
|---|---|
| 모델 | `stabilityai/stable-diffusion-xl-base-1.0` (캐시됨, 13GB) |
| 프레임워크 | diffusers 0.36.0 + peft 0.17.1 + accelerate 1.10.1 |
| 디바이스 | Apple Silicon MPS (PyTorch MPS backend) |
| LoRA 스크립트 | `train_dreambooth_lora_sdxl.py` (diffusers v0.36.0) |

### 2.4 해결한 주요 기술 이슈

| 이슈 | 원인 | 해결책 |
|---|---|---|
| `GradScaler("mps")` TypeError | MPS float64 미지원 | CUDA에서만 scaler 사용 |
| `accelerate` FileNotFoundError | PATH 미등록 | 바이너리 경로 직접 탐색 |
| accelerate empty-argv 버그 | macOS subprocess argv 오염 | accelerate 런처 제거, 직접 python3 실행 |
| fp16 backward RuntimeError | SDXL UNet MPS fp16 미지원 | fp32 (`--mixed_precision no`) 사용 |
| Accelerate fp16 MPS 거부 | accelerator.py 디바이스 체크 | `accelerator.py`, `modeling.py` 패치 |
| subprocess 빈 문자열 argv | bash `\` 연속 + subprocess list | **Shell script 파일 생성 후 `/bin/bash` 실행** |

### 2.5 LoRA 학습 파이프라인 구축 ✅

**최적 하이퍼파라미터 (MPS용)**:

```
resolution:            256px  (4× 속도 향상 vs 512px)
train_batch_size:      1
gradient_accumulation: 2  (effective batch = 2)
learning_rate:         5e-5
lr_warmup_steps:       10
max_train_steps:       100
rank:                  8
mixed_precision:       no (fp32)
gradient_checkpointing: False
train_text_encoder:    False (UNet LoRA만)
```

**속도 벤치마크**:

| 설정 | 초/step | 100 steps 예상 |
|---|---|---|
| 512px + grad_ckpt + text_enc (초기) | ~856s | ~23.8h |
| 512px + no text_enc + no grad_ckpt | ~408s | ~11.3h |
| **256px + no text_enc + no grad_ckpt** | **~400s** | **~11.1h** |

> 256px에서도 step당 속도가 크게 향상되지 않은 이유: SDXL UNet의 bottleneck은 attention 연산(MPS 미최적화)이며, 해상도보다 모델 크기가 지배적

### 2.6 현재 진행 중 🔄

```
basophil LoRA: step 25/100, loss=0.078, LR=4.67e-5 (warmup 완료 임박)
PID 90054, nohup으로 백그라운드 실행 중
run_all_lora.sh → 8클래스 순차 자동 실행
```

---

## 3. 앞으로의 실험 계획

### Phase 1: LoRA 학습 완료 (진행 중, ~88h)

```bash
# 자동 실행 중 (nohup)
/bin/bash lora/scripts/run_all_lora.sh

# 진행 상황 확인
/bin/bash scripts/check_progress.sh
```

**예상 완료 시점**: 클래스당 ~11h → 8클래스 순차 = **~3.5일** (2/20 금 → 2/24 화)

---

### Phase 2: 합성 이미지 생성 (`03_generate.py`)

```bash
python3 scripts/03_generate.py --all --multiplier 1 --denoise 0.25 0.35 0.45
```

**설정**:
- Pipeline: `StableDiffusionXLImg2ImgPipeline` + 클래스별 LoRA
- dtype: **fp16** (inference는 MPS에서 fp16 정상 동작)
- denoise_strength: 0.25 / 0.35 / 0.45 (3단계)
- guidance_scale: 6.0, steps: 25
- 생성량: 클래스당 train-set 크기 × 1배 (약 8,368장 × 3 denoise = **~25,000장**)

**예상 시간**: 25 inference steps × ~2s/step × 8,368장 × 3 = ~1,254,000초 → **약 14시간** (MPS fp16)

---

### Phase 3: 품질 필터링 (`04_filter_generated.py`)

```bash
python3 scripts/04_filter_generated.py --conf_threshold 0.7 --sharp_floor_pctile 20
```

**2단계 필터**:
1. **Class-identity gate**: baseline CNN confidence ≥ 0.70 AND argmax == target class
2. **Sharpness gate**: Laplacian variance ≥ real train p20 (클래스별 계산)

**예상 통과율**: 60-80% (denoise=0.35 기준 경험적 추정)

---

### Phase 4: CNN 학습 비교 실험 (`05_train_cnn.py`)

| 실험 | 명령 | 목적 |
|---|---|---|
| A: Real-only | ✅ 완료 (F1=0.9884) | 베이스라인 |
| B: Real+Augmented | `--mode real_augmented` | 전통적 augmentation |
| C: Real+Generated (ds025) | `--mode real_generated --denoise_tag ds025` | 약한 합성 |
| D: Real+Generated (ds035) | `--mode real_generated --denoise_tag ds035` | 중간 합성 |
| E: Real+Generated (ds045) | `--mode real_generated --denoise_tag ds045` | 강한 합성 |
| F: Real+Generated (혼합) | `--mode real_generated --denoise_tag all` | 전체 합성 |

---

### Phase 5: 견고성 평가 (`06_robustness_eval.py`)

```bash
python3 scripts/06_robustness_eval.py --all_ckpts
```

**5종 × 3 severity corruption**:

| Corruption | Severity 1 | 2 | 3 |
|---|---|---|---|
| Gaussian Noise | σ=0.05 | σ=0.10 | σ=0.15 |
| Blur | k=3 | k=5 | k=7 |
| JPEG artifact | q=90 | q=70 | q=50 |
| Brightness shift | ±0.1 | ±0.2 | ±0.3 |
| Contrast change | ×0.9 | ×0.8 | ×0.7 |

**비교 체크포인트**: A(baseline), B(augmented), C/D/E/F(generated variants)

---

### Phase 6: Ablation Study (`07_ablation.py`)

| 실험 | 내용 |
|---|---|
| A1: Class-wise vs Single LoRA | 클래스별 8개 vs 전체 통합 1개 |
| A2: Text2Img vs Img2Img | 조건 없는 생성 vs img2img conditioning |
| A3: Filter ON vs OFF | 품질 게이트 영향 |
| A4: 1× vs 2× vs 5× multiplier | 합성 데이터 비율의 영향 |

---

## 4. 핵심 Research Questions

1. **LoRA 합성 데이터가 baseline (real_only) 대비 macro-F1을 향상시키는가?**
   - 특히 최저 성능 클래스 `ig` (F1=0.9724)에서의 개선 기대

2. **최적 denoise strength는?**
   - 약한 denoise (0.25): real과 유사하나 다양성 낮음
   - 강한 denoise (0.45): 다양성 높으나 class-identity 손실 위험

3. **합성 데이터가 corruption robustness를 향상시키는가?**
   - 합성 이미지의 시각적 다양성이 OOD(Out-of-Distribution) 내성을 높이는지 검증

4. **품질 필터링의 효과는?**
   - A3 ablation: filter ON/OFF 비교

---

## 5. 현재 디렉토리 구조

```
wbc_synthesis/
├── data/
│   ├── raw/                    ✅ 17,092장 (PBC 데이터셋)
│   ├── processed/
│   │   ├── train/              ✅ 11,957장
│   │   ├── train_sharp/        ✅ 8,368장 (LoRA 소스)
│   │   ├── val/                ✅ 2,567장
│   │   └── test/               ✅ 2,568장
│   ├── generated/              ⏳ Phase 2 후 생성
│   └── filtered/               ⏳ Phase 3 후 생성
├── lora/
│   ├── weights/
│   │   └── basophil/           🔄 학습 중 (step 25/100)
│   └── scripts/                ✅ 8클래스 shell scripts
├── models/
│   └── baseline_cnn.pt         ✅ F1=0.9884
├── scripts/                    ✅ 전체 파이프라인 스크립트
├── logs/
│   └── lora_basophil.log       🔄 실시간 기록 중
└── results/
    └── baseline/               ✅ JSON 저장됨
```

---

## 6. 리스크 및 대응책

| 리스크 | 가능성 | 대응 |
|---|---|---|
| LoRA 100-step 품질 부족 | 중 | ablation 결과 보고 steps 조정 |
| 합성 이미지 pass-rate 낮음 | 중 | conf_threshold 0.7→0.6으로 완화 |
| 생성 시간 초과 | 낮 | multiplier=0.5로 축소 |
| MPS OOM during generation | 낮 | attention_slicing 활성화됨 |
| ig 클래스 개선 미달 | 중 | 생성 multiplier를 ig에만 2×로 설정 |