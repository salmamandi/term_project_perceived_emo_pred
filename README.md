# Decoupled Valence–Arousal Prediction: Movie-Level 8-Fold Training and Ensemble Evaluation

## Purpose

This README documents two scripts intended for students who will **reuse the same model architecture and training/evaluation procedure on another dataset**:

1. `train_8fold_no_judge_decoupled.py` — trains separate valence and arousal models across eight movie-level folds.
2. `test_8fold_ensemble_on_judge.py` — loads trained models, averages their predictions on a completely held-out movie, and evaluates continuous and binary valence/arousal predictions. It additionally supports a rule-guided model (RG-MAN) and a combined ensemble.

The original experiment predicts **perceived valence and arousal** from precomputed video and audio features, with movie genre information. The exact neural-network layers and loss formula are defined in imported modules (`model.py`, `loss.py`, `config.py`), **not** in these two scripts. Consult those modules before claiming an exact architecture or reproducing hyperparameters.

> **Scope:** This document describes what the provided code implements. The scripts themselves do not establish that every preprocessing step, threshold, or rule-prior file is free from test-label information. Those dependencies must be audited before claiming fully leakage-free evaluation.

## Model architecture: Decoupled Cross-Modal Attention Network (MGN)

**Source of architectural description:** The original project `README.md`, Section 4, describes the network below. The two supplied train/test scripts instantiate `PerceivedVA_Model` from `Thilok_perceive_emo_pred_model/model.py` but **do not themselves define its internal layers**. Accordingly, the diagram and dimensions below reproduce the original project's documented design; they should be checked against `model.py` and `config.py` before implementation on a new dataset.

### Architecture at a glance

```text
 Video sequence: T × 768                  Audio sequence: T × 1024
       (MViT-B)                                  (wav2vec2)
           |                                          |
 Linear projection 768 -> 256             Linear projection 1024 -> 256
           |                                          |
   Temporal BiLSTM                              Temporal BiLSTM
   output: T × 512                              output: T × 512
           |                                          |
           +-------------------+----------------------+
                               |
                Bidirectional cross-modal attention
                 Video attends to Audio (V <- A)
                 Audio attends to Video (A <- V)
                         4 attention heads
                               |
                  Time-varying modality gates
                         g_v(t), g_a(t)
                               |
                  Gated multimodal sequence
                               |
                  Scene attention pooling
                     512-dimensional vector
                               |
     9-dimensional multi-hot genre -> 32-dimensional embedding
                               |
               Concatenate: 512 + 32 = 544
                               |
                   Fused scene vector z
                               |
                +--------------+--------------+
                |                             |
      Valence output head              Arousal output head
       544 -> 128 -> 1                  544 -> 128 -> 1
        ReLU, dropout                    ReLU, dropout
           sigmoid                          sigmoid
                |                             |
         Valence in [0,1]               Arousal in [0,1]
```

**Crucial training distinction:** Although the model's forward pass returns both valence and arousal, `train_8fold_no_judge_decoupled.py` creates **two separately initialized full `PerceivedVA_Model` instances per fold**. The valence specialist is optimized with the valence prediction and label; the arousal specialist is optimized with the arousal prediction and label. At test time, the valence output is taken from valence-specialist checkpoints and the arousal output from arousal-specialist checkpoints. The diagram's two heads therefore describe the model's possible outputs, **not** one shared jointly trained model used for both final predictions.

### Layer-by-layer explanation

**1. Input representations.** The network does not operate on raw pixels or audio waveforms in these scripts. `SceneFeatureDataset` loads precomputed temporal feature sequences. The project README identifies MViT-B video embeddings (768 features per time step) and wav2vec2 audio embeddings (1024 features per time step). The dataset loader supplies a `mask` for padded/valid sequence positions and `clip_names` for looking up labels and genre.

**2. Modality-specific linear projection.** Separate learned linear layers map video `768 -> 256` and audio `1024 -> 256` into a common feature width. For a different feature extractor, update the input dimensions and confirm the projections in `model.py`.

**3. Temporal BiLSTM encoders.** The project README describes a bidirectional LSTM for each modality, yielding a 512-dimensional temporal representation per time step. Forward and backward recurrence provide temporal context within the clip.

**4. Bidirectional cross-modal attention.** Four attention heads allow video representations to attend to audio and audio representations to attend to video. The project describes scaled dot-product attention, conceptually:

$$
\operatorname{Attention}(Q,K,V)=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V.
$$

The two attention directions use different query and key/value modalities. Exact layer ordering, masking, and residual connections require inspection of `model.py`.

**5. Continuous modality gating.** Rather than always trusting both modalities equally, the architecture describes learned time-varying gates $g_v(t)$ and $g_a(t)$ between 0 and 1. The original README presents the conceptual fusion as:

$$
F_t=g_v(t)\odot V_t^{\mathrm{attn}}+g_a(t)\odot A_t^{\mathrm{attn}}.
$$

This is **learned soft weighting**, not a hard rule-based clip discard. The exact gate implementation and whether the gates are normalized against each other must be checked in the model source.

**6. Dominant scene attention pooling.** Attention weights over time produce one clip-level vector from the gated temporal sequence:

$$
c=\sum_{t=1}^{T}\alpha_tF_t,\qquad \sum_t\alpha_t=1.
$$

The project describes the pooled scene representation as 512-dimensional. This allows different time steps to contribute unequally to the final emotion prediction.

**7. Genre conditioning.** A nine-dimensional multi-hot movie-genre vector is projected to a 32-dimensional learned embedding. Concatenating this with the pooled scene vector gives a **544-dimensional fused representation**. Genre information is loaded per clip via `load_genre_labels`. A new dataset without genre labels needs an explicit architecture change or another well-defined contextual input.

**8. Prediction heads.** The documented head layout is `Linear(544,128) -> ReLU -> Dropout -> Linear(128,1) -> Sigmoid`, producing a continuous score in `[0,1]` for each emotion dimension. The test script uses these continuous scores for CCC and later thresholds them for binary accuracy.

### Optimization and task-specific decoupling

The training script performs:

```python
pred_v, pred_a, delta = model(video, audio, mask, genres)

if task_name == "valence":
    gt = valence_labels
    pred = pred_v
else:
    gt = arousal_labels
    pred = pred_a

loss = compute_loss_task(pred, gt, delta, alpha=0.5)
```

The project README describes a hybrid objective with MSE, concordance correlation coefficient (CCC), and a temporal-change penalty:

$$
\mathcal{L}=0.5\,\mathrm{MSE}(y,\hat y)+0.5\,(1-\mathrm{CCC}(y,\hat y))
+0.01\lVert\Delta\hat y\rVert_2^2.
$$

**Verification required:** The actual training script calls `compute_loss_task(...)` from `loss.py`; the supplied files do not expose that function's implementation. Verify the exact CCC reduction, delta term, and coefficient handling in `loss.py` before treating the formula above as the executed loss.

For each validation movie, the script saves the valence and arousal specialists independently according to their own best validation loss. During ensemble inference, each specialist contributes only its corresponding task prediction. This is the operational meaning of *decoupled* in the supplied training/testing workflow.

### What students must carry over to another dataset

- **Mandatory code:** `model.py`, `dataset.py`, `config.py`, `loss.py`, `utils.py`, plus the two supplied train/test scripts. `rule_guided_attention_model.py` is additionally imported by the unmodified test script.
- **Input contract:** Video tensor, audio tensor, valid-time-step mask, sample identifiers, labels, and genre vectors must match the model's expected shapes and value ranges.
- **Dimensions:** The 768/1024 input widths and 9-way genre representation are specific to the original feature pipeline. Update dimensions and associated layers if using other extractors.
- **Output target:** The network predicts continuous valence/arousal; binary labels are created separately with a documented threshold policy.
- **Independent models:** Preserve one valence specialist and one arousal specialist per fold if reproducing this decoupled setup; a single multitask model is a different experimental condition.
- **Architecture validation:** Check actual tensor shapes, masking, attention implementation, and `delta` semantics against the model code before claiming an exact reproduction.

---

## 1. Original dataset and experimental split

The scripts expect nine movies (896 clips in the reported experiment):

| Role | Movies | Reported clips |
|---|---|---:|
| Held-out test | `TheJudge` | 149 |
| Training/validation pool | `AboutTime`, `Gifted`, `LadyBird`, `TheBlindSide`, `TheFaultInOurStars`, `TheSpectacularNow`, `TheTheoryOfEverything`, `Wonder` | 747 |

The training script discovers the available movie names from `dataset.samples`, then excludes `TheJudge`:

```python
HELD_OUT_TARGET = "TheJudge"
all_available = sorted(set(s[0] for s in dataset.samples))
TRAIN_POOL_MOVIES = [m for m in all_available if m != HELD_OUT_TARGET]
pool_indices = [i for i, s in enumerate(dataset.samples)
                if s[0] in TRAIN_POOL_MOVIES]
```

**Important:** The script does not independently enforce that exactly nine movies, eight pool movies, or 747 pool clips are present. These are properties of the original dataset, not general assertions checked by the code. The test script separately hard-codes the eight training-pool movie names.

## 2. Training methodology: movie-level eight-fold cross-validation

The eight non-test movies form **eight folds**, one validation movie per fold:

- Fold 1: train on seven movies, validate on `AboutTime`.
- Fold 2: train on seven movies, validate on `Gifted`.
- Continue until every pool movie has served as validation once.
- **No `TheJudge` clips** are placed in either training or validation indices.

For each fold, two **separately initialized instances** of `PerceivedVA_Model` are trained:

- **Valence specialist:** model forward pass returns `(pred_v, pred_a, delta)`; loss uses `pred_v` and the valence label.
- **Arousal specialist:** another model instance uses `pred_a` and the arousal label.

The result is **eight valence checkpoints plus eight arousal checkpoints** (16 total). The two specialists are complete model instances, not simply two heads trained simultaneously in a single shared model.

### Training configuration

| Setting | Implementation |
|---|---|
| Epochs | `20` |
| Batch size | `config.BATCH_SIZE` |
| Learning rate | `config.LEARNING_RATE` |
| Optimizer | Adam |
| Weight decay | `1e-5` |
| Gradient clipping | Maximum norm `1.0` |
| Loss | `compute_loss_task(pred, gt, delta, alpha=0.5)` from `loss.py` |
| Checkpoint criterion | Lowest **validation loss** across epochs |
| Training loader | Shuffled |
| Validation loader | Not shuffled |

The script trains for all 20 epochs and saves a checkpoint whenever validation loss improves. **It does not implement an early-stopping break**, despite wording in its introductory comment.

Each model consumes the batch fields `(video, audio, mask, clip_names)` from `SceneFeatureDataset`/`collate_fn`. The script obtains a genre tensor for each clip using `load_genre_labels`. Exact feature dimensions and model internals come from `config.py` and `model.py`.

### Checkpoint output

```text
Thilok_Outputs/
└── saved_models_8fold_no_judge/
    ├── valence/
    │   ├── decoupled_val_val_AboutTime.pth
    │   ├── decoupled_val_val_Gifted.pth
    │   └── ... (8 files total)
    └── arousal/
        ├── decoupled_aro_val_AboutTime.pth
        ├── decoupled_aro_val_Gifted.pth
        └── ... (8 files total)
```

Each `.pth` is saved using `torch.save(model.state_dict(), save_path)`: it contains model parameters, **not** optimizer state or a complete training-resume snapshot. The movie name in the filename denotes the **validation movie** for that fold.

## 3. Testing methodology: ensemble on the held-out movie

The testing script constructs a dataset using the same feature roots and selects only clips whose movie identifier is `TheJudge`. The original evaluation reports 149 test clips.

### 3.1 Load saved models

For the **Decoupled MGN** ensemble, it loads:

- Up to eight valence models from `saved_models_8fold_no_judge/valence/`.
- Up to eight arousal models from `saved_models_8fold_no_judge/arousal/`.

The script also attempts to load **RG-MAN** models from `saved_models_8fold_no_judge/rg_man/`. It supports separate RG-MAN valence/arousal specialists or joint RG-MAN checkpoints, depending on which files exist. This is an **additional architecture**; the Decoupled training script does not train RG-MAN.

**Caution:** Missing checkpoints are silently skipped, with only a printed count. For a genuine eight-model ensemble, verify **8/8** valence and **8/8** arousal checkpoints have loaded. The testing script imports the RG-MAN model module even if only Decoupled results are wanted.

### 3.2 Prediction and averaging

Each loaded model predicts a continuous value for each test clip. The Decoupled ensemble uses **unweighted arithmetic averaging**:

\[
\hat v(c)=\frac{1}{K_v}\sum_{k=1}^{K_v}\hat v_k(c),\qquad
\hat a(c)=\frac{1}{K_a}\sum_{k=1}^{K_a}\hat a_k(c)
\]

Here `K_v` and `K_a` are the numbers of successfully loaded valence and arousal models, respectively; both should equal eight in the intended experiment.

The RG-MAN ensemble similarly averages its models' predictions. The **Combined Master Ensemble** then takes the arithmetic mean of the Decoupled and RG-MAN ensemble predictions for each dimension, when available:

\[
\hat v_{\mathrm{blend}}=\operatorname{mean}(\hat v_{\mathrm{Dec}},\hat v_{\mathrm{RG}}),\quad
\hat a_{\mathrm{blend}}=\operatorname{mean}(\hat a_{\mathrm{Dec}},\hat a_{\mathrm{RG}}).
\]

These are **prediction-level ensembles**; there is no learned ensemble-weighting stage in the supplied testing script.

### 3.3 Binary labels and thresholds

Continuous predictions are converted to binary classes using `bin_value_binary` from `utils.py`. For the original experiment, the test script reads movie-specific thresholds from:

- `config.VALENCE_THRESH_FILE`
- `config.AROUSAL_THRESH_FILE`

It selects the rows whose `movie_name` is `TheJudge`; if a row is missing, it falls back to `0.41` (valence) or `0.57` (arousal). The ground-truth binary labels come from `load_binary_labels(...)`.

**Critical reproducibility and leakage check:** Determine how these threshold CSVs were generated. If thresholds for the test movie were estimated from its ground-truth emotion annotations, the binary evaluation is not fully independent of test labels. For a new dataset, select and freeze a thresholding policy using only permitted training/validation information (or a genuinely predetermined threshold), and apply a consistent definition to predictions and ground truth.

### 3.4 Evaluation metrics

The testing script computes, for each available ensemble configuration:

| Metric | Meaning / code source |
|---|---|
| Valence CCC | Concordance correlation coefficient for continuous valence predictions, via `compute_metrics` |
| Arousal CCC | Concordance correlation coefficient for continuous arousal predictions, via `compute_metrics` |
| Valence accuracy | Binary valence classification accuracy, via `compute_classification_metrics` |
| Arousal accuracy | Binary arousal classification accuracy |
| Valence macro-F1 | Macro-averaged binary valence F1 |
| Arousal macro-F1 | Macro-averaged binary arousal F1 |
| Joint clip accuracy | Percentage of clips where **both** binary valence **and** binary arousal are correct |

The script also calculates a heuristic high-confidence flag when `abs(2*v - 1) >= 0.65` **and** `abs(2*a - 1) >= 0.65`. This is not a calibrated confidence score and is centered on `0.5`, which may differ from the movie-specific decision thresholds. Although the header mentions high-confidence accuracy, the provided summary calculation **does not report a separate high-confidence accuracy metric**.

### 3.5 Saved evaluation files

```text
Thilok_Outputs/results_8fold_no_judge_test_on_judge/
├── the_judge_8fold_ensemble_summary.csv
└── the_judge_per_clip_ensemble_predictions.csv
```

- **Summary CSV:** one row per available configuration, with clips evaluated, CCC, accuracy, macro-F1 and joint clip accuracy.
- **Per-clip CSV:** clip name; RG-MAN, Decoupled, and blended continuous/binary predictions; blended heuristic confidence; continuous and binary ground truth; and per-configuration joint correctness.

The script **overwrites** these files on each run. Back up previous results before rerunning.

## 4. Code dependencies and expected project layout

The two supplied scripts **cannot run independently**. Their imports require at least:

```text
project_root/
├── train_8fold_no_judge_decoupled.py
├── test_8fold_ensemble_on_judge.py
├── Thilok_perceive_emo_pred_model/
│   ├── config.py
│   ├── dataset.py
│   ├── model.py
│   ├── loss.py
│   ├── utils.py
│   └── rule_guided_attention_model.py   # required by current test script
└── Thilok_Outputs/
    └── saved_models_8fold_no_judge/
        ├── valence/
        ├── arousal/
        └── rg_man/                    # optional models; imported code still required
```

Additional inputs are specified by `config.py`: video features, audio features, label file, and threshold CSVs. The testing script optionally reads rule-prior files named:

- `audio_valence_misalignment_check.csv`
- `video_valence_misalignment_check.csv`
- `useful_nonuseful_names_manual_correct.json`

These are used to construct **RG-MAN** rule priors; their absence yields default priors rather than necessarily stopping execution. Their provenance should be checked for test-label leakage.

The scripts use Python, NumPy, pandas, and PyTorch. Exact package versions and GPU requirements are **not specified by these scripts**. A compatible environment and feature format must be established from the rest of the project.

## 5. Running the original experiment

From the directory containing both scripts and the imported package:

```bash
# Activate an environment with compatible dependencies
conda activate YOUR_ENV_NAME

# Train the Decoupled models (generates 16 checkpoints)
python train_8fold_no_judge_decoupled.py

# Evaluate saved checkpoints on TheJudge
python test_8fold_ensemble_on_judge.py
```

Optional GPU/background training command documented in the training script:

```bash
CUDA_VISIBLE_DEVICES=1 nohup python -u train_8fold_no_judge_decoupled.py \
  >> Thilok_Outputs/train_8fold_decoupled_log.txt 2>&1 &
```

**Before running the test script:** check the printed checkpoint counts, feature paths, label paths, and threshold files. RG-MAN results require its own trained checkpoints; they are not produced by the Decoupled training script.

## 6. Adapting the experiment to a new dataset

Students should retain the **protocol** while replacing dataset-specific assumptions:

1. **Prepare features and labels.** Implement or adapt `SceneFeatureDataset` so each example yields video embeddings, audio embeddings, a sequence mask, and a unique sample identifier. Ensure `load_labels` and `load_genre_labels` support the new identifiers and labels. If genre is unavailable, modify the model and inputs consistently rather than supplying arbitrary genre values.
2. **Verify feature dimensions.** Set `VIDEO_DIM`, `AUDIO_DIM`, `GENRE_DIM`, and other model settings in `config.py` to match the new dataset. Pretrained feature extractors and embedding shapes are not defined by the two scripts.
3. **Choose an independent test group.** The original group is a movie; in another dataset, use a scientifically appropriate group such as participant, session, or source recording. Keep **all samples from that group** out of training, validation, preprocessing fits, and model selection.
4. **Create folds within the remaining groups.** The original code assumes eight training/validation movies and trains one fold per movie. With `N` remaining groups, a leave-one-group-out approach creates `N` models per task. Alternatively, define a different grouped K-fold scheme explicitly and update the number of expected checkpoints.
5. **Update hard-coded names and paths.** Replace `HELD_OUT_TARGET = "TheJudge"`, the testing script's `TRAIN_POOL_MOVIES` list, all `TheJudge` filters, threshold row lookups, output names, and model/checkpoint directories as needed.
6. **Train task-specific models.** For every fold, train one valence and one arousal model using training groups, choose each checkpoint by validation loss, and record which group was held out for validation.
7. **Evaluate on the untouched test group.** Load all intended checkpoints, predict each test sample, average predictions per task, and calculate CCC plus binary metrics using a **predeclared, leakage-free** threshold policy.
8. **Audit rule priors if using RG-MAN.** If the new dataset lacks equivalent rule features, either generate them independently from permissible inputs or remove the RG-MAN branch from the testing script. The Decoupled MGN model itself does not need RG-MAN priors.
9. **Report a meaningful baseline.** Compare the ensemble with a single model trained on the available training groups and, where useful, individual fold-model results. Keep the same test set and evaluation policy across comparisons.

### Recommended changes before reusing the scripts

- Make `TEST_GROUP`, `TRAIN_GROUPS`, `N_FOLDS`, paths, and output prefix configurable instead of hard-coded.
- Require the expected number of checkpoints; fail loudly rather than silently evaluating fewer models.
- Save the training/validation group assignments, random seeds, hyperparameters, and per-fold validation losses.
- Fit any normalization only on permitted training data, and save/apply those fitted statistics consistently.
- Freeze binary threshold definitions without accessing held-out test labels.
- Remove the RG-MAN import and evaluation branches when running a Decoupled-only experiment.
- Handle missing predictions explicitly before calculating the combined ensemble.
- Report the number of evaluated clips, and verify all sample identifiers match across features, labels, and predictions.

## 7. Reproducibility checklist

- [ ] All held-out test-group samples are absent from training and validation.
- [ ] Feature preprocessing and normalization do not use held-out test labels or test-derived fitted statistics.
- [ ] All expected valence and arousal checkpoints exist and load successfully.
- [ ] Binary ground truth and predicted labels use a documented, consistent thresholding policy.
- [ ] Thresholds and any RG-MAN rule priors are audited for leakage.
- [ ] The same held-out test set is used for every compared method.
- [ ] Per-sample predictions and aggregate metrics are saved.
- [ ] Model architecture, feature dimensions, seeds, and dependency versions are documented.

## 8. What these two scripts do *not* establish

- The exact **implemented** attention/gating/temporal architecture: the architecture section above reproduces the original project README, but must be verified against `model.py` and (for RG-MAN) `rule_guided_attention_model.py`.
- The exact mathematical form of `compute_loss_task`: inspect `loss.py`.
- How the original audio/video embeddings and label thresholds were generated: inspect preprocessing scripts and data files.
- Whether the original reported numerical results reproduce: requires the actual features, labels, checkpoints, and execution environment.
- Whether the ensemble outperforms a single full-training-pool model: requires a controlled baseline experiment.

**Bottom line:** The supplied training script implements **leave-one-movie-out validation within an eight-movie training pool**, saving eight valence and eight arousal specialists. The supplied testing script implements **mean-aggregation inference on a ninth, held-out movie**, plus optional RG-MAN and blended comparisons. For another dataset, preserve the group-separated training/evaluation design while replacing all movie-specific assumptions and verifying leakage-free thresholds and preprocessing.

