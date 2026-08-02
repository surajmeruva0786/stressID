"""Central configuration for the small-scale StressID pipeline.

Every stage reads its knobs from here so a run is reproducible from one object.
Defaults are tuned for the SMALL subset (fast smoke-scale runs on a 4 GB GPU).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

# ---------------------------------------------------------------- paths
REPO_ROOT = Path(__file__).resolve().parents[2]          # F:\stressID
DATASET_ROOT = REPO_ROOT / "StressID Dataset"
RESEARCH_ROOT = REPO_ROOT / "research_way"
DATA_DIR = RESEARCH_ROOT / "data"
RESULTS_DIR = RESEARCH_ROOT / "results"

MODALITIES = ("physio", "audio", "video")

# Tasks kept in the small subset. Mix of low-stress (Breathing, Relax) and
# high-stress (Math, Stroop, Speaking, Reading) tasks. Breathing/Relax have NO
# audio in StressID, which exercises the missing-modality path for free.
SMALL_TASKS = ("Breathing", "Relax", "Reading", "Math", "Stroop", "Speaking")

ALL_TASKS = (
    "Breathing", "Counting1", "Counting2", "Counting3", "Math", "Reading",
    "Relax", "Speaking", "Stroop", "Video1", "Video2",
)


@dataclass
class Config:
    # ---- subset (Stage 0) ----
    n_subjects: int = 16
    tasks: tuple = SMALL_TASKS
    require_all_modalities: bool = True   # subject must have physio+audio+video dirs
    require_all_tasks: bool = True        # subject must have every task in `tasks`
    seed: int = 1337

    # ---- windowing (Stage 1) ----
    # NOTE: the planning doc assumes 3-5 min recordings -> 30 s windows. Real
    # StressID tasks are 60-90 s, so 30 s windows yield only ~3 windows/task.
    # 10 s / 5 s hop gives ~11 windows for a 60 s task: a real temporal sequence.
    window_sec: float = 10.0
    hop_sec: float = 5.0
    max_windows: int = 16                 # pad/truncate the window sequence here

    # ---- physio ----
    physio_fs_raw: int = 500
    physio_fs: int = 64                   # resample target -> 640 samples/window
    physio_channels: tuple = ("ECG", "EDA", "RR")
    # "raw"      -> 1D-CNN + BiLSTM over the resampled waveform (original)
    # "features" -> A1: neurokit2 HRV/EDA/RSP descriptors per window -> MLP.
    # A1 exists because the learned encoder memorises on 448 recordings while
    # the origin paper's handcrafted features reach 0.73 (see §12.1).
    physio_mode: str = "raw"

    # ---- audio (log-mel front end; drop-in slot for wav2vec2) ----
    audio_sr: int = 16000
    n_mels: int = 40
    mel_win: int = 400                    # 25 ms
    mel_hop: int = 400                    # 25 ms -> 400 frames per 10 s window
    audio_frames: int = 400

    # ---- video ----
    video_fps_target: float = 2.5         # source is 5 fps -> take every 2nd frame
    video_face_size: int = 32             # 32x32 grayscale face crop
    video_frames: int = 25                # frames kept per window

    # ---- model ----
    d_model: int = 128
    n_heads: int = 4
    tokens_per_modality: int = 4
    fusion_layers: int = 1
    temporal_layers: int = 2
    dropout: float = 0.2
    modality_dropout_p: float = 0.25      # Stage 3 masked-token training

    # ---- losses ----
    w_binary: float = 1.0
    w_affect3: float = 0.5
    w_regression: float = 0.2
    w_subject_inv: float = 0.1
    subj_inv_temperature: float = 0.1

    # ---- training ----
    n_folds: int = 4
    epochs: int = 40
    batch_size: int = 8
    lr: float = 3e-4
    weight_decay: float = 1e-2
    grad_clip: float = 1.0
    seeds: tuple = (0, 1, 2)              # >=3 seeds for CI reporting
    device: str = "cuda"
    # A2: stop once val F1 has not improved for this many epochs (0 = never).
    # Val F1 peaks between epoch 1 and 22 and then decays while train BCE keeps
    # falling to 0.09, so the late epochs are pure memorisation.
    early_stop_patience: int = 0

    # ---- evaluation ----
    ece_bins: int = 10

    # ---- bookkeeping ----
    run_name: str = "small"
    # Manifest / splits / window cache are namespaced by this tag. The cache MUST
    # be per-tag: physio is z-scored per subject with stats pooled over that
    # subject's recordings *in the manifest*, so the same (subject, task) window
    # sequence differs between the 6-task subset and the 11-task full corpus.
    data_tag: str = "small"

    @property
    def window_samples_physio(self) -> int:
        return int(self.window_sec * self.physio_fs)

    @property
    def cache_dir(self) -> Path:
        return DATA_DIR / f"cache_{self.data_tag}"

    @property
    def physfeat_dir(self) -> Path:
        return DATA_DIR / f"physfeat_{self.data_tag}"

    @property
    def manifest_path(self) -> Path:
        return DATA_DIR / f"manifest_{self.data_tag}.csv"

    @property
    def splits_path(self) -> Path:
        return DATA_DIR / f"splits_{self.data_tag}.json"

    @property
    def results_dir(self) -> Path:
        return RESULTS_DIR / self.run_name

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, default=list), encoding="utf-8")


DEFAULT = Config()


def a1a2_config(**overrides) -> Config:
    """§12.1 A1 + A2: domain physio features + reduced capacity.

    Same corpus, splits and seeds as `full_config`, so every number is directly
    comparable to `results/full/`. Two changes only:

      A1  physio_mode="features" - hand the model neurokit2 HRV/EDA/RSP
          descriptors instead of asking a CNN+BiLSTM to rediscover them from
          raw signal on 448 training recordings.
      A2  d_model 128->64, fusion/temporal layers ->1, dropout 0.2->0.4,
          early stopping patience 8 - the 1.2 M-parameter model memorises
          (train BCE -> 0.09 while val F1 peaks at epoch 1-22 then decays).

    Both move in the same direction (less capacity spent on representation
    learning), so they share a run; `results/full/` is the control.
    """
    cfg = full_config(
        physio_mode="features",
        d_model=64,
        fusion_layers=1,
        temporal_layers=1,
        dropout=0.4,
        early_stop_patience=8,
        run_name="a1a2",
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def full_config(**overrides) -> Config:
    """The whole corpus: every subject, all 11 tasks, no modality requirement.

    64 subjects / 700 recordings. Physio exists for every recording; audio only
    for the 7 speech tasks (54% of recordings) and video for 83%, so the
    missing-modality path is driven entirely by real absence.
    """
    cfg = Config(
        n_subjects=64,
        tasks=ALL_TASKS,
        require_all_modalities=False,
        require_all_tasks=False,
        n_folds=5,
        run_name="full",
        data_tag="full",
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg
