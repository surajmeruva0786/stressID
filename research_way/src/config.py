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

    # ---- evaluation ----
    ece_bins: int = 10

    # ---- bookkeeping ----
    run_name: str = "small"

    @property
    def window_samples_physio(self) -> int:
        return int(self.window_sec * self.physio_fs)

    @property
    def cache_dir(self) -> Path:
        return CACHE_DIR

    @property
    def results_dir(self) -> Path:
        return RESULTS_DIR / self.run_name

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, default=list), encoding="utf-8")


DEFAULT = Config()
