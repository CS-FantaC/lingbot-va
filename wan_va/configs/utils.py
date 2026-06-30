import json
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[2]
_RMBENCH_NORM_STAT_PATH = _REPO_ROOT / "norm_stats" / "rmbench_norm_stat.json"


def _load_rmbench_norm_stat(default_norm_stat):
    if not _RMBENCH_NORM_STAT_PATH.is_file():
        return default_norm_stat

    with open(_RMBENCH_NORM_STAT_PATH, "r", encoding="utf-8") as f:
        payload = json.load(f)

    norm_stat = payload.get("norm_stat", payload)
    q01 = norm_stat["q01"]
    q99 = norm_stat["q99"]
    if len(q01) != va_rmbench_train_cfg.action_dim or len(q99) != va_rmbench_train_cfg.action_dim:
        raise ValueError(
            f"Invalid rmbench norm stat shape from {_RMBENCH_NORM_STAT_PATH}: "
            f"len(q01)={len(q01)}, len(q99)={len(q99)}, expected {va_rmbench_train_cfg.action_dim}."
        )
    return {"q01": q01, "q99": q99}

