import argparse
import os
import time
import yaml
from pathlib import Path

def load_config(cfg_path: str):
    cfg_path = Path(cfg_path)
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # handle simple inheritance
    if "inherits" in cfg:
        base_path = cfg_path.parent / cfg["inherits"]
        with open(base_path, "r", encoding="utf-8") as f:
            base = yaml.safe_load(f)
        base.update(cfg)
        cfg = base
        cfg.pop("inherits", None)

    return cfg

def make_run_dir(cfg):
    ts = time.strftime("%Y%m%d_%H%M%S")
    method = cfg.get("method", {}).get("name", "unknown")
    run_name = f"{ts}_{method}"
    out_root = Path(cfg["logging"]["save_dir"])
    run_dir = out_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    run_dir = make_run_dir(cfg)

    # save resolved config for reproducibility
    with open(run_dir / "config_resolved.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    print(f"[run] method={cfg.get('method', {}).get('name')}")
    print(f"[run] output={run_dir}")

    # dispatch
    method = cfg.get("method", {}).get("name")
    if method == "rrr":
        from src.rrr.train_rrr import train_rrr
        train_rrr(cfg, run_dir)
    elif method == "step_credit":
        from src.step_credit.train_step_credit import train_step_credit
        train_step_credit(cfg, run_dir)
    else:
        raise ValueError(f"Unknown method: {method}")

if __name__ == "__main__":
    main()