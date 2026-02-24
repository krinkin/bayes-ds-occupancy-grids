#!/usr/bin/env python3
"""Spatial block bootstrap 95% CIs for Intel Lab metric deltas.

Usage::

    source .venv/bin/activate
    python scripts/bootstrap-intel-lab.py --config configs/intel-lab/default.yaml
    python scripts/bootstrap-intel-lab.py --config configs/intel-lab/default.yaml --sensitivity

Loads the Intel Lab dataset, rebuilds probability maps for each arm,
then runs spatial block bootstrap to compute 95% percentile CIs for
the per-metric delta (DS/TBM minus Bayesian+count).

The spatial block bootstrap preserves spatial autocorrelation by
resampling non-overlapping B x B cell blocks with replacement.

Results are saved to results/intel-lab/bootstrap-cis.json and
appended to results/intel-lab/analysis-summary.md.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.experiments.intel_lab.adapter import IntelLabAdapter
from src.experiments.intel_lab.loader import load_intel_lab
from src.experiments.intel_lab.runner import (
    build_multi_robot_prob_maps,
    build_single_robot_prob_maps,
    spatial_block_bootstrap_delta_ci,
)


def _load_config(config_path: str | None) -> dict:
    if config_path is None:
        return {}
    with open(config_path) as f:
        return yaml.safe_load(f)


def _sign_note(metric: str, lo: float, hi: float) -> str:
    """Return significance note: does the CI exclude zero?

    For brier_score (lower-is-better): positive delta = DS/TBM worse.
    For all other metrics (higher-is-better): positive delta = DS/TBM better.
    """
    lower_is_better = metric == "brier_score"
    if lo > 0:
        return "DS/TBM significantly worse (CI > 0)" if lower_is_better else "DS/TBM significantly better (CI > 0)"
    if hi < 0:
        return "DS/TBM significantly better (CI < 0)" if lower_is_better else "DS/TBM significantly worse (CI < 0)"
    return "not significant (CI spans 0)"


def _run_bootstrap(
    prob_maps: dict,
    gt,
    gt_mask,
    metrics: list[str],
    block_size: int,
    n_iter: int,
    seed: int,
) -> dict[str, dict]:
    """Run spatial block bootstrap and return CIs dict."""
    return spatial_block_bootstrap_delta_ci(
        prob_bc=prob_maps["bayesian_count"],
        prob_ds=prob_maps["dstbm"],
        gt=gt,
        gt_mask=gt_mask,
        metrics=metrics,
        block_size=block_size,
        n_iter=n_iter,
        seed=seed,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Spatial block bootstrap CIs for Intel Lab metric deltas"
    )
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument("--n-iter", type=int, default=10000, help="Bootstrap iterations")
    parser.add_argument("--block-size", type=int, default=10,
                        help="Block side length in cells (default 10 = 1.0 m at 0.1 m)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--sensitivity", action="store_true",
                        help="Run sensitivity analysis with B=5,10,20")
    args = parser.parse_args(argv)

    cfg = _load_config(args.config)
    data_path = cfg.get("data", {}).get("path", "data/intel-lab/intel.gfs")
    resolution = cfg.get("grid", {}).get("resolution", 0.1)
    max_range = cfg.get("grid", {}).get("max_range", 8.0)
    train_fraction = cfg.get("grid", {}).get("train_fraction", 0.8)
    multi_splits = cfg.get("experiment", {}).get("multi_robot_splits", [2, 4])
    output_dir = Path(cfg.get("output", {}).get("dir", "results/intel-lab"))
    l_occ = float(cfg.get("sensor", {}).get("l_occ", 2.0))
    l_free = float(cfg.get("sensor", {}).get("l_free", -0.5))
    metrics = ["cell_accuracy", "boundary_sharpness", "brier_score"]
    arm_names = ["bayesian_count", "dstbm"]

    block_meters = args.block_size * resolution
    print("Intel Lab Spatial Block Bootstrap CI Analysis")
    print(f"  Data: {data_path}")
    print(f"  Block size: {args.block_size} cells ({block_meters:.1f} m)")
    print(f"  Bootstrap iterations: {args.n_iter}, seed: {args.seed}")
    if args.sensitivity:
        print("  Sensitivity analysis: B = {5, 10, 20}")
    print(flush=True)

    # Load and set up
    t0 = time.time()
    scans = load_intel_lab(data_path)
    print(f"  Loaded {len(scans)} scans ({time.time()-t0:.1f}s)", flush=True)

    adapter = IntelLabAdapter(
        scans=scans,
        resolution=resolution,
        max_range=max_range,
        train_fraction=train_fraction,
    )
    gt, gt_mask = adapter.get_ground_truth()
    n_gt_cells = int(gt_mask.sum())
    n_wall_cells = int((gt_mask & (gt >= 0.5)).sum())
    print(f"  GT cells: {n_gt_cells}, wall cells: {n_wall_cells}")
    print(flush=True)

    # Build prob maps once (reused across block sizes)
    configs: list[tuple[str, dict]] = []

    print("=== Single-Robot ===", flush=True)
    t0 = time.time()
    single_maps = build_single_robot_prob_maps(adapter, arm_names, l_occ, l_free)
    print(f"  Built probability maps ({time.time()-t0:.1f}s)", flush=True)
    configs.append(("single_robot", single_maps))

    for num_robots in multi_splits:
        key = f"split_{num_robots}_robots"
        print(f"\n=== {num_robots}-Robot Split ===", flush=True)
        t0 = time.time()
        multi_maps = build_multi_robot_prob_maps(adapter, num_robots, arm_names, l_occ, l_free)
        print(f"  Built probability maps ({time.time()-t0:.1f}s)", flush=True)
        configs.append((key, multi_maps))

    # Primary bootstrap (default block size)
    all_cis: dict[str, dict[str, dict]] = {}
    print(f"\n--- Bootstrap (B={args.block_size}) ---", flush=True)
    for cfg_key, prob_maps in configs:
        t0 = time.time()
        cis = _run_bootstrap(prob_maps, gt, gt_mask, metrics,
                             args.block_size, args.n_iter, args.seed)
        print(f"  {cfg_key}: done ({time.time()-t0:.1f}s)", flush=True)
        all_cis[cfg_key] = cis

        for m, ci in cis.items():
            lo, hi = ci["lo"], ci["hi"]
            note = _sign_note(m, lo, hi)
            print(f"    {m}: CI [{lo:+.4f}, {hi:+.4f}] ({ci['n_blocks']} blocks)  {note}")

    # Save primary CIs
    output_dir.mkdir(parents=True, exist_ok=True)
    ci_path = output_dir / "bootstrap-cis.json"
    with ci_path.open("w") as fh:
        json.dump(all_cis, fh, indent=2)
    print(f"\nBootstrap CIs saved to {ci_path}", flush=True)

    # Sensitivity analysis
    sensitivity_cis: dict[int, dict[str, dict[str, dict]]] = {}
    if args.sensitivity:
        block_sizes = [5, 10, 20]
        for bs in block_sizes:
            sensitivity_cis[bs] = {}
            bs_meters = bs * resolution
            print(f"\n--- Sensitivity: B={bs} ({bs_meters:.1f} m) ---", flush=True)
            for cfg_key, prob_maps in configs:
                t0 = time.time()
                cis = _run_bootstrap(prob_maps, gt, gt_mask, metrics,
                                     bs, args.n_iter, args.seed)
                print(f"  {cfg_key}: done ({time.time()-t0:.1f}s)", flush=True)
                sensitivity_cis[bs][cfg_key] = cis

                for m, ci in cis.items():
                    lo, hi = ci["lo"], ci["hi"]
                    print(f"    {m}: CI [{lo:+.4f}, {hi:+.4f}] ({ci['n_blocks']} blocks)")

        # Save sensitivity results
        sens_path = output_dir / "bootstrap-sensitivity.json"
        # Convert int keys to str for JSON
        sens_out = {str(bs): v for bs, v in sensitivity_cis.items()}
        with sens_path.open("w") as fh:
            json.dump(sens_out, fh, indent=2)
        print(f"\nSensitivity results saved to {sens_path}", flush=True)

    # Load existing results for point estimates
    results_path = output_dir / "results.json"
    results: dict = {}
    if results_path.exists():
        with results_path.open() as fh:
            results = json.load(fh)

    # Update analysis-summary.md
    _append_bootstrap_section(
        output_dir / "analysis-summary.md",
        all_cis,
        results,
        metrics,
        n_iter=args.n_iter,
        block_size=args.block_size,
        resolution=resolution,
        sensitivity_cis=sensitivity_cis if args.sensitivity else None,
    )
    print(f"Updated {output_dir / 'analysis-summary.md'}")


def _append_bootstrap_section(
    summary_path: Path,
    all_cis: dict,
    results: dict,
    metrics: list[str],
    n_iter: int,
    block_size: int,
    resolution: float,
    sensitivity_cis: dict | None = None,
) -> None:
    """Append spatial block bootstrap CI section to analysis-summary.md."""
    block_meters = block_size * resolution
    lines: list[str] = []
    lines.append(
        f"\n## Bootstrap Confidence Intervals "
        f"(95%, spatial block bootstrap, B={block_size} = {block_meters:.1f} m)\n"
    )
    lines.append(f"Bootstrap iterations: {n_iter}, seed: 42\n\n")
    lines.append(
        "Delta = DS/TBM minus Bayesian+count.  "
        "For cell_accuracy/boundary_sharpness: negative = DS/TBM worse.  "
        "For brier_score: positive = DS/TBM worse.\n\n"
    )

    # Wide table
    header_cols = ["Setup", "Metric", "B+count", "DS/TBM", "Delta", "95% CI", "Blocks", "Sig"]
    col_widths = [14, 22, 10, 10, 10, 22, 8, 6]
    sep = " | "

    def row(*cells: str) -> str:
        parts = [str(c).ljust(w) for c, w in zip(cells, col_widths)]
        return "| " + sep.join(parts) + " |\n"

    def divider() -> str:
        parts = ["-" * w for w in col_widths]
        return "|" + "|".join("-" + p + "-" for p in parts) + "|\n"

    lines.append(row(*header_cols))
    lines.append(divider())

    config_display = {
        "single_robot": "single_robot",
        "split_2_robots": "2-robot",
        "split_4_robots": "4-robot",
    }

    for cfg_key, cfg_label in config_display.items():
        ci_block = all_cis.get(cfg_key, {})
        res_bc = results.get(cfg_key, {}).get("bayesian_count", {})
        res_ds = results.get(cfg_key, {}).get("dstbm", {})
        paired = results.get(cfg_key, {}).get("_paired_deltas", {})
        for m in metrics:
            bc_val = res_bc.get(m, float("nan"))
            ds_val = res_ds.get(m, float("nan"))
            if m in paired:
                delta = paired[m]
            elif bc_val == bc_val and ds_val == ds_val:
                delta = ds_val - bc_val
            else:
                delta = float("nan")
            ci = ci_block.get(m, {})
            lo = ci.get("lo", float("nan"))
            hi = ci.get("hi", float("nan"))
            n_blk = ci.get("n_blocks", "?")
            if lo == lo and hi == hi:
                ci_str = f"[{lo:+.4f}, {hi:+.4f}]"
                sig = "yes" if (lo > 0 or hi < 0) else "no"
            else:
                ci_str = "n/a"
                sig = "n/a"
            delta_str = f"{delta:+.4f}" if delta == delta else "n/a"
            bc_str = f"{bc_val:.4f}" if bc_val == bc_val else "n/a"
            ds_str = f"{ds_val:.4f}" if ds_val == ds_val else "n/a"
            lines.append(row(cfg_label, m, bc_str, ds_str, delta_str, ci_str,
                             str(n_blk), sig))

    lines.append("\n### Interpretation\n\n")
    lines.append(
        "- **Significant** = 95% CI excludes zero; i.e., p < 0.05 under bootstrap test.\n"
    )
    lines.append(
        "- Spatial block bootstrap with non-overlapping "
        f"{block_size}x{block_size} cell blocks ({block_meters:.1f} m).\n"
    )
    lines.append(
        "- Block resampling preserves spatial autocorrelation within blocks, "
        "yielding conservative CIs.\n"
    )

    # Sensitivity table
    if sensitivity_cis:
        lines.append("\n### Block Size Sensitivity Analysis\n\n")
        lines.append("| Block | Setup | Metric | 95% CI | Blocks |\n")
        lines.append("|-------|-------|--------|--------|--------|\n")
        for bs in sorted(sensitivity_cis.keys()):
            bs_meters = bs * resolution
            bs_label = f"B={bs} ({bs_meters:.1f}m)"
            for cfg_key, cfg_label in config_display.items():
                ci_block = sensitivity_cis[bs].get(cfg_key, {})
                for m in metrics:
                    ci = ci_block.get(m, {})
                    lo = ci.get("lo", float("nan"))
                    hi = ci.get("hi", float("nan"))
                    n_blk = ci.get("n_blocks", "?")
                    if lo == lo and hi == hi:
                        ci_str = f"[{lo:+.4f}, {hi:+.4f}]"
                    else:
                        ci_str = "n/a"
                    lines.append(
                        f"| {bs_label} | {cfg_label} | {m} | {ci_str} | {n_blk} |\n"
                    )
                    bs_label = ""  # only show block size once per group

    with summary_path.open("a") as fh:
        fh.writelines(lines)


if __name__ == "__main__":
    main()
