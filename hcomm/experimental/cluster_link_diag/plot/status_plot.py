#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import argparse
import csv
import json
import logging
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt


os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "output"


@dataclass
class SeriesPlotConfig:
    turns: list
    series: dict
    labels: dict
    output_png: Path
    title: str
    ylabel: str
    selected_keys: list = None


def find_latest_run_dir(output_root: Path) -> Path:
    candidates = []
    for path in output_root.iterdir():
        if path.is_dir() and ((path / "link_lat.txt").exists() or (path / "link_pass_rate.txt").exists()):
            candidates.append(path)
    if not candidates:
        raise RuntimeError(f"no run output directory found under {output_root}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def clean_cell(value: str) -> str:
    return value.strip().strip(",")


def parse_l1_matrix(path: Path, limit: int):
    if not path.exists():
        return [], []
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return [], []

    labels = [clean_cell(item) for item in lines[0].split(",") if clean_cell(item)]
    rows = []
    for line in lines[1:]:
        values = []
        for item in line.split(","):
            item = clean_cell(item)
            if not item:
                continue
            try:
                values.append(float(item))
            except ValueError:
                values.append(float("nan"))
        if values:
            rows.append(values[: len(labels)])
    if limit > 0:
        rows = rows[-limit:]
    return labels, rows


def parse_l2_table(path: Path, value_column: str, limit: int):
    if not path.exists():
        return [], {}, {}

    records_by_turn = defaultdict(dict)
    labels_by_key = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            try:
                turn = int(row["turn"])
                task_index = int(row["task_index"])
                value = float(row[value_column])
            except (KeyError, TypeError, ValueError):
                continue
            key = (task_index, row.get("from_label", ""), row.get("to_label", ""))
            labels_by_key[key] = f"{row.get('from_label', '')}->{row.get('to_label', '')}"
            records_by_turn[turn][key] = value

    turns = sorted(records_by_turn)
    if limit > 0:
        turns = turns[-limit:]
    keys = sorted({key for turn in turns for key in records_by_turn[turn]})
    series = {key: [records_by_turn[turn].get(key, float("nan")) for turn in turns] for key in keys}
    labels = {key: labels_by_key.get(key, str(key)) for key in keys}
    return turns, series, labels


def short_label(label: str) -> str:
    return label.replace("[", "").replace("]", "")


def plot_matrix(labels, rows, output_png: Path, title: str, ylabel: str):
    if not labels or not rows:
        return False

    output_png.parent.mkdir(parents=True, exist_ok=True)
    xs = list(range(len(rows)))
    fig, ax = plt.subplots(figsize=(14, 7))
    for index, label in enumerate(labels):
        ys = [row[index] if index < len(row) else float("nan") for row in rows]
        ax.plot(xs, ys, linewidth=1.5, marker=".", markersize=3, label=short_label(label))
    ax.set_title(title)
    ax.set_xlabel("recent turn")
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="best", fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(output_png, dpi=180)
    plt.close(fig)
    return True


def plot_series(config: SeriesPlotConfig):
    if not config.turns or not config.series:
        return False
    keys = config.selected_keys or sorted(config.series)
    if not keys:
        return False

    config.output_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(15, 8))
    for key in keys:
        ax.plot(
            config.turns,
            config.series[key],
            linewidth=1.2,
            marker=".",
            markersize=2.5,
            label=config.labels.get(key, str(key)),
        )
    ax.set_title(config.title)
    ax.set_xlabel("turn")
    ax.set_ylabel(config.ylabel)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="best", fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(config.output_png, dpi=180)
    plt.close(fig)
    return True


def average(values):
    valid = [value for value in values if value == value]
    if not valid:
        return float("nan")
    return sum(valid) / len(valid)


def top_average_keys(series, count: int):
    ranked = [(average(values), key) for key, values in series.items()]
    ranked = [(avg, key) for avg, key in ranked if avg == avg]
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [key for _, key in ranked[:count]], ranked[:count]


def load_l2_paths(path: Path):
    if not path or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if data.get("schema") != "disp_probe.l2_ring_fullmesh_paths.v1":
        return {}

    result = {}
    for src_label, dst_map in data.get("paths", {}).items():
        for dst_label, item in dst_map.items():
            hops = item.get("hops", [])
            sport = item.get("sport", "")
            result[(src_label, dst_label)] = {
                "sport": sport,
                "hops": hops,
            }
    return result


def find_l2_path_json(run_dir: Path, output_root: Path, explicit_path: Path = None):
    candidates = []
    if explicit_path:
        candidates.append(explicit_path)
    candidates.extend(
        [
            run_dir / "l2_fullmesh_path.json",
            output_root / "l2_fullmesh_path.json",
            output_root / "l2_path.json",
        ]
    )
    for path in candidates:
        if path and path.exists():
            return path
    recursive_candidates = [path for path in output_root.rglob("l2_fullmesh_path.json") if path.is_file()]
    if recursive_candidates:
        return max(recursive_candidates, key=lambda path: path.stat().st_mtime)
    return None


def format_l2_path(key, l2_paths):
    _, src_label, dst_label = key
    item = l2_paths.get((src_label, dst_label))
    if not item:
        return ""
    hops = item.get("hops", [])
    sport = item.get("sport", "")
    path = "->".join(hop for hop in hops if hop)
    if sport != "":
        return f"sport={sport};{path}"
    return path


def write_l2_top10(path: Path, ranked, labels, l2_paths):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("rank\tavg_l2_path_lat\tpair\tpath\n")
        for index, (avg_value, key) in enumerate(ranked, start=1):
            f.write(f"{index}\t{avg_value:.6f}\t{labels.get(key, str(key))}\t{format_l2_path(key, l2_paths)}\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Plot recent network status from probe_controller output")
    parser.add_argument("input_path", type=Path, nargs="?", default=None, help="Specific output/<time> run directory")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Root output directory")
    parser.add_argument("--input-dir", type=Path, default=None, help="Specific output/<time> run directory")
    parser.add_argument("--run-dir", type=Path, default=None, help="Alias of --input-dir")
    parser.add_argument("--limit", type=int, default=100, help="Number of recent turns to plot")
    parser.add_argument("--top-l2", type=int, default=10, help="Number of highest average L2 latency pairs to plot")
    parser.add_argument("--plot-dir", type=Path, default=None, help="Directory for generated figures")
    parser.add_argument("--l2-path-json", type=Path, default=None, help="Path to l2_fullmesh_path.json")
    return parser.parse_args()


def resolve_run_paths(args):
    run_dir = args.input_path or args.input_dir or args.run_dir or find_latest_run_dir(args.output_root)
    if not run_dir.exists() or not run_dir.is_dir():
        raise RuntimeError(f"input run directory unavailable: {run_dir}")
    plot_dir = args.plot_dir or (run_dir / "status_plot")
    l2_path_json = find_l2_path_json(run_dir, args.output_root, args.l2_path_json)
    return run_dir, plot_dir, l2_path_json


def plot_l1_status(run_dir, plot_dir, limit):
    l1_labels, l1_lat_rows = parse_l1_matrix(run_dir / "link_lat.txt", limit)
    _, l1_pass_rows = parse_l1_matrix(run_dir / "link_pass_rate.txt", limit)
    plot_matrix(l1_labels, l1_lat_rows, plot_dir / "l1_latency.png", "L1 latency, recent turns", "latency (ms)")
    plot_matrix(l1_labels, l1_pass_rows, plot_dir / "l1_passrate.png", "L1 pass rate, recent turns", "pass rate")


def plot_l2_status(run_dir, plot_dir, args, l2_path_json):
    l2_paths = load_l2_paths(l2_path_json) if l2_path_json else {}
    l2_dir = run_dir / "l2_status"
    turns, l2_pass_series, l2_pass_labels = parse_l2_table(l2_dir / "l2_path_passrate.txt", "pass_rate", args.limit)
    plot_series(
        SeriesPlotConfig(
            turns,
            l2_pass_series,
            l2_pass_labels,
            plot_dir / "l2_passrate.png",
            "L2 pass rate, recent turns",
            "pass rate",
        )
    )

    lat_turns, l2_lat_series, l2_lat_labels = parse_l2_table(l2_dir / "l2_path_lat.txt", "l2_path_lat", args.limit)
    top_keys, ranked = top_average_keys(l2_lat_series, args.top_l2)
    plot_series(
        SeriesPlotConfig(
            lat_turns,
            l2_lat_series,
            l2_lat_labels,
            plot_dir / "l2_latency_top10.png",
            f"L2 latency top {len(top_keys)} average pairs, recent turns",
            "latency (ms)",
            selected_keys=top_keys,
        )
    )
    write_l2_top10(plot_dir / "l2_latency_top10.txt", ranked, l2_lat_labels, l2_paths)


def log_outputs(run_dir, plot_dir, l2_path_json):
    logging.info("run_dir=%s", run_dir)
    logging.info("plot_dir=%s", plot_dir)
    if l2_path_json:
        logging.info("l2_path_json=%s", l2_path_json)
    for name in (
        "l1_latency.png",
        "l1_passrate.png",
        "l2_passrate.png",
        "l2_latency_top10.png",
        "l2_latency_top10.txt",
    ):
        path = plot_dir / name
        if path.exists():
            logging.info("wrote %s", path)


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    run_dir, plot_dir, l2_path_json = resolve_run_paths(args)
    plot_l1_status(run_dir, plot_dir, args.limit)
    plot_l2_status(run_dir, plot_dir, args, l2_path_json)
    log_outputs(run_dir, plot_dir, l2_path_json)


if __name__ == "__main__":
    main()
