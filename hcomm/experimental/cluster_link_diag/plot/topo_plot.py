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
import json
import logging
import os
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ALLPATH_JSON = PROJECT_ROOT / "output" / "allpath.json"
DEFAULT_LLDP_TOPO_JSON = PROJECT_ROOT / "output" / "probe_topo_lldp.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "topo_plot"


class UnionFind:
    def __init__(self, items):
        self.parent = {item: item for item in items}

    def find(self, item):
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != item:
            parent = self.parent[item]
            self.parent[item] = root
            item = parent
        return root

    def union(self, left, right):
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def load_allpath(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if data.get("schema") != "disp_probe.allpath.v1":
        raise RuntimeError(f"{path} is not disp_probe.allpath.v1")
    return data


def load_lldp_topo(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if "mesh_map" not in data:
        raise RuntimeError(f"{path} does not contain mesh_map")
    return data


def extract_lldp_tor_groups(lldp_topo: dict, label_by_ip: dict):
    groups = []
    for mesh_name, mesh in sorted(lldp_topo.get("mesh_map", {}).items()):
        group = build_lldp_tor_group(mesh_name, mesh, label_by_ip)
        if group:
            groups.append(group)
    groups.sort(key=lambda group: dev_sort_key(group["members"][0]))
    return groups


def build_lldp_tor_group(mesh_name, mesh, label_by_ip):
    members = []
    switch_ips = set()
    kid_to_link = mesh.get("kid_to_link", {})
    for device_ip, links in kid_to_link.items():
        label = label_by_ip.get(device_ip)
        if label:
            members.append(label)
        switch_ips.update(extract_switch_ips(links))
    if not members:
        return None
    return {
        "mesh_name": mesh_name,
        "members": sorted(members, key=dev_sort_key),
        "switch_ips": sorted(switch_ips, key=ip_sort_key),
    }


def extract_switch_ips(links):
    return {link[1] for link in links if len(link) >= 2 and link[1]}


def sample_is_same_leaf(sample: dict, dst_ip: str) -> bool:
    hops = [hop for hop in sample.get("hops", []) if hop]
    return len(hops) == 2 and hops[-1] == dst_ip


def infer_leaf_groups(allpath: dict):
    devices = allpath.get("device_list", [])
    labels = [device["label"] for device in devices]
    ip_by_label = {device["label"]: device["device_ip"] for device in devices}
    uf = UnionFind(labels)

    for src_label, dst_map in allpath.get("paths", {}).items():
        for dst_label, samples in dst_map.items():
            dst_ip = ip_by_label.get(dst_label)
            if not dst_ip or not samples:
                continue
            same_leaf_samples = sum(1 for sample in samples if sample_is_same_leaf(sample, dst_ip))
            if same_leaf_samples == len(samples):
                uf.union(src_label, dst_label)

    grouped = defaultdict(list)
    for label in labels:
        grouped[uf.find(label)].append(label)

    groups = sorted(
        (sorted(group, key=dev_sort_key) for group in grouped.values()),
        key=lambda group: dev_sort_key(group[0]),
    )
    leaf_by_dev = {}
    for index, group in enumerate(groups):
        leaf_id = f"leaf{index}"
        for label in group:
            leaf_by_dev[label] = leaf_id
    return groups, leaf_by_dev


def leaf_groups_from_lldp(allpath: dict, lldp_topo: dict):
    label_by_ip = {device["device_ip"]: device["label"] for device in allpath.get("device_list", [])}
    tor_groups = extract_lldp_tor_groups(lldp_topo, label_by_ip)
    if not tor_groups:
        return None, None, None

    leaf_groups = [group["members"] for group in tor_groups]
    leaf_by_dev = {}
    leaf_meta = {}
    for index, group in enumerate(tor_groups):
        leaf_id = f"tor{index}"
        leaf_meta[leaf_id] = {
            "mesh_name": group["mesh_name"],
            "switch_ips": group["switch_ips"],
        }
        for label in group["members"]:
            leaf_by_dev[label] = leaf_id
    return leaf_groups, leaf_by_dev, leaf_meta


def dev_sort_key(label: str):
    if label.startswith("dev"):
        suffix = label[3:]
        parts = suffix.split("_", 1)
        if parts[0].isdigit():
            repeat_index = 1
            if len(parts) > 1 and parts[1].isdigit():
                repeat_index = int(parts[1])
            return (0, repeat_index, int(parts[0]), label)
    return (1, 0, 0, label)


def ip_sort_key(ip: str):
    parts = ip.split(".")
    if len(parts) == 4 and all(part.isdigit() for part in parts):
        return tuple(int(part) for part in parts)
    return (999, ip)


def ipv4_to_int(ip: str):
    parts = ip.split(".")
    if len(parts) != 4 or not all(part.isdigit() for part in parts):
        return None
    values = [int(part) for part in parts]
    if any(value < 0 or value > 255 for value in values):
        return None
    value = 0
    for part in values:
        value = value * 256 + part
    return value


def int_to_ipv4(value: int):
    return ".".join(str((value >> shift) & 0xFF) for shift in (24, 16, 8, 0))


def short_ip_range(ips):
    if not ips:
        return ""
    values = sorted(ipv4_to_int(ip) for ip in ips)
    if any(value is None for value in values):
        return ",".join(sorted(ips))
    first = int_to_ipv4(values[0])
    last = int_to_ipv4(values[-1])
    first_parts = first.split(".")
    last_parts = last.split(".")
    if first_parts[:3] == last_parts[:3]:
        return f"{first}-{last_parts[3]}"
    return f"{first}..{last}"


def collect_port_slots(allpath: dict, label_by_ip: dict):
    intervals = collect_mid_intervals(allpath, label_by_ip)
    clusters = merge_mid_intervals(intervals)
    return build_port_slot_nodes(clusters)


def collect_mid_intervals(allpath, label_by_ip):
    intervals = {}
    for dst_map in allpath.get("paths", {}).values():
        for samples in dst_map.values():
            for sample in samples:
                update_mid_interval(intervals, sample, label_by_ip)
    return intervals


def update_mid_interval(intervals, sample, label_by_ip):
    hops = [hop for hop in sample.get("hops", []) if hop]
    mids = tuple(hop for hop in hops if hop not in label_by_ip and hop != "29.182.0.1")
    if len(mids) < 2:
        return
    values = [ipv4_to_int(ip) for ip in mids]
    if any(value is None for value in values):
        return
    intervals.setdefault(mids, (min(values), max(values), mids))


def merge_mid_intervals(intervals):
    clusters = []
    current = None
    for start, end, mids in sorted(intervals.values(), key=lambda item: (item[0], item[1], item[2])):
        current = append_mid_interval(clusters, current, start, end, mids)
    return clusters


def append_mid_interval(clusters, current, start, end, mids):
    if current is None or start > current["end"]:
        current = {"start": start, "end": end, "mid_pairs": [mids]}
        clusters.append(current)
        return current
    current["end"] = max(current["end"], end)
    current["mid_pairs"].append(mids)
    return current


def build_port_slot_nodes(clusters):
    slot_by_mids = {}
    slot_nodes = []
    for index, cluster in enumerate(clusters):
        slot_name = f"slot{index:02d}"
        slot_id = f"port_slot:{slot_name}"
        ips = sorted(
            {ip for mids in cluster["mid_pairs"] for ip in mids},
            key=lambda ip: ipv4_to_int(ip) if ipv4_to_int(ip) is not None else 10**20,
        )
        for mids in cluster["mid_pairs"]:
            slot_by_mids[mids] = slot_id
        slot_nodes.append(
            {
                "id": slot_id,
                "type": "port_slot",
                "label": f"{slot_name}\n{short_ip_range(ips)}",
                "slot": slot_name,
                "ips": ips,
                "mid_pairs": sorted(cluster["mid_pairs"], key=lambda pair: tuple(ip_sort_key(ip) for ip in pair)),
            }
        )
    return slot_by_mids, slot_nodes


def resolve_leaf_groups(allpath, lldp_topo):
    leaf_meta = {}
    if lldp_topo:
        leaf_groups, leaf_by_dev, leaf_meta = leaf_groups_from_lldp(allpath, lldp_topo)
    else:
        leaf_groups = leaf_by_dev = None
    if leaf_groups is None or leaf_by_dev is None:
        leaf_groups, leaf_by_dev = infer_leaf_groups(allpath)
    return leaf_groups, leaf_by_dev, leaf_meta


def add_edge(edges, left, right, sport):
    if left == right:
        return
    key = tuple(sorted((left, right)))
    if key not in edges:
        edges[key] = {"nodes": key, "sports": set()}
    if sport is not None:
        edges[key]["sports"].add(sport)


def add_device_nodes(nodes, edges, devices, leaf_by_dev, server_by_label):
    for device in devices:
        label = device["label"]
        node_id = f"device:{label}"
        nodes[node_id] = {
            "id": node_id,
            "type": "device",
            "label": label,
            "device_ip": device["device_ip"],
            "leaf": leaf_by_dev[label],
            "server_ip": server_by_label.get(label, ""),
        }
        add_edge(edges, node_id, f"leaf:{leaf_by_dev[label]}", None)


def add_leaf_nodes(nodes, leaf_groups, leaf_meta, lldp_topo):
    for group_index, group in enumerate(leaf_groups):
        leaf_id = f"tor{group_index}" if lldp_topo else f"leaf{group_index}"
        switch_ips = leaf_meta.get(leaf_id, {}).get("switch_ips", [])
        label = f"ToR{group_index}"
        if switch_ips:
            label += "\n" + ",".join(switch_ips)
        label += "\n" + ",".join(group)
        nodes[f"leaf:{leaf_id}"] = {
            "id": f"leaf:{leaf_id}",
            "type": "leaf",
            "label": label,
            "members": group,
            "switch_ips": switch_ips,
        }


def add_path_edges(edges, allpath, context):
    for src_label, dst_map in allpath.get("paths", {}).items():
        src_leaf = f"leaf:{context['leaf_by_dev'][src_label]}"
        for dst_label, samples in dst_map.items():
            add_sample_edges(edges, samples, src_leaf, dst_label, context)


def add_sample_edges(edges, samples, src_leaf, dst_label, context):
    dst_leaf = f"leaf:{context['leaf_by_dev'][dst_label]}"
    dst_ip = context["ip_by_label"].get(dst_label, "")
    for sample in samples:
        hops = [hop for hop in sample.get("hops", []) if hop]
        if not hops:
            continue
        sport = sample.get("sport")
        intermediate_hops = [hop for hop in hops if hop not in context["label_by_ip"] and hop != "29.182.0.1"]
        if not intermediate_hops:
            add_edge(edges, src_leaf, dst_leaf, sport)
            continue
        slot_node = context["slot_by_mids"].get(tuple(intermediate_hops))
        if slot_node:
            add_edge(edges, src_leaf, slot_node, sport)
            if hops[-1] == dst_ip:
                add_edge(edges, slot_node, dst_leaf, sport)


def build_topology(allpath: dict, lldp_topo=None):
    devices = allpath.get("device_list", [])
    ip_by_label = {device["label"]: device["device_ip"] for device in devices}
    label_by_ip = {device["device_ip"]: device["label"] for device in devices}
    server_by_label = make_server_by_label(allpath)
    leaf_groups, leaf_by_dev, leaf_meta = resolve_leaf_groups(allpath, lldp_topo)
    slot_by_mids, slot_nodes = collect_port_slots(allpath, label_by_ip)

    nodes = {}
    edges = {}
    add_device_nodes(nodes, edges, devices, leaf_by_dev, server_by_label)
    add_leaf_nodes(nodes, leaf_groups, leaf_meta, lldp_topo)

    for node in slot_nodes:
        nodes[node["id"]] = node

    path_context = {
        "leaf_by_dev": leaf_by_dev,
        "ip_by_label": ip_by_label,
        "label_by_ip": label_by_ip,
        "slot_by_mids": slot_by_mids,
    }
    add_path_edges(edges, allpath, path_context)
    return nodes, list(edges.values()), leaf_groups


def make_server_by_label(allpath: dict):
    server_by_label = {}
    repeat_count = defaultdict(int)
    scope = allpath.get("scope", {})
    for server_ip, device_ids in scope.items():
        for device_id in device_ids:
            device_id = str(device_id)
            repeat_count[device_id] += 1
            control_dev = device_id
            if repeat_count[device_id] > 1:
                control_dev = f"{device_id}_{repeat_count[device_id]}"
            server_by_label[f"dev{control_dev}"] = server_ip
    return server_by_label


def _collect_layout_nodes(nodes):
    devices_by_leaf = defaultdict(list)
    devices_by_server = defaultdict(list)
    leaf_nodes = []
    slot_nodes = []
    for node in nodes.values():
        if node["type"] == "device":
            devices_by_leaf[node["leaf"]].append(node)
            devices_by_server[node.get("server_ip") or "unknown-server"].append(node)
        elif node["type"] == "leaf":
            leaf_nodes.append(node)
        elif node["type"] == "port_slot":
            slot_nodes.append(node)
    return devices_by_leaf, devices_by_server, leaf_nodes, slot_nodes


def _layout_leaf_nodes(leaf_nodes, positions):
    leaf_nodes.sort(key=lambda node: node["id"])
    leaf_count = max(1, len(leaf_nodes))
    leaf_x = {}
    for index, node in enumerate(leaf_nodes):
        x = 0.0 if leaf_count == 1 else index / (leaf_count - 1)
        leaf_x[node["id"].split(":", 1)[1]] = x
        positions[node["id"]] = (x, 0.9)
    return leaf_x


def _layout_server_devices(devices_by_server, positions):
    server_items = sorted(devices_by_server.items(), key=lambda item: ip_sort_key(item[0]))
    server_count = max(1, len(server_items))
    for server_index, (server_ip, server_devices) in enumerate(server_items):
        _layout_one_server(server_index, server_count, server_devices, positions)


def _server_row_bounds(server_index, server_count):
    server_left = server_index / server_count
    server_right = (server_index + 1) / server_count
    server_width = server_right - server_left
    inner_margin = min(0.05, server_width * 0.16)
    return server_left + inner_margin, server_right - inner_margin


def _layout_one_server(server_index, server_count, server_devices, positions):
    row_left, row_right = _server_row_bounds(server_index, server_count)
    leaf_ids = sorted({node["leaf"] for node in server_devices})
    row_context = {"left": row_left, "right": row_right, "leaf_count": max(1, len(leaf_ids)), "positions": positions}
    for leaf_index, leaf_id in enumerate(leaf_ids):
        devices = [node for node in server_devices if node["leaf"] == leaf_id]
        _layout_server_leaf_devices(devices, leaf_index, row_context)


def _layout_server_leaf_devices(devices, leaf_index, row_context):
    devices.sort(key=lambda node: dev_sort_key(node["label"]))
    count = len(devices)
    y = 0.0 if row_context["leaf_count"] == 1 else 0.18 - leaf_index * 0.36
    for index, node in enumerate(devices):
        x = (
            (row_context["left"] + row_context["right"]) / 2
            if count == 1
            else row_context["left"] + (row_context["right"] - row_context["left"]) * index / (count - 1)
        )
        row_context["positions"][node["id"]] = (x, y)


def _layout_serverless_devices(devices_by_leaf, positions, leaf_x):
    for leaf_id, devices in devices_by_leaf.items():
        devices.sort(key=lambda node: dev_sort_key(node["label"]))
        if f"leaf:{leaf_id}" not in positions:
            continue
        base_x = leaf_x.get(leaf_id, 0.5)
        count = len(devices)
        serverless = [node for node in devices if node["id"] not in positions]
        if serverless:
            span = 0.28
            for index, node in enumerate(serverless):
                offset = 0.0 if count == 1 else span * (index / (count - 1) - 0.5)
                positions[node["id"]] = (base_x + offset, 0.0)


def _layout_slot_nodes(slot_nodes, positions):
    slot_nodes.sort(key=lambda node: node["slot"])
    slot_count = len(slot_nodes)
    for index, node in enumerate(slot_nodes):
        x = 0.5 if slot_count == 1 else index / (slot_count - 1)
        positions[node["id"]] = (x, 1.75)


def layout(nodes: dict):
    devices_by_leaf, devices_by_server, leaf_nodes, slot_nodes = _collect_layout_nodes(nodes)
    positions = {}
    leaf_x = _layout_leaf_nodes(leaf_nodes, positions)
    _layout_server_devices(devices_by_server, positions)
    _layout_serverless_devices(devices_by_leaf, positions, leaf_x)
    _layout_slot_nodes(slot_nodes, positions)
    return positions


def draw(nodes: dict, edges: list, leaf_groups, output_png: Path):
    positions = layout(nodes)
    fig, ax = plt.subplots(figsize=(18, 11))
    ax.axis("off")
    port_slot_count = sum(1 for node in nodes.values() if node.get("type") == "port_slot")
    fig.suptitle(f"LLDP ToR topology: ToR={len(leaf_groups)}, inter-ToR slots={port_slot_count}", fontsize=14, y=0.98)
    xs = [position[0] for position in positions.values()]
    x_min = min(xs, default=0.0)
    x_max = max(xs, default=1.0)
    ax.set_xlim(x_min - 0.12, x_max + 0.12)
    ax.set_ylim(-0.72, 2.15)
    draw_edges(ax, edges, positions)
    draw_server_boxes(ax, nodes, positions)
    draw_nodes(ax, nodes, positions)
    save_figure(fig, output_png)


def draw_edges(ax, edges, positions):
    for edge in edges:
        left, right = edge["nodes"]
        if left not in positions or right not in positions:
            continue
        x1, y1 = positions[left]
        x2, y2 = positions[right]
        sports = edge["sports"]
        width = 0.8 if not sports else min(3.0, 0.8 + len(sports) / 16.0)
        ax.plot([x1, x2], [y1, y2], color="#7A7F87", linewidth=width, alpha=0.55, zorder=1)


def draw_nodes(ax, nodes, positions):
    style = {
        "device": {"color": "#4C78A8", "size": 700, "marker": "o"},
        "leaf": {"color": "#F58518", "size": 1500, "marker": "s"},
        "port_slot": {"color": "#54A24B", "size": 760, "marker": "D"},
    }
    default_style = {"color": "#7A7F87", "size": 640, "marker": "o"}
    for node in nodes.values():
        node_id = node.get("id")
        node_type = node.get("type", "")
        if node_id not in positions:
            continue
        x, y = positions[node_id]
        node_style = style.get(node_type, default_style)
        ax.scatter([x], [y], s=node_style["size"], c=node_style["color"], marker=node_style["marker"], zorder=2)
        label = node.get("label", "")
        if node_type == "device":
            label += f"\n{node.get('device_ip', '')}"
        if node_type == "port_slot":
            ax.text(x, y + 0.08, label, fontsize=6, ha="center", va="bottom", rotation=70)
        else:
            fontsize = 7 if node_type == "device" else 8
            ax.text(x, y - 0.06, label, fontsize=fontsize, ha="center", va="top")


def save_figure(fig, output_png):
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_png, dpi=180, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


def draw_server_boxes(ax, nodes: dict, positions: dict):
    devices_by_server = defaultdict(list)
    for node in nodes.values():
        if node.get("type") != "device":
            continue
        server_ip = node.get("server_ip") or "unknown-server"
        if node["id"] in positions:
            devices_by_server[server_ip].append(node)

    for server_ip, devices in sorted(devices_by_server.items(), key=lambda item: ip_sort_key(item[0])):
        xs = [positions[node["id"]][0] for node in devices]
        ys = [positions[node["id"]][1] for node in devices]
        if not xs or not ys:
            continue
        pad_x = 0.055
        pad_y = 0.13
        x0 = min(xs) - pad_x
        x1 = max(xs) + pad_x
        y0 = min(ys) - pad_y
        y1 = max(ys) + pad_y
        rect = Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            linewidth=1.2,
            edgecolor="#4C566A",
            facecolor="#ECEFF4",
            alpha=0.26,
            linestyle="--",
            zorder=0,
        )
        ax.add_patch(rect)
        ax.text(
            (x0 + x1) / 2,
            y0 - 0.045,
            f"server {server_ip}",
            fontsize=8,
            ha="center",
            va="top",
            color="#2E3440",
            zorder=3,
        )


def path_matrix(allpath: dict):
    labels = [device["label"] for device in allpath.get("device_list", [])]
    summary = allpath.get("summary", {})
    matrix = []
    for src in labels:
        row = []
        for dst in labels:
            if src == dst:
                row.append(None)
                continue
            value = summary.get(src, {}).get(dst, {}).get("unique_paths")
            if value is None:
                samples = allpath.get("paths", {}).get(src, {}).get(dst, [])
                value = len({tuple(sample.get("hops", [])) for sample in samples if sample.get("hops")})
            row.append(value)
        matrix.append(row)
    return labels, matrix


def draw_path_matrix(allpath: dict, output_png: Path):
    labels, matrix = path_matrix(allpath)
    numeric = [[0 if value is None else value for value in row] for row in matrix]
    max_value = max((value for row in numeric for value in row), default=1)

    fig_size = max(7, len(labels) * 0.75)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    image = ax.imshow(numeric, cmap="YlGnBu", vmin=0, vmax=max_value)

    ax.set_xticks(range(len(labels)), labels=labels)
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.tick_params(axis="x", labelrotation=45)
    ax.set_xlabel("dst")
    ax.set_ylabel("src")

    for row_index, row in enumerate(matrix):
        for col_index, value in enumerate(row):
            text = "-" if value is None else str(value)
            color = "white" if value is not None and value > max_value * 0.55 else "black"
            ax.text(col_index, row_index, text, ha="center", va="center", color=color, fontsize=11)

    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("unique paths")
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_png, dpi=180)
    plt.close(fig)


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Plot topology and path matrix from output/allpath.json")
    parser.add_argument("--input", type=Path, default=DEFAULT_ALLPATH_JSON, help="Path to allpath.json")
    parser.add_argument("--lldp-topo", type=Path, default=DEFAULT_LLDP_TOPO_JSON, help="Path to probe_topo_lldp.json")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for topo.png and path_matrix.png",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Legacy topo.png path; path_matrix.png is saved next to it",
    )
    args = parser.parse_args()

    allpath = load_allpath(args.input)
    lldp_topo = load_lldp_topo(args.lldp_topo) if args.lldp_topo and args.lldp_topo.exists() else None
    nodes, edges, leaf_groups = build_topology(allpath, lldp_topo)
    if args.output:
        output_dir = args.output.parent
        topo_png = args.output
    else:
        output_dir = args.output_dir
        topo_png = output_dir / "topo.png"
    matrix_png = output_dir / "path_matrix.png"

    draw(nodes, edges, leaf_groups, topo_png)
    draw_path_matrix(allpath, matrix_png)
    logging.info("%s_count=%s", "tor" if lldp_topo else "leaf", len(leaf_groups))
    for index, group in enumerate(leaf_groups):
        prefix = f"tor{index}" if lldp_topo else f"leaf{index}"
        logging.info("%s: %s", prefix, ",".join(group))
    slot_nodes = sorted((node for node in nodes.values() if node["type"] == "port_slot"), key=lambda node: node["slot"])
    logging.info("port_slot_count=%s", len(slot_nodes))
    for node in slot_nodes:
        logging.info("%s: %s", node["slot"], ",".join(node["ips"]))
    logging.info("wrote %s", topo_png)
    logging.info("wrote %s", matrix_png)


if __name__ == "__main__":
    main()
