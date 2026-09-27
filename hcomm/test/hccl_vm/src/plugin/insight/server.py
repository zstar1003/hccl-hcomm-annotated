#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import argparse
import inspect
import json
import mimetypes
import os
import re
import signal
import sys
import threading
import traceback
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PurePosixPath
from urllib.parse import parse_qs, unquote, urlparse

try:
    import yaml
except ImportError:
    yaml = None


INSIGHT_DIR = Path(__file__).resolve().parent
MANIFEST_PATH = INSIGHT_DIR / "manifest.json"


def log_line(level: str, component: str, message: str) -> None:
    frame = inspect.currentframe()
    caller = frame.f_back if frame else None
    if caller is not None:
        filename = Path(caller.f_code.co_filename).name
        lineno = caller.f_lineno
        func_name = caller.f_code.co_name
    else:
        filename = Path(__file__).name
        lineno = 0
        func_name = "unknown"

    pid = os.getpid()
    tid = threading.get_native_id()
    level_name = str(level).lower()
    print(
        f"[{level_name}][PID:{pid}][TID:{tid}]"
        f"[{filename}:{lineno}][{func_name}] [HVM] {component} : {message}",
        flush=True,
    )


def log_http_status(status_code: int) -> str:
    return "warn" if status_code < HTTPStatus.INTERNAL_SERVER_ERROR else "error"


def load_manifest_settings() -> dict:
    """Load settings from manifest.json. Logs an error and exits on failure."""
    if not MANIFEST_PATH.is_file():
        log_line("error", "BOOT", f"manifest.json not found: {MANIFEST_PATH}")
        sys.exit(1)
    try:
        with MANIFEST_PATH.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        log_line("error", "BOOT", f"Failed to parse manifest.json: {exc}")
        sys.exit(1)

    setting = manifest.get("setting")
    if not isinstance(setting, dict):
        log_line("error", "BOOT", "manifest.json is missing 'setting' section or it is not an object.")
        sys.exit(1)
    return setting


def _require_setting(settings: dict, key: str):
    if key not in settings:
        log_line("error", "BOOT", f"manifest.json 'setting' is missing required key: '{key}'")
        sys.exit(1)
    return settings[key]


_settings = load_manifest_settings()

DIST_DIR = (INSIGHT_DIR / _require_setting(_settings, "dist_path")).resolve()
DATA_DIR = (INSIGHT_DIR / _require_setting(_settings, "data_path")).resolve()
TOPO_CONFIG_DIR = (INSIGHT_DIR / _require_setting(_settings, "topo_config_path")).resolve() if "topo_config_path" in _settings else None
DEFAULT_PORT = int(_require_setting(_settings, "port"))
RANK_PATTERN = re.compile(r"rank_(\d+)\.msgpack$")
MEMORY_RANK_DIR_PATTERN = re.compile(r"rank_(\d+)$")
MAX_PATH_SEGMENT_LENGTH = 255
CHECKER_V3_STAGE_NAME = "input_graph"
CHECKER_V3_GRAPH_FILES = {"graph.msgpack", "layout.msgpack"}


class InsightServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, server_address: tuple[str, int], request_handler_class):
        super().__init__(server_address, request_handler_class)
        self._shutdown_requested = threading.Event()

    def request_shutdown(self) -> None:
        if self._shutdown_requested.is_set():
            return
        self._shutdown_requested.set()
        self.shutdown()


def safe_join(base: Path, *parts: str) -> Path:
    candidate = (base.joinpath(*parts)).resolve()
    candidate.relative_to(base.resolve())
    return candidate


def ensure_single_path_part(value: str, label: str) -> str:
    if value in {"", ".", ".."}:
        raise ValueError(f"{label} is invalid.")
    if len(value) > MAX_PATH_SEGMENT_LENGTH:
        raise ValueError(f"{label} is too long.")
    if any(separator in value for separator in ("/", "\\")):
        raise ValueError(f"{label} must be a single path segment.")
    return value


def parse_rank(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Query parameter 'rank' must be an integer.") from exc


def extract_rank_from_filename(name: str) -> int | None:
    match = RANK_PATTERN.fullmatch(name)
    return int(match.group(1)) if match else None


def extract_rank_from_dir(name: str) -> int | None:
    match = MEMORY_RANK_DIR_PATTERN.fullmatch(name)
    return int(match.group(1)) if match else None


def read_json_file(file_path: Path) -> dict:
    with file_path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"JSON file must contain an object: {file_path}")
    return data


def read_run_manifest(dataset_dir: Path) -> dict | None:
    manifest_path = dataset_dir / "manifest.json"
    if not manifest_path.is_file():
        return None
    return read_json_file(manifest_path)


def extract_checker_v3_ranks(run_manifest: dict | None) -> list[int]:
    if not isinstance(run_manifest, dict):
        return []

    graph_stage_stats = run_manifest.get("graph_stage_stats")
    if isinstance(graph_stage_stats, dict):
        for stage_name in ("input_graph", "v3_final_graph", "origin_graph", "revamp_graph"):
            stage_stats = graph_stage_stats.get(stage_name)
            if not isinstance(stage_stats, dict):
                continue

            stage_ranks = stage_stats.get("ranks")
            if isinstance(stage_ranks, list):
                rank_ids = sorted(
                    {
                        rank_info.get("rank_id")
                        for rank_info in stage_ranks
                        if isinstance(rank_info, dict) and isinstance(rank_info.get("rank_id"), int)
                    }
                )
                if rank_ids:
                    return rank_ids

            rank_count = stage_stats.get("rank_count")
            if isinstance(rank_count, int) and rank_count > 0:
                return list(range(rank_count))

    op_param = run_manifest.get("op_param")
    rank_size = op_param.get("rank_size") if isinstance(op_param, dict) else None
    if isinstance(rank_size, int) and rank_size > 0:
        return list(range(rank_size))

    return []


def build_dataset_summary(dataset_dir: Path) -> dict:
    manifest_exists = (dataset_dir / "manifest.json").is_file()
    try:
        run_manifest = read_run_manifest(dataset_dir) or {}
        manifest_error = None
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        run_manifest = {}
        manifest_error = str(exc)

    return {
        "name": dataset_dir.name,
        "data_id": run_manifest.get("data_id", dataset_dir.name),
        "success": run_manifest.get("success"),
        "op_param": run_manifest.get("op_param"),
        "ret_code": run_manifest.get("ret_code"),
        "error_count": run_manifest.get("error_count"),
        "has_manifest": manifest_exists,
        "manifest_error": manifest_error,
    }


def list_dataset_summaries() -> list[dict]:
    if not DATA_DIR.exists():
        return []

    dataset_dirs = sorted((item for item in DATA_DIR.iterdir() if item.is_dir()), key=lambda item: item.name)
    return [build_dataset_summary(dataset_dir) for dataset_dir in dataset_dirs]


def build_dataset_manifest(dataset: str) -> dict:
    dataset = ensure_single_path_part(dataset, "Dataset name")
    dataset_dir = safe_join(DATA_DIR, dataset)
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset not found: {dataset}")

    graph_dir = dataset_dir / "graph"
    memory_dir = dataset_dir / "memory"
    validation_dir = dataset_dir / "validation"

    memory_ranks = []
    if memory_dir.exists():
        for rank_dir in sorted((item for item in memory_dir.iterdir() if item.is_dir()), key=lambda item: item.name):
            rank = extract_rank_from_dir(rank_dir.name)
            memory_ranks.append(
                {
                    "name": rank_dir.name,
                    "rank": rank,
                    "files": sorted(file.name for file in rank_dir.glob("*.msgpack")),
                }
            )

    try:
        run_manifest = read_run_manifest(dataset_dir)
        manifest_error = None
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        run_manifest = None
        manifest_error = str(exc)

    checker_v3_ranks = extract_checker_v3_ranks(run_manifest)

    graph_groups = []
    if graph_dir.exists():
        top_level_msgpack_files = sorted(file.name for file in graph_dir.glob("*.msgpack"))
        if CHECKER_V3_GRAPH_FILES.issubset(set(top_level_msgpack_files)):
            synthetic_ranks = sorted(
                rank_info["rank"] for rank_info in memory_ranks if isinstance(rank_info.get("rank"), int)
            ) or checker_v3_ranks
            graph_groups.append(
                {
                    "name": CHECKER_V3_STAGE_NAME,
                    "files": top_level_msgpack_files,
                    "has_index": False,
                    "ranks": synthetic_ranks,
                    "is_checker_v3": True,
                }
            )

        for group_dir in sorted((item for item in graph_dir.iterdir() if item.is_dir()), key=lambda item: item.name):
            msgpack_files = sorted(file.name for file in group_dir.glob("*.msgpack"))
            ranks = sorted(
                rank for rank in (extract_rank_from_filename(name) for name in msgpack_files) if rank is not None
            )
            graph_groups.append(
                {
                    "name": group_dir.name,
                    "files": msgpack_files,
                    "has_index": "index.msgpack" in msgpack_files,
                    "ranks": ranks,
                    "is_checker_v3": False,
                }
            )

    validation_files = sorted(file.name for file in validation_dir.glob("*.msgpack")) if validation_dir.exists() else []

    return {
        "name": dataset,
        "data_id": (run_manifest or {}).get("data_id", dataset),
        "children": sorted(item.name for item in dataset_dir.iterdir() if item.is_dir()),
        "run_manifest": run_manifest,
        "manifest_error": manifest_error,
        "op_param": (run_manifest or {}).get("op_param"),
        "success": (run_manifest or {}).get("success"),
        "ret_code": (run_manifest or {}).get("ret_code"),
        "error_count": (run_manifest or {}).get("error_count"),
        "graph": graph_groups,
        "memory": memory_ranks,
        "validation": validation_files,
    }


def resolve_msgpack_path(dataset: str, kind: str, rank: int | None, group: str | None, file_name: str | None) -> Path:
    dataset = ensure_single_path_part(dataset, "Dataset name")
    dataset_dir = safe_join(DATA_DIR, dataset)
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset not found: {dataset}")

    if kind == "graph":
        if not group:
            raise ValueError("Query parameter 'group' is required when kind=graph.")
        group = ensure_single_path_part(group, "Graph group")
        graph_group_dir = safe_join(dataset_dir, "graph", group)
        if not graph_group_dir.is_dir():
            raise FileNotFoundError(f"Graph group not found: {group}")
        resolved_name = file_name or ("index.msgpack" if rank is None else f"rank_{rank}.msgpack")
        file_path = safe_join(graph_group_dir, resolved_name)
    elif kind == "memory":
        if rank is None:
            raise ValueError("Query parameter 'rank' is required when kind=memory.")
        memory_rank_dir = safe_join(dataset_dir, "memory", f"rank_{rank}")
        if not memory_rank_dir.is_dir():
            raise FileNotFoundError(f"Memory rank not found: rank_{rank}")
        resolved_name = file_name or "manifest.msgpack"
        file_path = safe_join(memory_rank_dir, resolved_name)
    elif kind == "validation":
        resolved_name = file_name or "issues.msgpack"
        file_path = safe_join(dataset_dir, "validation", resolved_name)
    else:
        raise ValueError("Query parameter 'kind' must be one of: graph, memory, validation.")

    if not file_path.is_file():
        raise FileNotFoundError(f"Msgpack file not found: {file_path.name}")

    return file_path


def resolve_dataset_file_path(dataset: str, relative_path: str) -> Path:
    dataset = ensure_single_path_part(dataset, "Dataset name")
    dataset_dir = safe_join(DATA_DIR, dataset)
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset not found: {dataset}")

    requested_path = PurePosixPath(unquote(relative_path))
    parts = [part for part in requested_path.parts if part not in ("", "/")]
    if not parts:
        raise ValueError("Query parameter 'path' is required.")
    if any(part in {".", ".."} for part in parts):
        raise ValueError("Query parameter 'path' is invalid.")
    if any(len(part) > MAX_PATH_SEGMENT_LENGTH for part in parts):
        raise ValueError("Query parameter 'path' is too long.")

    file_path = safe_join(dataset_dir, *parts)
    if not file_path.is_file():
        raise FileNotFoundError(f"File not found: {requested_path.as_posix()}")
    return file_path


def list_available_topologies() -> list[dict]:
    if TOPO_CONFIG_DIR is None or not TOPO_CONFIG_DIR.is_dir():
        return []
    results = []
    for f in sorted(TOPO_CONFIG_DIR.iterdir()):
        if f.is_file() and f.suffix in (".yaml", ".yml"):
            results.append({"name": f.stem, "filename": f.name, "path": str(f)})
    return results


def load_cluster_topology(topo_name: str) -> dict:
    if TOPO_CONFIG_DIR is None or not TOPO_CONFIG_DIR.is_dir():
        raise FileNotFoundError("Topology config directory not configured")
    if yaml is None:
        raise RuntimeError("PyYAML is required for topology loading: pip install pyyaml")

    topo_name = ensure_single_path_part(topo_name, "Topology name")
    candidates = [TOPO_CONFIG_DIR / topo_name, TOPO_CONFIG_DIR / (topo_name + ".yaml"), TOPO_CONFIG_DIR / (topo_name + ".yml")]
    cluster_path = None
    for c in candidates:
        if c.is_file():
            cluster_path = c
            break
    if cluster_path is None:
        raise FileNotFoundError(f"Cluster topology config not found: {topo_name}")

    with cluster_path.open("r", encoding="utf-8") as f:
        cluster_config = yaml.safe_load(f)

    server_list_raw = cluster_config.get("server_list", [])
    super_node_list_raw = cluster_config.get("super_node_list", [])
    is_new_format = bool(server_list_raw and not super_node_list_raw)

    sp_servers = {}
    if is_new_format:
        for entry in server_list_raw:
            sp_id = entry.get("super_pod_id", 0)
            ids = _expand_id_range(entry)
            srv_topo = entry.get("server_topo", "")
            soc_version = entry.get("soc_version", "")
            for sid in ids:
                sp_servers.setdefault(sp_id, []).append({
                    "id": sid, "server_topo": srv_topo, "soc_version": soc_version
                })
    else:
        for entry in super_node_list_raw:
            ids = _expand_id_range(entry)
            sp_topo = entry.get("superpod_topo", "")
            for sid in ids:
                sp_yaml_path = _find_superpod_yaml(sp_topo)
                if sp_yaml_path is None:
                    continue
                with sp_yaml_path.open("r", encoding="utf-8") as f:
                    sp_config = yaml.safe_load(f)
                sp_server_list = sp_config.get("server_list", [])
                expanded = []
                for srv_entry in sp_server_list:
                    srv_ids = _expand_id_range(srv_entry)
                    srv_topo = srv_entry.get("server_topo", "")
                    soc_version = srv_entry.get("soc_version", "")
                    for s in srv_ids:
                        expanded.append({"id": s, "server_topo": srv_topo, "soc_version": soc_version})
                sp_servers[sid] = expanded

    server_topo_cache = {}
    for sp_id, servers in sp_servers.items():
        for srv in servers:
            topo_name_key = srv["server_topo"]
            if topo_name_key not in server_topo_cache:
                server_topo_cache[topo_name_key] = _load_server_topo(topo_name_key)

    return {
        "name": cluster_config.get("name", ""),
        "description": cluster_config.get("description", ""),
        "super_node_num": cluster_config.get("super_node_num", 0),
        "server_num": cluster_config.get("server_num", 0),
        "is_new_format": is_new_format,
        "super_pods": {str(k): v for k, v in sp_servers.items()},
        "server_topo_configs": server_topo_cache,
    }


def _expand_id_range(entry: dict) -> list[int]:
    ids = []
    if "id" in entry:
        raw = entry["id"]
        if isinstance(raw, list):
            ids.extend(int(x) for x in raw)
        else:
            ids.append(int(raw))
    if "id_range" in entry:
        rng = entry["id_range"]
        if isinstance(rng, list) and len(rng) == 2:
            ids.extend(range(int(rng[0]), int(rng[1]) + 1))
        elif isinstance(rng, list):
            ids.extend(int(x) for x in rng)
    return ids


def _find_superpod_yaml(topo_name: str) -> Path | None:
    if not topo_name:
        return None
    if TOPO_CONFIG_DIR is None:
        return None
    sp_dir = TOPO_CONFIG_DIR.parent / "super_pod"
    candidates = [
        Path(topo_name) if Path(topo_name).is_file() else None,
        sp_dir / topo_name if (sp_dir / topo_name).is_file() else None,
        sp_dir / (topo_name + ".yaml") if (sp_dir / (topo_name + ".yaml")).is_file() else None,
    ]
    for c in candidates:
        if c is not None:
            return c
    return None


def _load_server_topo(topo_name: str) -> dict | None:
    if yaml is None:
        return None
    if TOPO_CONFIG_DIR is None:
        return None
    srv_dir = TOPO_CONFIG_DIR.parent / "server_or_pod"
    candidates = [
        srv_dir / topo_name if (srv_dir / topo_name).is_file() else None,
        srv_dir / (topo_name + ".yaml") if (srv_dir / (topo_name + ".yaml")).is_file() else None,
    ]
    for c in candidates:
        if c is not None:
            with c.open("r", encoding="utf-8") as f:
                return yaml.safe_load(f)
    return None


class BackendRequestHandler(BaseHTTPRequestHandler):
    server_version = "HVRMInsightBackend/0.1"

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

        try:
            if path == "/api/health":
                self.send_json({"status": "ok"})
                return

            if path in {"/api/data-files", "/api/datasets"}:
                self.handle_dataset_list()
                return

            if path.startswith("/api/datasets/"):
                dataset = unquote(path.removeprefix("/api/datasets/")).strip("/")
                self.handle_dataset_manifest(dataset)
                return

            if path == "/api/msgpack":
                self.handle_msgpack(parsed.query)
                return

            if path == "/api/file":
                self.handle_dataset_file(parsed.query)
                return

            if path == "/api/topology/list":
                self.handle_topology_list()
                return

            if path.startswith("/api/topology/cluster/"):
                topo_name = unquote(path.removeprefix("/api/topology/cluster/")).strip("/")
                self.handle_topology_cluster(topo_name)
                return

            self.serve_static(path)
        except FileNotFoundError as exc:
            self.log_error_request(HTTPStatus.NOT_FOUND, str(exc))
            self.send_json({"error": str(exc)}, status=HTTPStatus.NOT_FOUND)
        except ValueError as exc:
            self.log_error_request(HTTPStatus.BAD_REQUEST, str(exc))
            self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
        except OSError:
            self.log_error_request(HTTPStatus.BAD_REQUEST, "Invalid path or file name.")
            self.send_json({"error": "Invalid path or file name."}, status=HTTPStatus.BAD_REQUEST)
        except Exception as exc:  # pragma: no cover - defensive fallback
            self.log_exception(exc)
            self.send_json({"error": f"Internal server error: {exc}"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_HEAD(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Allow", "GET, HEAD, OPTIONS")
        self.end_headers()

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Allow", "GET, HEAD, OPTIONS")
        self.send_header("Access-Control-Allow-Methods", "GET, HEAD, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, format: str, *args) -> None:
        return

    def log_request(self, code: int | str = "-", size: int | str = "-") -> None:
        try:
            status_code = int(code)
        except (TypeError, ValueError):
            status_code = None

        if status_code is None:
            return
        if status_code is not None and status_code < HTTPStatus.BAD_REQUEST:
            return

        log_line(log_http_status(status_code), "HTTP", (
            f"{self.command} {self.path} from {self.address_string()} "
            f"status={code} size={size}"
        ))

    def log_error_request(self, status: HTTPStatus, message: str) -> None:
        log_line(log_http_status(int(status)), "HTTP", (
            f"{self.command} {self.path} status={int(status)} - {message}"
        ))

    def log_exception(self, exc: Exception) -> None:
        log_line("error", "HTTP", f"{self.command} {self.path} status=500 - {exc}")
        traceback.print_exc()

    def handle_dataset_list(self) -> None:
        datasets = list_dataset_summaries()
        self.send_json({"datasets": datasets})

    def handle_dataset_manifest(self, dataset: str) -> None:
        if not dataset:
            raise ValueError("Dataset name is required.")
        manifest = build_dataset_manifest(dataset)
        self.send_json(manifest)

    def handle_msgpack(self, query: str) -> None:
        params = parse_qs(query)
        dataset = self.first_param(params, "dataset")
        kind = self.first_param(params, "kind")
        rank = parse_rank(self.first_param(params, "rank"))
        group = self.first_param(params, "group")
        file_name = self.first_param(params, "file")

        if not dataset:
            raise ValueError("Query parameter 'dataset' is required.")
        if not kind:
            raise ValueError("Query parameter 'kind' is required.")

        file_path = resolve_msgpack_path(dataset=dataset, kind=kind, rank=rank, group=group, file_name=file_name)
        self.send_file(file_path, content_type="application/msgpack")

    def handle_dataset_file(self, query: str) -> None:
        params = parse_qs(query)
        dataset = self.first_param(params, "dataset")
        relative_path = self.first_param(params, "path")

        if not dataset:
            raise ValueError("Query parameter 'dataset' is required.")
        if not relative_path:
            raise ValueError("Query parameter 'path' is required.")

        file_path = resolve_dataset_file_path(dataset, relative_path)
        self.send_file(file_path)

    def handle_topology_list(self) -> None:
        topologies = list_available_topologies()
        self.send_json({"topologies": topologies})

    def handle_topology_cluster(self, topo_name: str) -> None:
        data = load_cluster_topology(topo_name)
        self.send_json(data)

    def serve_static(self, raw_path: str) -> None:
        requested_path = PurePosixPath(unquote(raw_path))
        parts = [part for part in requested_path.parts if part not in ("", "/")]

        if not parts:
            file_path = DIST_DIR / "index.html"
            self.send_file(file_path)
            return

        try:
            file_path = safe_join(DIST_DIR, *parts)
        except ValueError:
            self.send_json({"error": "Invalid static path."}, status=HTTPStatus.BAD_REQUEST)
            return

        if file_path.is_dir():
            index_file = file_path / "index.html"
            if index_file.is_file():
                self.send_file(index_file)
                return

        if file_path.is_file():
            self.send_file(file_path)
            return

        if "." in parts[-1]:
            raise FileNotFoundError(f"Static asset not found: {parts[-1]}")

        self.send_file(DIST_DIR / "index.html")

    def send_json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_file(self, file_path: Path, content_type: str | None = None) -> None:
        if not file_path.is_file():
            raise FileNotFoundError(f"File not found: {file_path.name}")

        body = file_path.read_bytes()
        guessed_type, _ = mimetypes.guess_type(str(file_path))
        resolved_type = content_type or guessed_type or "application/octet-stream"

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", resolved_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    @staticmethod
    def first_param(params: dict[str, list[str]], key: str) -> str | None:
        values = params.get(key)
        return values[0] if values else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal local backend for hvrm_insight.")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to. Default: 127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"Port to bind to. Default: {DEFAULT_PORT}")
    return parser.parse_args()


def format_access_url(host: str, port: int) -> str:
    display_host = "localhost" if host in {"0.0.0.0", "127.0.0.1", "::", "::1"} else host
    return f"http://{display_host}:{port}"


def print_running_url(host: str, port: int) -> None:
    url = format_access_url(host, port)
    log_line("info", "BOOT", f"Running at {url}")


def process_plugin_message(line: str, server: InsightServer) -> None:
    try:
        message = json.loads(line)
    except json.JSONDecodeError as exc:
        log_line("error", "PLUGIN", f"Invalid command payload: {exc}")
        return

    action = message.get("action", "")
    payload = message.get("payload", {})
    if not isinstance(payload, dict):
        payload = {}

    if action == "status":
        status = payload.get("status", "")
        if status == "finish":
            print_running_url(server.server_address[0], server.server_address[1])
        return

    if action == "stop":
        server.request_shutdown()
        return


def stdin_command_loop(server: InsightServer) -> None:
    try:
        for raw_line in sys.stdin:
            line = raw_line.strip()
            if not line:
                continue
            process_plugin_message(line, server)
            if server._shutdown_requested.is_set():
                return
    except Exception as exc:  # pragma: no cover - defensive fallback
        log_line("error", "PLUGIN", f"stdin loop failed: {exc}")
        traceback.print_exc()
    finally:
        server.request_shutdown()


def main() -> None:
    if not DIST_DIR.is_dir():
        raise FileNotFoundError(f"Missing dist directory: {DIST_DIR}")

    args = parse_args()
    server = InsightServer((args.host, args.port), BackendRequestHandler)

    stdin_thread = threading.Thread(target=stdin_command_loop, args=(server,), daemon=True)
    stdin_thread.start()

    def handle_shutdown_signal(_signum, _frame) -> None:
        threading.Thread(target=server.request_shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, handle_shutdown_signal)

    print_running_url(args.host, args.port)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.request_shutdown()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
