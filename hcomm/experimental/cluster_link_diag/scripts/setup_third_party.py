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

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "third_party" / "manifest.json"
DEFAULT_PREFIX = Path("/usr/local/third_lib")
LOGGER = logging.getLogger("cluster_link_diag.setup_third_party")


class SetupError(RuntimeError):
    pass


def log(message: str) -> None:
    LOGGER.info("[third-party] %s", message)


def run(cmd: List[str], dry_run: bool) -> None:
    log("run: " + " ".join(cmd))
    if not dry_run:
        subprocess.run(cmd, check=True)


def check_output(cmd: List[str], dry_run: bool, cwd: Path | None = None) -> str:
    log("run: " + " ".join(cmd))
    if dry_run:
        return ""
    return subprocess.check_output(cmd, cwd=cwd, text=True).strip()


def load_manifest(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def resolve_repo_path(path: str) -> Path:
    return (PROJECT_ROOT / path).resolve()


def component_installed(component: Dict[str, Any], prefix: Path) -> bool:
    checks = component.get("verify") or []
    return all((prefix / item).exists() for item in checks)


def validate_git_fetch(component: Dict[str, Any]) -> Dict[str, Any]:
    fetch = component.get("fetch")
    if not fetch:
        raise SetupError(f"no fetch recipe for {component['name']}")
    if fetch.get("type") != "git":
        raise SetupError(f"unsupported fetch type for {component['name']}: {fetch.get('type')}")
    if shutil.which("git") is None:
        raise SetupError("git is required to fetch third-party sources")
    if fetch.get("build") != "cmake":
        raise SetupError(f"unsupported build recipe for {component['name']}: {fetch.get('build')}")
    return fetch


def prepare_component_dirs(component: Dict[str, Any], prefix: Path, dry_run: bool) -> tuple[Path, Path]:
    source_root = prefix / "src"
    build_root = prefix / "build"
    source_dir = source_root / f"{component['name']}-{component['version']}"
    build_dir = build_root / f"{component['name']}-{component['version']}"
    if not dry_run:
        source_root.mkdir(parents=True, exist_ok=True)
        build_root.mkdir(parents=True, exist_ok=True)
    return source_dir, build_dir


def clone_component_source(fetch: Dict[str, Any], source_dir: Path, dry_run: bool) -> None:
    if not source_dir.exists():
        run([
            "git",
            "clone",
            "--depth",
            "1",
            "--branch",
            fetch["tag"],
            fetch["url"],
            str(source_dir),
        ], dry_run)


def verify_component_commit(component: Dict[str, Any], fetch: Dict[str, Any], source_dir: Path, dry_run: bool) -> None:
    expected_commit = fetch.get("commit")
    if expected_commit:
        actual_commit = check_output(["git", "rev-parse", "HEAD"], dry_run, cwd=source_dir)
        if not dry_run and actual_commit.lower() != expected_commit.lower():
            raise SetupError(
                f"{component['name']} commit mismatch: expected {expected_commit}, got {actual_commit}"
            )
    else:
        log(f"warning: {component['name']} has no fetch.commit lock; trusting tag {fetch['tag']}")


def build_and_install_component(task: Dict[str, Any]) -> None:
    cmake_cmd = [
        "cmake",
        "-S",
        str(task["source_dir"]),
        "-B",
        str(task["build_dir"]),
        f"-DCMAKE_INSTALL_PREFIX={task['prefix']}",
        "-DCMAKE_BUILD_TYPE=Release",
    ]
    cmake_cmd.extend(task["fetch"].get("cmake_args", []))
    run(cmake_cmd, task["dry_run"])
    run(["cmake", "--build", str(task["build_dir"]), f"-j{task['jobs']}"], task["dry_run"])
    run(["cmake", "--install", str(task["build_dir"])], task["dry_run"])


def fetch_and_install_component(component: Dict[str, Any], prefix: Path, jobs: int, dry_run: bool) -> None:
    fetch = validate_git_fetch(component)
    source_dir, build_dir = prepare_component_dirs(component, prefix, dry_run)
    clone_component_source(fetch, source_dir, dry_run)
    verify_component_commit(component, fetch, source_dir, dry_run)
    build_task = {
        "fetch": fetch,
        "source_dir": source_dir,
        "build_dir": build_dir,
        "prefix": prefix,
        "jobs": jobs,
        "dry_run": dry_run,
    }
    build_and_install_component(build_task)
    if component["name"] == "CLI11":
        compat_header = prefix / "include" / "CLI11.hpp"
        if not dry_run and not compat_header.exists():
            compat_header.write_text("#pragma once\n#include <CLI/CLI.hpp>\n", encoding="utf-8")


def install_components(manifest: Dict[str, Any], prefix: Path, jobs: int, dry_run: bool) -> None:
    for component in manifest["components"]:
        required = bool(component.get("required", True))
        try:
            if not dry_run and component.get("verify") and component_installed(component, prefix):
                log(f"ok: {component['name']} is already installed")
                continue
            fetch_and_install_component(component, prefix, jobs, dry_run)
            if not dry_run and component.get("verify") and not component_installed(component, prefix):
                raise SetupError(f"installation verification failed for {component['name']}")
        except (SetupError, subprocess.CalledProcessError) as exc:
            if required:
                raise
            log(f"skip optional component: {component['name']} ({exc})")


def write_metadata(manifest_path: Path, requirements_path: Path, prefix: Path, dry_run: bool) -> None:
    metadata_dir = prefix / "share" / "disp_probe" / "third_party"
    if dry_run:
        log(f"write metadata under {metadata_dir}")
        return
    metadata_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(manifest_path, metadata_dir / "manifest.json")
    if requirements_path.is_file():
        shutil.copy2(requirements_path, metadata_dir / "requirements.txt")
    write_env_script(prefix, metadata_dir / "env.sh")


def write_env_script(prefix: Path, target: Path) -> None:
    content = f"""#!/usr/bin/env bash

export THIRDLIB_ROOT="${{THIRDLIB_ROOT:-{prefix}}}"
export ASCEND_HOME_PATH="${{ASCEND_HOME_PATH:-/usr/local/Ascend}}"
export ASCEND_CANN_PATH="${{ASCEND_CANN_PATH:-${{ASCEND_HOME_PATH}}/ascend-toolkit/latest/aarch64-linux}}"

_disp_probe_prepend_path() {{
    case ":${{!1:-}}:" in
        *":$2:"*) ;;
        *) export "$1=$2${{!1:+:${{!1}}}}" ;;
    esac
}}

_disp_probe_prepend_path PATH "${{THIRDLIB_ROOT}}/python/venv/bin"
_disp_probe_prepend_path CPATH "${{THIRDLIB_ROOT}}/include"
_disp_probe_prepend_path CPATH "${{THIRDLIB_ROOT}}/include/eigen3"
_disp_probe_prepend_path LIBRARY_PATH "${{THIRDLIB_ROOT}}/lib"
_disp_probe_prepend_path LD_LIBRARY_PATH "${{THIRDLIB_ROOT}}/lib"
_disp_probe_prepend_path LD_LIBRARY_PATH "${{ASCEND_CANN_PATH}}/lib64"
_disp_probe_prepend_path CMAKE_PREFIX_PATH "${{THIRDLIB_ROOT}}"

export MPLCONFIGDIR="${{MPLCONFIGDIR:-${{TMPDIR:-/tmp}}/disp_probe_matplotlib}}"
mkdir -p "${{MPLCONFIGDIR}}" 2>/dev/null || true

unset -f _disp_probe_prepend_path
"""
    target.write_text(content, encoding="utf-8")
    target.chmod(0o755)


def install_python(requirements: Path, prefix: Path, args: argparse.Namespace) -> None:
    if args.skip_python:
        log("skip python dependency installation")
        return
    if not requirements.is_file():
        raise SetupError(f"missing python requirements: {requirements}")

    venv_dir = prefix / "python" / "venv"
    cmd = [sys.executable, "-m", "venv"]
    if args.python_system_site_packages:
        cmd.append("--system-site-packages")
    cmd.append(str(venv_dir))
    run(cmd, args.dry_run)

    pip = venv_dir / "bin" / "pip"
    pip_cmd = [str(pip), "install", "-r", str(requirements)]
    if args.require_python_hashes:
        pip_cmd.append("--require-hashes")
    if args.offline:
        if not args.wheelhouse:
            raise SetupError("--offline requires --wheelhouse")
        pip_cmd.extend(["--no-index", "--find-links", str(args.wheelhouse)])
    run(pip_cmd, args.dry_run)


def path_exists_any(paths: Iterable[Path]) -> bool:
    return any(path.exists() for path in paths)


def cann_candidate_roots(ascend_home: Path, ascend_cann_path: Path | None) -> List[Path]:
    roots = []
    if ascend_cann_path is not None:
        roots.append(ascend_cann_path)
    roots.extend([
        ascend_home,
        ascend_home / "cann",
        ascend_home / "aarch64-linux",
        ascend_home / "ascend-toolkit" / "latest" / "aarch64-linux",
    ])
    toolkit_dir = ascend_home / "ascend-toolkit"
    if toolkit_dir.is_dir():
        roots.extend(sorted(toolkit_dir.glob("*/aarch64-linux"), reverse=True))

    unique_roots: List[Path] = []
    seen = set()
    for root in roots:
        key = str(root)
        if key not in seen:
            unique_roots.append(root)
            seen.add(key)
    return unique_roots


def check_prerequisites(prefix: Path) -> List[str]:
    missing: List[str] = []

    ascend_home = Path(os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend"))
    ascend_cann_env = os.environ.get("ASCEND_CANN_PATH")
    ascend_cann_path = Path(ascend_cann_env) if ascend_cann_env else None
    cann_roots = cann_candidate_roots(ascend_home, ascend_cann_path)
    cann_header_ok = path_exists_any(root / "include" / "acl" / "acl.h" for root in cann_roots)
    hccl_header_ok = path_exists_any(root / "include" / "hccl" / "hccn_rping.h" for root in cann_roots)
    cann_lib_ok = path_exists_any(root / "lib64" / "libhccl.so" for root in cann_roots)
    hccl_plf_lib_ok = path_exists_any(root / "lib64" / "libhccl_plf.so" for root in cann_roots)
    ascendcl_lib_ok = path_exists_any(root / "lib64" / "libascendcl.so" for root in cann_roots)
    if not (cann_header_ok and hccl_header_ok and cann_lib_ok and hccl_plf_lib_ok and ascendcl_lib_ok):
        checked = ", ".join(str(root) for root in cann_roots)
        missing.append(f"Ascend CANN/HCCL/ACL under ASCEND_CANN_PATH or ASCEND_HOME_PATH; checked: {checked}")

    if shutil.which("hccn_tool") is None:
        missing.append("hccn_tool")

    return missing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Install disp_probe third-party dependencies into one prefix.")
    parser.add_argument("--prefix", type=Path, default=DEFAULT_PREFIX, help="installation prefix")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="third-party manifest")
    parser.add_argument("--requirements", type=Path, default=None, help="python requirements file")
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 4, help="parallel build jobs")
    parser.add_argument("--skip-python", action="store_true", help="do not create the python virtualenv")
    parser.add_argument("--python-system-site-packages", action="store_true", help="let the venv see system packages")
    parser.add_argument(
        "--require-python-hashes",
        action="store_true",
        help="install python requirements with pip --require-hashes",
    )
    parser.add_argument("--offline", action="store_true", help="install python wheels from --wheelhouse only")
    parser.add_argument("--wheelhouse", type=Path, default=None, help="directory containing offline python wheels")
    parser.add_argument("--check-only", action="store_true", help="only check detected prerequisites")
    parser.add_argument("--strict-prereq", action="store_true", help="fail when external prerequisites are missing")
    parser.add_argument("--dry-run", action="store_true", help="print actions without changing files")
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    manifest_path = args.manifest.resolve()
    manifest = load_manifest(manifest_path)
    prefix = args.prefix.resolve()
    requirements = (args.requirements or resolve_repo_path(manifest["python"]["requirements"])).resolve()

    try:
        if not args.check_only:
            if args.dry_run:
                log(f"create prefix {prefix}")
            else:
                prefix.mkdir(parents=True, exist_ok=True)
            install_components(manifest, prefix, args.jobs, args.dry_run)
            install_python(requirements, prefix, args)
            write_metadata(manifest_path, requirements, prefix, args.dry_run)

        missing = check_prerequisites(prefix)
        if missing:
            log("external prerequisite check: missing")
            for item in missing:
                log(f"  - {item}")
            if args.strict_prereq:
                return 2
        else:
            log("external prerequisite check: ok")

        log(f"done. source env with: source {prefix}/share/disp_probe/third_party/env.sh")
        return 0
    except (OSError, subprocess.CalledProcessError, SetupError) as exc:
        log(f"failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
