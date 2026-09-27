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

import logging
import os
import sys
from dataclasses import dataclass
from typing import Any

_CONSOLE_LOGGER = logging.getLogger("cluster_link_diag.dispatcher")


@dataclass
class SshConnection:
    client: Any
    host_ip: str
    host_name: str
    user: str
    pswd: str
    transport: Any = None
    sock: Any = None


@dataclass
class CommandExecContext:
    commands: Any
    client: Any
    logger: Any = None
    print_cmd: bool = True
    host: str = None
    user: str = None
    pswd: str = None
    su_pswd: str = None


@dataclass
class TreeExecContext:
    before_cmd: Any = None
    after_cmd: Any = None
    host_cmd: Any = None
    child_dict: Any = None
    transport: Any = None
    path: Any = None


@dataclass
class FileTransferTask:
    from_path: str
    to_path: str
    host_name: str = None
    child_dict: Any = None
    transport: Any = None
    path: Any = None


@dataclass
class HostUserTask:
    host_name: str
    child_dict: Any
    path: Any
    logger: Any
    user: str
    pswd: str
    transport: Any = None
    recurse: bool = False


def get_console_logger():
    if not _CONSOLE_LOGGER.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(message)s"))
        _CONSOLE_LOGGER.addHandler(handler)
        _CONSOLE_LOGGER.propagate = False
    _CONSOLE_LOGGER.setLevel(logging.INFO)
    return _CONSOLE_LOGGER


def log_info(message):
    get_console_logger().info(message)


def log_warning(message):
    get_console_logger().warning(message)


def log_error(message):
    get_console_logger().error(message)


def normalize_child_dict(child_dict):
    import json_info as cfg

    cfg.ensure_config_loaded()
    if child_dict is None:
        child_dict = cfg.control_topo
    if child_dict is None:
        return None
    if isinstance(child_dict, str):
        if len(child_dict) == 0:
            return None
        child_dict = [child_dict]
    if isinstance(child_dict, list):
        if len(child_dict) == 0:
            return None
        child_dict = {host: None for host in child_dict}
    return child_dict


def prepare_tree_walk(child_dict, path):
    if path is None:
        path = []
    child_dict = normalize_child_dict(child_dict)
    return child_dict, path


def load_file_transfer_args(args):
    import json_info as cfg

    cfg.load_config(args.file)
    from_fpath = args.from_path
    to_fpath = resolve_destination_path(from_fpath, args.to_path)
    return from_fpath, to_fpath


def finalize_command_args(args, log_dir, enable_logger_ref):
    import json_info as cfg

    cfg.load_config(args.file)
    enable_logger_ref["value"] = args.enable_logger
    if args.enable_logger:
        os.makedirs(log_dir, exist_ok=True)
        log_info(f"log will be saved in: {log_dir}")
    return args.commands


def add_common_command_args(parser):
    parser.add_argument(
        "--enable_logger",
        "-l",
        required=False,
        default=False,
        action="store_true",
        help="enable logger",
    )
    parser.add_argument(
        "--file",
        "-f",
        default=None,
        help="control json file",
    )


def setup_file_logger(host_path, log_dir, enable_logger):
    if not enable_logger:
        return None
    logger = logging.getLogger(host_path)
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger

    log_file = os.path.join(log_dir, f"{host_path}.log")
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def close_logger(logger):
    if logger is None:
        return
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)


def normalize_command_list(commands):
    return [commands] if isinstance(commands, str) else commands


def exec_command_with_auth(client, cmd, pswd=None, su_pswd=None):
    import json_info as cfg

    needs_sudo_password = "sudo " in cmd or cmd.strip().startswith("sudo")
    needs_su_password = "su " in cmd or cmd.strip().startswith("su")
    cmd = cfg.sudo_cmd(cmd, pswd)
    cmd = cfg.su_cmd(cmd, su_pswd)

    stdin, stdout, stderr = client.exec_command(cmd)
    if needs_sudo_password and pswd:
        stdin.write(pswd + "\n")
        stdin.flush()
    if needs_su_password and su_pswd:
        stdin.write(su_pswd + "\n")
        stdin.flush()
    return stdin, stdout, stderr


def connect_ssh_client(connection):
    import json_info as cfg

    if connection.transport is None:
        cfg.connect_ssh_client(connection)
        return

    channel = connection.transport.open_channel(
        "direct-tcpip",
        (connection.host_ip, cfg.default_ssh_port),
        ("", 0),
        timeout=cfg.default_timeout,
    )
    forwarded = SshConnection(
        connection.client,
        connection.host_ip,
        connection.host_name,
        connection.user,
        connection.pswd,
        sock=channel,
    )
    cfg.connect_ssh_client(forwarded)


def resolve_destination_path(from_fpath, to_fpath):
    import json_info as cfg

    if to_fpath[-1] == "/":
        to_fpath += from_fpath.split("/")[-1]
    if to_fpath.startswith("."):
        cfg.ensure_config_loaded()
        to_fpath = cfg.to_path + to_fpath[1:]
    return to_fpath
