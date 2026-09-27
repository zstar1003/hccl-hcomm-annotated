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

import shlex

import paramiko
from scp import SCPClient  # 需要额外安装 scp 模块

import dispatcher_common as common
import json_info as cfg


def disp_file_scp(from_path, to_path, child_dict=None, transport: paramiko.Transport = None, path: list = None):
    child_dict, path = common.prepare_tree_walk(child_dict, path)
    if child_dict is None:
        return
    for this_host_name, this_child_dict in child_dict.items():
        task = common.FileTransferTask(from_path, to_path, this_host_name, this_child_dict, transport, path)
        _disp_file_scp_to_host(task)


def _upload_with_scp(client, from_path, to_path):
    common.log_info(f"{from_path} -> {to_path}")
    to_folder = to_path[: to_path.rfind("/")]
    stdin, stdout, stderr = client.exec_command(f"mkdir -p {shlex.quote(to_folder)}")
    stdout.channel.recv_exit_status()
    with SCPClient(client.get_transport()) as scp:
        scp.put(from_path, to_path)


def _recurse_child(task, client):
    if not task.child_dict:
        return
    with client.get_transport() as next_transport:
        task.path.append(task.host_name)
        try:
            disp_file_scp(task.from_path, task.to_path, task.child_dict, next_transport, path=task.path)
        finally:
            task.path.pop()


def _disp_file_scp_to_host(task):
    if task.host_name not in cfg.host_to_user_pair:
        return
    host_ip = cfg.clean_ip(task.host_name)
    first_user = True
    for user, pswd in cfg.host_to_user_pair[task.host_name].items():
        client = paramiko.SSHClient()
        cfg.setup_ssh_host_key_policy(client)
        connection = common.SshConnection(client, host_ip, task.host_name, user, pswd, task.transport)
        common.connect_ssh_client(connection)
        common.log_info(f"\033[33m[{','.join(task.path)}]->{task.host_name}\033[0m")
        _upload_with_scp(client, task.from_path, task.to_path)
        if first_user:
            first_user = False
            _recurse_child(task, client)
        client.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("python disp_file_scp.py")

    parser.add_argument(
        "from_path",
        help="source path to distribute",
    )

    parser.add_argument(
        "to_path",
        help="destination path on remote host",
    )
    parser.add_argument(
        "--file",
        "-f",
        default=None,
        help="control json file",
    )

    from_fpath, to_fpath = common.load_file_transfer_args(parser.parse_args())

    disp_file_scp(from_fpath, to_fpath)
