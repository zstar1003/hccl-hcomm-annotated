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

import os
import shlex

import paramiko

import dispatcher_common as common
import json_info as cfg


def disp_file(
    from_path,
    to_path,
    child_dict=None,
    transport: paramiko.Transport = None,  # 这行以及之后的参数仅用于递归
    path: list = None,
):
    child_dict, path = common.prepare_tree_walk(child_dict, path)
    if child_dict is None:
        return
    for this_host_name, this_child_dict in child_dict.items():
        task = common.FileTransferTask(from_path, to_path, this_host_name, this_child_dict, transport, path)
        _disp_file_to_host(task)


def _expand_local_path(from_path):
    if not from_path.startswith("~"):
        return from_path
    return from_path.replace("~", os.path.expanduser("~"), 1)


def _expand_remote_path(to_path, user):
    if not to_path.startswith("~"):
        return to_path
    return to_path.replace("~", f"/home/{user}" if user != "root" else "/root")


def _upload_with_sftp(client, from_path, to_path):
    common.log_info(f"{from_path} -> {to_path}")
    to_folder = to_path[: to_path.rfind("/")]
    stdin, stdout, stderr = client.exec_command(f"mkdir -p {shlex.quote(to_folder)}")
    stdout.channel.recv_exit_status()
    with client.open_sftp() as sftp:
        sftp.put(from_path, to_path)
        sftp.chmod(to_path, os.stat(from_path).st_mode)


def _recurse_child(task, client):
    if not task.child_dict:
        return
    next_transport = client.get_transport()
    task.path.append(task.host_name)
    try:
        disp_file(task.from_path, task.to_path, task.child_dict, next_transport, path=task.path)
    finally:
        task.path.pop()
        next_transport.close()


def _disp_file_to_host(task):
    host_ip = cfg.clean_ip(task.host_name)
    expanded_from_path = _expand_local_path(task.from_path)
    first_user = True
    for user, pswd in cfg.host_to_user_pair[task.host_name].items():
        client = paramiko.SSHClient()
        cfg.setup_ssh_host_key_policy(client)
        connection = common.SshConnection(client, host_ip, task.host_name, user, pswd, task.transport)
        common.connect_ssh_client(connection)
        common.log_info(f"\033[33m[{','.join(task.path)}]->{task.host_name}\033[0m")
        _upload_with_sftp(client, expanded_from_path, _expand_remote_path(task.to_path, user))
        if first_user:
            first_user = False
            child_task = common.FileTransferTask(
                expanded_from_path, task.to_path, task.host_name, task.child_dict, task.transport, task.path
            )
            _recurse_child(child_task, client)
        client.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("python disp_file.py")

    parser.add_argument(
        "from_path",
        nargs="?",
        default="./README.md",
        help="from_path",
    )

    parser.add_argument(
        "to_path",
        nargs="?",
        default="./bin/",
        help="from_path",
    )
    parser.add_argument(
        "--file",
        "-f",
        default=None,
        help="control json file",
    )

    from_fpath, to_fpath = common.load_file_transfer_args(parser.parse_args())

    disp_file(from_fpath, to_fpath)
