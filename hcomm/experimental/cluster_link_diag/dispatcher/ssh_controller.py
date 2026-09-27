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

import paramiko

import dispatcher_common as common
from exec_cmd import exec_commands
import json_info as cfg


def ssh_controller(commands, host, user, pswd, default_timeout=5):
    client = paramiko.SSHClient()
    connection = common.SshConnection(client, host, host, user, pswd)
    cfg.connect_ssh_client(connection)

    exec_context = common.CommandExecContext(commands, client, pswd=pswd, su_pswd=cfg.host_to_su_pswd[host])
    exec_commands(exec_context)

    client.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("python ssh_controller.py")
    parser.add_argument(
        "commands",
        nargs="*",
        default=["sudo ls -l", "pwd", 'for i in {1..5}; do echo "test";sleep 1;\ndone'],
        help="command to execute",
    )
    parser.add_argument(
        "--exec_dir",
        "-d",
        type=str,
        default=None,
        help="directory to execute commands",
    )
    parser.add_argument(
        "--controller", "-c", type=str, default=None, help="controller ip"
    )
    parser.add_argument(
        "--file", "-f", default=None, help="control json file"
    )

    args = parser.parse_args()
    cfg.load_config(args.file)
    commands = args.commands
    exec_dir = args.exec_dir if args.exec_dir is not None else cfg.to_path
    host = args.controller if args.controller is not None else cfg.controller_ip
    user = next(iter(cfg.host_to_user_pair[host]))
    pswd = cfg.host_to_user_pair[host][user]

    if exec_dir and len(exec_dir) > 0:
        commands = [f"cd {exec_dir} && {cmd}" for cmd in commands]

    ssh_controller(commands, host, user, pswd, default_timeout=cfg.default_timeout)
