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

import time

import paramiko

import dispatcher_common as common
import json_info as cfg

log_dir = f"./logs/{time.strftime('%Y-%m-%d %H:%M:%S')}"
_enable_logger_ref = {"value": False}


def setup_logger(host_path, host_name):
    """为每个主机设置独立的logger"""
    return common.setup_file_logger(host_path, log_dir, _enable_logger_ref["value"])


def exec_commands(context):
    commands = context.commands
    if isinstance(commands, str):
        commands = [commands]

    for i, cmd in enumerate(commands):
        original_cmd = cmd.replace("\n", "")[:1000]
        if context.print_cmd:
            if len(commands) > 1:
                cmd_display = f"[{i}][{original_cmd}]"
            else:
                cmd_display = f"[{original_cmd}]"
            
            common.log_info(f"\033[34m{cmd_display}\033[0m")

        stdin, stdout, stderr = common.exec_command_with_auth(context.client, cmd, context.pswd, context.su_pswd)
        output_lines = []
        
        # 使用非阻塞方式输出
        while not stdout.channel.exit_status_ready():
            if stdout.channel.recv_ready():
                data = stdout.channel.recv(1024).decode()
                common.log_info(data.rstrip())
                output_lines.append(data)
            time.sleep(0.1)  # 避免CPU占用过高

        # 检查是否有剩余输出
        remaining_output = stdout.channel.recv(4096).decode()
        if remaining_output:
            common.log_info(remaining_output.rstrip())
            output_lines.append(remaining_output)

        # 记录命令输出到日志
        if context.logger and output_lines:
            full_output = ''.join(output_lines)
            if full_output.strip():  # 只记录非空输出
                context.logger.info(f"output:\n{full_output}")

        err = stderr.read()
        if err:
            error_msg = err.decode()
            common.log_error(f"\033[31m[Error] {error_msg.rstrip()}\033[0m")
            if context.logger:
                context.logger.error(f"Error:\n{error_msg}")


def re_exec(context):
    child_dict, path = common.prepare_tree_walk(context.child_dict, context.path)
    if child_dict is None:
        return

    for this_host_name, this_child_dict in child_dict.items():
        _exec_host(context, this_host_name, this_child_dict, path)


def _make_exec_context(commands, client, task):
    return common.CommandExecContext(
        [cfg.make_cmd_abs(cmd, task.user) for cmd in commands],
        client,
        logger=task.logger,
        pswd=task.pswd,
        su_pswd=cfg.host_to_su_pswd[task.host_name],
    )


def _exec_optional_commands(commands, client, task):
    if not commands:
        return
    exec_commands(_make_exec_context(commands, client, task))


def _recurse_exec(context, child_dict, client, host_name, path):
    if not child_dict:
        return
    next_transport = client.get_transport()
    path.append(host_name)
    try:
        re_exec(
            common.TreeExecContext(
                context.before_cmd,
                context.after_cmd,
                context.host_cmd,
                child_dict,
                next_transport,
                path,
            )
        )
    finally:
        path.pop()
        next_transport.close()


def _exec_host_user(context, task):
    client = paramiko.SSHClient()
    cfg.setup_ssh_host_key_policy(client)
    try:
        host_ip = cfg.clean_ip(task.host_name)
        connection = common.SshConnection(client, host_ip, task.host_name, task.user, task.pswd, context.transport)
        common.connect_ssh_client(connection)
        host_path = ('[' + ','.join(task.path) + ']->' if task.path else '') + task.host_name
        common.log_info(f"\033[33m{host_path}\033[0m")
        _exec_optional_commands(context.before_cmd, client, task)
        _exec_optional_commands(context.host_cmd if task.child_dict else None, client, task)
        if task.recurse:
            _recurse_exec(context, task.child_dict, client, task.host_name, task.path)
        _exec_optional_commands(context.after_cmd, client, task)
    except (paramiko.SSHException, OSError, RuntimeError, ValueError) as err:
        common.log_error(f"\033[31m[Error] {str(err)}\033[0m")
    finally:
        client.close()


def _exec_host(context, host_name, child_dict, path):
    host_path = ('[' + ','.join(path) + ']->' if path else '') + host_name
    logger = setup_logger(host_path, host_name)
    for index, (user, pswd) in enumerate(cfg.host_to_user_pair[host_name].items()):
        task = common.HostUserTask(host_name, child_dict, path, logger, user, pswd, recurse=(index == 0))
        _exec_host_user(context, task)
    common.close_logger(logger)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("python exec_cmd.py")

    parser.add_argument(
        "commands",
        nargs="*",
        default=["whoami", "ls ./"],
        help="commands to execuate for all topo",
    )
    
    common.add_common_command_args(parser)

    commands = common.finalize_command_args(parser.parse_args(), log_dir, _enable_logger_ref)

    re_exec(common.TreeExecContext(before_cmd=commands))
