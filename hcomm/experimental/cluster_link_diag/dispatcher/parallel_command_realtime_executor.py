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
import subprocess
import threading

import dispatcher_common as common


class ParallelCommandRealtimeExecutor:
    def __init__(self):
        self.commands = []
        self.threads = []
        self.lock = threading.Lock()
        self.return_codes = []
        self.results = self.return_codes

    @staticmethod
    def _execute_single_command(command):
        """执行单个命令并实时输出"""
        process = subprocess.Popen(
            command.split(" "),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )

        # 实时读取输出
        for line in process.stdout:
            common.log_info(line.rstrip())

        process.wait()
        return process.returncode

    def add_command(self, command):
        """添加要执行的命令"""
        self.commands.append(command)
        self.return_codes.append(0)
        self.results = self.return_codes

    def set_commands(self, commands):
        """设置要执行的命令列表"""
        self.commands = commands
        self.return_codes = [0] * len(commands)
        self.results = self.return_codes

    def exec_commands(self, commands=None):
        if commands is None:
            commands = self.commands
        else:
            self.set_commands(commands)
        if not commands:
            return []

        # 单个任务直接执行
        if len(self.commands) == 1:
            return_code = self._execute_single_command(self.commands[0])
            self.return_codes = [return_code]
            self.results = self.return_codes
            return self.return_codes

        # 多个任务并行执行
        for idx, cmd in enumerate(self.commands):
            thread = threading.Thread(
                target=lambda i, c: self._execute_command_with_id(c, i), args=(idx, cmd)
            )
            self.threads.append(thread)
            thread.start()

        # 等待所有线程完成
        for thread in self.threads:
            thread.join()

        return self.return_codes

    def get_results(self):
        """获取所有任务的执行结果"""
        return self.return_codes

    def _execute_command_with_id(self, command, task_id):
        """执行命令并添加任务ID前缀输出"""
        process = subprocess.Popen(
            command.split(" "),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )

        # 实时读取输出并添加前缀
        for line in process.stdout:
            with self.lock:
                common.log_info(f"[{task_id}] {line.rstrip()}")

        process.wait()
        return_code = process.returncode
        with self.lock:
            self.return_codes[task_id] = return_code
        return return_code


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("python parallel_command_realtime_executor.py")
    parser.add_argument(
        "commands",
        nargs="*",
        help="commands to execute",
        default=["ping 8.8.8.8", "ping 8.8.8.8", "ping 8.8.8.8"],
    )
    args = parser.parse_args()

    commands = args.commands

    common.log_info(f"Executing {len(commands)} commands in parallel... :{commands}")

    executor = ParallelCommandRealtimeExecutor()
    return_codes = executor.exec_commands(commands)
    for i, return_code in enumerate(return_codes):
        common.log_info(f"[{i}][{commands[i]}]" + f" returned {return_code}")
