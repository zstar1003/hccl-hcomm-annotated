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

import queue
import shlex
import subprocess
import threading
from typing import Dict, List

import dispatcher_common as common


class ParallelCommandExecutor:
    """并行命令执行器,优雅地捕获输出"""

    def __init__(self):
        self.processes = []
        self.output_queue = queue.Queue()
        self.threads = []
        self.commands = []

    @staticmethod
    def _collect_queue_item(outputs, completed, index, line):
        if line is None:
            completed.add(index)
            return
        outputs[index].append(line)

    def add_command(self, command):
        """添加要执行的命令"""
        self.commands.append(command)

    def set_commands(self, commands):
        """设置要执行的命令列表"""
        self.commands = commands

    def exec_commands(self, commands: List[str] = None) -> List[str]:
        """
        并行执行多个命令并捕获输出
        :param commands: 命令列表
        :return: 字典 {进程索引: 完整输出}
        """
        if commands is None:
            commands = self.commands
        else:
            self.commands = commands
        if not commands:
            return []
        # 启动所有进程
        for cmd in commands:
            self._start_process(cmd)

        # 为每个进程启动输出捕获线程
        for i, proc in enumerate(self.processes):
            t = threading.Thread(
                target=self._capture_output, args=(proc, i), daemon=True
            )
            t.start()
            self.threads.append(t)

        # 等待所有进程完成
        self._wait_for_completion()

        # 收集所有输出
        return self._collect_outputs()

    def get_exit_codes(self) -> list[int]:
        """获取所有进程的退出码"""
        return [proc.returncode for proc in self.processes]

    def _start_process(self, command: str):
        """启动一个子进程"""
        cmd_args = command if isinstance(command, list) else shlex.split(command)
        proc = subprocess.Popen(
            cmd_args,
            shell=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # 合并错误输出到标准输出
            bufsize=1,  # 行缓冲
            universal_newlines=True,  # 文本模式
            errors="replace",  # 处理编码错误
        )
        self.processes.append(proc)

    def _capture_output(self, proc: subprocess.Popen, index: int):
        """捕获单个进程的输出"""
        try:
            # 实时读取输出
            while True:
                line = proc.stdout.readline()
                if not line and proc.poll() is not None:
                    break
                if line:
                    # 将输出放入队列（带进程索引）
                    self.output_queue.put((index, line.strip()))
        except Exception as e:
            self.output_queue.put((index, f"error captured output: {str(e)}"))
        finally:
            # 标记进程完成
            self.output_queue.put((index, None))

    def _wait_for_completion(self):
        """等待所有进程完成"""
        # 等待所有进程结束
        for proc in self.processes:
            try:
                proc.wait(timeout=60)  # 设置超时时间
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()

    def _collect_outputs(self) -> list[str]:
        """从队列收集所有输出"""
        outputs = {i: [] for i in range(len(self.processes))}
        completed = set()

        while len(completed) < len(self.processes):
            try:
                index, line = self.output_queue.get(timeout=1)
                self._collect_queue_item(outputs, completed, index, line)
            except queue.Empty:
                self._mark_finished_processes(completed)

        return [lines for i, lines in outputs.items()]

    def _mark_finished_processes(self, completed):
        for i, proc in enumerate(self.processes):
            if i not in completed and proc.poll() is not None:
                completed.add(i)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("python parallel_command_executor.py")
    parser.add_argument(
        "commands",
        nargs="*",
        help="commands to execute",
        default=["ping 8.8.8.8", "ping 8.8.8.8", "ping 8.8.8.8"],
    )
    args = parser.parse_args()

    commands = args.commands

    executor = ParallelCommandExecutor()
    outputs = executor.exec_commands(commands)
    for i, output in enumerate(outputs):
        common.log_info(f"[{i}][{commands[i]}]")
        common.log_info("\n".join(output))
        common.log_info("")

    exit_codes = executor.get_exit_codes()
    for i, code in enumerate(exit_codes):
        common.log_info(f"[{i}][{commands[i]}] exit code: {code}")
