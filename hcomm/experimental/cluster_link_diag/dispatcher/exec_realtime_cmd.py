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

# exec_realtime_cmd.py:
import os
import signal
import threading
import time

import paramiko

import dispatcher_common as common
import json_info as cfg

log_dir = f"./logs/{time.strftime('%Y-%m-%d %H:%M:%S')}"
_enable_logger_ref = {"value": False}

# 添加全局执行器实例引用
_executor_instance = None


def signal_handler(sig, frame):
    """Handle Ctrl+C signal."""
    common.log_info("\n\033[33mReceived Ctrl+C, closing all connections...\033[0m")
    if _executor_instance:
        _executor_instance.force_close()


def setup_signal_handler(executor):
    """Set up signal handler."""
    global _executor_instance
    _executor_instance = executor
    signal.signal(signal.SIGINT, signal_handler)


def setup_logger(host_path, host_name):
    """Set up an independent logger for each host."""
    return common.setup_file_logger(host_path, log_dir, _enable_logger_ref["value"])


class ParallelSSHExecutor:

    _ipv4_print_len = 15

    def __init__(self):
        self.clients = []  # 存储所有客户端连接
        self.threads = []  # 存储所有线程
        self.lock = threading.Lock()  # 用于同步访问共享资源
        self.completed = False  # 标记所有命令是否执行完成
        self.loggers = {}  # 存储每个主机的logger
        self.force_closing = False  # 标记是否正在强制关闭

    @staticmethod
    def _log_command_header(context, commands, index, cmd):
        original_cmd = cmd.replace("\n", "")[:1000]
        if context.print_cmd:
            cmd_display = f"[{index}][{original_cmd}]" if len(commands) > 1 else f"[{original_cmd}]"
            common.log_info(f"\033[34m{cmd_display}\033[0m")
        if context.logger:
            context.logger.info(f"command: {original_cmd}")

    def force_close(self):
        """Forcefully close all connections."""
        self.force_closing = True
        common.log_info("\033[33mForcefully closing all SSH connections...\033[0m")
        
        # 首先尝试正常关闭所有客户端
        for client in self.clients[::-1]:
            try:
                # 尝试关闭传输层
                transport = client.get_transport()
                if transport and transport.is_active():
                    transport.close()
                client.close()
            except (paramiko.SSHException, OSError):
                continue
        
        # 设置线程超时并强制结束
        for thread in self.threads:
            if thread.is_alive():
                thread.join(timeout=2)  # 等待2秒
        
        # 如果还有存活线程，强制结束程序
        alive_threads = [t for t in self.threads if t.is_alive()]
        if alive_threads:
            self.completed = False
            common.log_error(
                "\033[31mSome threads cannot terminate normally, please check if remote "
                "commands are still running.\033[0m"
            )

    def exec_realtime_commands(self, context):
        try:
            commands = common.normalize_command_list(context.commands)
            self._exec_realtime_command_list(context, commands)
            self._mark_completed()
        except (paramiko.SSHException, OSError, RuntimeError, ValueError) as err:
            self._log_command_exception(context, err)

    def re_realtime_exec(
        self,
        target_cmd=None,
        child_dict=None,
        transport: paramiko.Transport = None,
        path: list = None,
    ):
        child_dict, path = common.prepare_tree_walk(child_dict, path)
        if child_dict is None:
            return

        for this_host_name, this_child_dict in child_dict.items():
            if self.force_closing:
                return
            task = common.HostUserTask(this_host_name, this_child_dict, path, None, "", "", transport=transport)
            self._realtime_exec_host(target_cmd, task)

    def wait_and_close(self):
        # 等待所有线程完成
        for thread in self.threads:
            thread.join()

        # 关闭所有客户端连接
        for client in self.clients[::-1]:
            try:
                client.close()
            except (paramiko.SSHException, OSError):
                pass
        
        # 清理所有logger的handler
        if _enable_logger_ref["value"]:
            for logger in self.loggers.values():
                for handler in logger.handlers[:]:
                    handler.close()
                    logger.removeHandler(handler)

    def _mark_completed(self):
        with self.lock:
            self.completed = True

    def _exec_realtime_command_list(self, context, commands):
        for index, cmd in enumerate(commands):
            if self.force_closing:
                return
            self._exec_one_realtime_command(context, commands, index, cmd)

    def _log_command_exception(self, context, err):
        if self.force_closing:
            return
        common.log_error(
            f"\033[31m[{context.host:<{self._ipv4_print_len}}] error occurred while executing command: "
            f"{str(err)}\033[0m"
        )
        if context.logger:
            context.logger.error(f"error occurred while executing command: {str(err)}")

    def _log_realtime_line(self, context, line):
        common.log_info(f"[{context.host:<{self._ipv4_print_len}}] {line.rstrip()}")
        if context.logger:
            context.logger.info(line.strip())

    def _log_realtime_error(self, context, error_msg):
        for line in error_msg.splitlines():
            common.log_error(f"\033[31m[{context.host:<{self._ipv4_print_len}}] [Error] {line}\033[0m")
        if context.logger and error_msg.strip():
            context.logger.error(f"Error:\n{error_msg}")

    def _drain_stdout(self, context, stdout):
        output_buffer = ""
        while not stdout.channel.exit_status_ready():
            if self.force_closing:
                return ""
            if stdout.channel.recv_ready():
                data = stdout.channel.recv(1024).decode()
                lines = (output_buffer + data).splitlines(keepends=True)
                for line in lines[:-1]:
                    self._log_realtime_line(context, line)
                output_buffer = lines[-1] if lines else ""
            time.sleep(0.1)
        return output_buffer

    def _flush_stdout(self, context, stdout, output_buffer):
        remaining_output = stdout.channel.recv(4096).decode()
        if remaining_output:
            for line in (output_buffer + remaining_output).splitlines(keepends=True):
                self._log_realtime_line(context, line)
            return
        if output_buffer:
            self._log_realtime_line(context, output_buffer)

    def _exec_one_realtime_command(self, context, commands, index, cmd):
        self._log_command_header(context, commands, index, cmd)
        stdin, stdout, stderr = common.exec_command_with_auth(context.client, cmd, context.pswd, context.su_pswd)
        output_buffer = self._drain_stdout(context, stdout)
        if self.force_closing:
            return
        self._flush_stdout(context, stdout, output_buffer)
        err = stderr.read()
        if err:
            self._log_realtime_error(context, err.decode())

    def _start_realtime_thread(self, target_cmd, client, task):
        if not target_cmd:
            return
        exec_context = common.CommandExecContext(
            [cfg.make_cmd_abs(cmd, task.user) for cmd in target_cmd],
            client,
            logger=task.logger,
            host=task.host_name,
            user=task.user,
            pswd=task.pswd,
            su_pswd=cfg.host_to_su_pswd[task.host_name],
        )
        thread = threading.Thread(target=self.exec_realtime_commands, args=(exec_context,))
        thread.start()
        with self.lock:
            self.threads.append(thread)

    def _recurse_realtime_child(self, target_cmd, client, task):
        if not task.child_dict:
            return
        next_transport = client.get_transport()
        task.path.append(task.host_name)
        try:
            self.re_realtime_exec(target_cmd, task.child_dict, next_transport, path=task.path)
        finally:
            task.path.pop()

    def _realtime_exec_host_user(self, target_cmd, task):
        if self.force_closing:
            return
        client = paramiko.SSHClient()
        cfg.setup_ssh_host_key_policy(client)
        try:
            host_ip = cfg.clean_ip(task.host_name)
            connection = common.SshConnection(client, host_ip, task.host_name, task.user, task.pswd, task.transport)
            common.connect_ssh_client(connection)
            with self.lock:
                self.clients.append(client)
            host_path = ('[' + ','.join(task.path) + ']->' if task.path else '') + task.host_name
            common.log_info(f"\033[33m{host_path}\033[0m")
            self._start_realtime_thread(target_cmd, client, task)
            if task.recurse:
                self._recurse_realtime_child(target_cmd, client, task)
        except (paramiko.SSHException, OSError, RuntimeError, ValueError) as err:
            if not self.force_closing:
                common.log_error(f"\033[31mconnect to {task.host_name} failed: {str(err)}\033[0m")

    def _realtime_exec_host(self, target_cmd, task):
        host_name = task.host_name
        if host_name not in cfg.host_to_user_pair:
            return
        host_path = ('[' + ','.join(task.path) + ']->' if task.path else '') + host_name
        logger = setup_logger(host_path, host_name)
        self.loggers[host_path] = logger
        for index, (user, pswd) in enumerate(cfg.host_to_user_pair[host_name].items()):
            user_task = common.HostUserTask(
                host_name,
                task.child_dict,
                task.path,
                logger,
                user,
                pswd,
                transport=task.transport,
                recurse=(index == 0),
            )
            self._realtime_exec_host_user(target_cmd, user_task)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("python exec_realtime_cmd.py")

    parser.add_argument(
        "commands",
        nargs="+",
        help="one or more commands to execute for all hosts",
    )
    
    common.add_common_command_args(parser)

    commands = common.finalize_command_args(parser.parse_args(), log_dir, _enable_logger_ref)

    # 使用并行执行器
    executor = ParallelSSHExecutor()
    
    # 设置信号处理器
    setup_signal_handler(executor)
    
    try:
        executor.re_realtime_exec(commands)
        executor.wait_and_close()
    except (paramiko.SSHException, OSError, RuntimeError, ValueError) as err:
        common.log_error(f"\033[31mprogram execution error: {str(err)}\033[0m")
        executor.force_close()
