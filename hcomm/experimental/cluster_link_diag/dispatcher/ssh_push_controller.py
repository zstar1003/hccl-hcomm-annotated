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
import posixpath
import re
import shlex
import shutil
import subprocess
import tarfile
import time
from dataclasses import dataclass

import paramiko
import paramiko.ssh_exception

import dispatcher_common as common
import json_info as cfg


@dataclass
class IncrementalApplyTask:
    remote_repo: str
    remote_temp_dir: str
    local_tar_path: str
    tar_filename: str
    deleted_files: list


class GitSync:
    def __init__(self, host, username, *auth_args, **auth_kwargs):
        if len(auth_args) > 3:
            raise ValueError("too many GitSync authentication arguments")
        self.host = host
        self.username = username
        self.password = auth_args[0] if len(auth_args) > 0 else auth_kwargs.get("password")
        self.key_filename = auth_args[1] if len(auth_args) > 1 else auth_kwargs.get("key_filename")
        self.port = auth_args[2] if len(auth_args) > 2 else auth_kwargs.get("port", 22)
        self.ssh = None
        self.sftp = None

    @staticmethod
    def create_incremental_tar(local_repo, local_tar_path, changed_files):
        with tarfile.open(local_tar_path, "w:gz") as tar:
            for file in changed_files:
                GitSync.add_existing_file_to_tar(tar, local_repo, file)

    @staticmethod
    def add_existing_file_to_tar(tar, local_repo, file):
        try:
            file_path = os.path.join(local_repo, file)
            if os.path.exists(file_path):
                tar.add(file_path, arcname=file)
        except (OSError, tarfile.TarError) as err:
            common.log_error(f"[error][file:{file}] {str(err)}")

    @staticmethod
    def bundle_has_data(local_repo, local_bundle_path):
        return bool(subprocess.check_output(
            ["git", "bundle", "list-heads", local_bundle_path],
            cwd=local_repo,
            text=True,
        ))

    @staticmethod
    def get_local_commit(local_repo):
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=local_repo, text=True).strip()

    @staticmethod
    def create_bundle(local_repo, local_bundle_path, remote_commit):
        subprocess.run(
            ["git", "bundle", "create", local_bundle_path, f"{remote_commit}..HEAD"],
            cwd=local_repo,
            check=True,
        )

    def connect(self):
        """建立SSH和SFTP连接"""
        cfg.ensure_config_loaded()
        self.ssh = paramiko.SSHClient()
        cfg.setup_ssh_host_key_policy(self.ssh)

        if self.key_filename:
            self.ssh.connect(
                self.host,
                port=self.port,
                username=self.username,
                key_filename=self.key_filename,
                timeout=cfg.default_timeout,
            )
        else:
            self.ssh.connect(
                self.host,
                port=self.port,
                username=self.username,
                password=self.password,
                timeout=cfg.default_timeout,
            )

        self.sftp = self.ssh.open_sftp()

    def disconnect(self):
        """关闭连接"""
        if self.sftp:
            self.sftp.close()
        if self.ssh:
            self.ssh.close()

    def execute_command(self, command):
        """执行远程命令"""
        stdin, stdout, stderr = self.ssh.exec_command(command)
        output = stdout.read().decode()
        error = stderr.read().decode()

        if error and len(error):
            common.log_error(f"\033[31m[error][{command}]\033[0m\n{error.rstrip()}\033[0m")
        elif output and len(output):
            common.log_info(f"\033[34m[output][{command}]\033[0m\n{output.rstrip()}\033[0m")
        return output, error

    @staticmethod
    def remote_path_join(*args):
        """Linux远程路径拼接"""
        path = "/".join(args)
        # 处理可能的双斜杠
        path = path.replace("//", "/")
        return path

    @staticmethod
    def quote_remote_path(path):
        return shlex.quote(path)

    @staticmethod
    def safe_git_path(path):
        if not path or path.startswith("/"):
            return False
        if any(part == ".." for part in path.split("/")):
            return False
        normalized = posixpath.normpath(path)
        if normalized == "." or normalized.startswith("../") or normalized == "..":
            return False
        return True

    @staticmethod
    def validate_commit(commit):
        if not re.fullmatch(r"[0-9a-fA-F]{40}", commit):
            raise ValueError(f"invalid remote commit: {commit!r}")
        return commit

    def ensure_remote_dir(self, remote_dir):
        """确保远程目录存在"""
        self.execute_command(f"mkdir -p {self.quote_remote_path(remote_dir)}")

    def full_sync(self, local_repo, remote_repo):
        """
        全量同步整个仓库
        :param local_repo: 本地仓库路径
        :param remote_repo: 远程仓库路径
        """
        common.log_info("[---full_sync---]")

        local_temp_dir = os.path.join(local_repo, "temp")
        os.makedirs(local_temp_dir, exist_ok=True, mode=0o777)
        clean_remote_cmd = (
            f"find {self.quote_remote_path(remote_repo)} -mindepth 1 ! -name .git -exec rm -rf {{}} +"
        )
        self.execute_command(clean_remote_cmd)
        remote_temp_dir = self.remote_path_join(remote_repo, "temp")
        self.ensure_remote_dir(remote_temp_dir)

        timestamp = time.strftime("%Y%m%d%H%M%S")
        tar_filename = f"repo_full_{timestamp}.tar.gz"
        local_tar_path = os.path.join(local_temp_dir, tar_filename)
        self.create_full_tar(local_repo, local_tar_path)
        self.apply_full_tar(remote_repo, remote_temp_dir, local_tar_path, tar_filename)
        os.remove(local_tar_path)
        common.log_info("[---full_sync_done---]")

    @staticmethod
    def create_full_tar(local_repo, local_tar_path):
        with tarfile.open(local_tar_path, "w:gz") as tar:
            for item in os.listdir(local_repo):
                if item != "temp" and not item.startswith("__pycache__"):
                    item_path = os.path.join(local_repo, item)
                    tar.add(item_path, arcname=item)
                else:
                    common.log_info(f"[ignore] {item}")

    def apply_full_tar(self, remote_repo, remote_temp_dir, local_tar_path, tar_filename):
        remote_tar_path = self.remote_path_join(remote_temp_dir, tar_filename)
        self.sftp.put(local_tar_path, remote_tar_path)
        commands = [
            (
                f"tar --no-same-owner -xzf {self.quote_remote_path(remote_tar_path)} "
                f"-C {self.quote_remote_path(remote_repo)}"
            ),
            f"rm {self.quote_remote_path(remote_tar_path)}",
        ]
        for cmd in commands:
            self.execute_command(cmd)

    def incremental_sync(self, local_repo, remote_repo):
        """
        增量同步,基于git diff,支持文件删除
        :param local_repo: 本地仓库路径
        :param remote_repo: 远程仓库路径
        """
        common.log_info("[---incremental_sync---]")

        local_temp_dir = os.path.join(local_repo, "temp")
        os.makedirs(local_temp_dir, exist_ok=True, mode=0o777)
        remote_temp_dir = self.remote_path_join(remote_repo, "temp")
        self.ensure_remote_dir(remote_temp_dir)

        remote_commit = self.get_remote_commit(remote_repo)
        if not remote_commit:
            common.log_warning("[warning] remote repo is empty, doing full sync")
            self.full_sync(local_repo, remote_repo)
            return
        remote_commit = self.validate_commit(remote_commit)

        changed_files, deleted_files = self.get_changed_files(local_repo, remote_commit)
        if not changed_files and not deleted_files:
            common.log_warning("[warning] no change detected, doing nothing")
            return

        self.log_changed_files(changed_files, deleted_files)
        timestamp = time.strftime("%Y%m%d%H%M%S")
        tar_filename = f"repo_update_{timestamp}.tar.gz"
        local_tar_path = os.path.join(local_temp_dir, tar_filename)
        self.create_incremental_tar(local_repo, local_tar_path, changed_files)
        apply_task = IncrementalApplyTask(remote_repo, remote_temp_dir, local_tar_path, tar_filename, deleted_files)
        self.apply_incremental_tar(apply_task)
        os.remove(local_tar_path)
        common.log_info("[---incremental_sync_done---]")

    def get_remote_commit(self, remote_repo):
        cmd = f"cd {self.quote_remote_path(remote_repo)} && git rev-parse HEAD 2>/dev/null || echo ''"
        remote_commit, _ = self.execute_command(cmd)
        return remote_commit.strip()

    def get_changed_files(self, local_repo, remote_commit):
        changed_files = subprocess.check_output(
            ["git", "diff", "--name-only", remote_commit, "HEAD"],
            cwd=local_repo,
            text=True,
        ).splitlines()
        deleted_files = subprocess.check_output(
            ["git", "diff", "--name-only", "--diff-filter=D", remote_commit, "HEAD"],
            cwd=local_repo,
            text=True,
        ).splitlines()
        unsafe_files = [file for file in changed_files + deleted_files if not self.safe_git_path(file)]
        if unsafe_files:
            raise ValueError(f"unsafe git path in diff: {unsafe_files[0]}")
        return changed_files, deleted_files

    @staticmethod
    def log_changed_files(changed_files, deleted_files):
        common.log_info(f"[info] detect {len(changed_files)} changes and {len(deleted_files)} deleted files")
        for file in changed_files:
            common.log_info(f"[info] add {file}")
        for file in deleted_files:
            common.log_info(f"[info] delete {file}")

    def apply_incremental_tar(self, task):
        remote_tar_path = self.remote_path_join(task.remote_temp_dir, task.tar_filename)
        self.sftp.put(task.local_tar_path, remote_tar_path)
        commands = [
            (
                f"cd {self.quote_remote_path(task.remote_repo)} && "
                f"tar --no-same-owner -xzf {self.quote_remote_path(remote_tar_path)}"
            ),
            f"rm {self.quote_remote_path(remote_tar_path)}",
        ]
        for file in task.deleted_files:
            remote_file = self.remote_path_join(task.remote_repo, file)
            commands.append(f"rm -f {self.quote_remote_path(remote_file)}")
        for cmd in commands:
            self.execute_command(cmd)

    def sync_with_bundle(self, local_repo, remote_repo):
        """
        使用git bundle进行高效同步
        :param local_repo: 本地仓库路径
        :param remote_repo: 远程仓库路径
        """
        common.log_info("[---sync_with_bundle---]")

        local_temp_dir = os.path.join(local_repo, "temp")
        os.makedirs(local_temp_dir, exist_ok=True, mode=0o777)
        remote_temp_dir = self.remote_path_join(remote_repo, "temp")
        self.ensure_remote_dir(remote_temp_dir)

        remote_commit = self.get_remote_commit(remote_repo)
        if not remote_commit:
            common.log_info("remote repo is empty, doing full sync")
            self.full_sync(local_repo, remote_repo)
            return
        remote_commit = self.validate_commit(remote_commit)

        self_commit = self.get_local_commit(local_repo)
        if self_commit == remote_commit:
            common.log_info(
                f"local version [{self_commit[:5]}] same as remote [{remote_commit[:5]}], no need to sync"
            )
            return

        bundle_name = f"update_{time.strftime('%Y%m%d%H%M%S')}.bundle"
        local_bundle_path = os.path.join(local_temp_dir, bundle_name)
        self.execute_command(f"mkdir -p {self.quote_remote_path(self.remote_path_join(remote_repo, '.git'))}")
        self.create_bundle(local_repo, local_bundle_path, remote_commit)
        if not self.bundle_has_data(local_repo, local_bundle_path):
            common.log_warning("[warning] bundle is empty, doing nothing")
            os.remove(local_bundle_path)
            return

        self.apply_bundle(remote_repo, remote_temp_dir, local_bundle_path, bundle_name)
        os.remove(local_bundle_path)
        common.log_info("[---sync_with_bundle_done---]")

    def apply_bundle(self, remote_repo, remote_temp_dir, local_bundle_path, bundle_name):
        remote_bundle_path = self.remote_path_join(remote_temp_dir, bundle_name)
        self.sftp.put(local_bundle_path, remote_bundle_path)
        commands = [
            (
                f"cd {self.quote_remote_path(remote_repo)} && "
                f"git fetch {self.quote_remote_path(remote_bundle_path)} HEAD"
            ),
            f"cd {self.quote_remote_path(remote_repo)} && git reset --hard FETCH_HEAD",
            f"rm {self.quote_remote_path(remote_bundle_path)}",
        ]
        for cmd in commands:
            self.execute_command(cmd)


def cleanup_local_temp(local_repo):
    temp_dir = os.path.join(local_repo, "temp")
    try:
        shutil.rmtree(temp_dir)
    except FileNotFoundError:
        return
    except OSError as err:
        common.log_error(f"[warning] failed to remove temp dir {temp_dir}: {str(err)}")


# 使用示例
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("python ssh_controller.py")
    parser.add_argument(
        "mode",
        choices=["full", "incremental", "bundle"],
        default="bundle",
        help="sync mode",
        nargs="?",
    )
    parser.add_argument(
        "--controller", "-c", type=str, default=None, help="controller ip"
    )
    parser.add_argument(
        "--local_repo", "-l", type=str, default=None, help="local repo path"
    )
    parser.add_argument(
        "--remote_repo", "-r", type=str, default=None, help="remote repo path"
    )
    parser.add_argument(
        "--file", "-f", default=None, help="control json file"
    )

    args = parser.parse_args()
    cfg.load_config(args.file)

    mode = args.mode
    host = args.controller if args.controller is not None else cfg.controller_ip
    user = next(iter(cfg.host_to_user_pair[host]))
    # 使用密码或密钥认证（二选一）
    pswd = cfg.host_to_user_pair[host][user]
    key_filename = cfg.get_key_filename(host, user)

    # 仓库路径
    local_repo = args.local_repo if args.local_repo is not None else cfg.abs_from_path
    remote_repo = args.remote_repo if args.remote_repo is not None else cfg.abs_to_path

    # 创建同步器实例
    sync = GitSync(host, user, password=pswd, key_filename=key_filename)

    try:
        # 建立连接
        sync.connect()

        if mode == "full":
            # 尝试删除远程仓库
            sync.execute_command(f"rm -rf {sync.quote_remote_path(remote_repo)}")

        # 确保远程仓库目录存在
        sync.ensure_remote_dir(remote_repo)

        if mode == "full":
            # 执行全量同步
            sync.full_sync(local_repo, remote_repo)
        elif mode == "incremental":
            # 执行增量同步
            sync.incremental_sync(local_repo, remote_repo)
        else:
            # 使用git bundle进行高效同步
            sync.sync_with_bundle(local_repo, remote_repo)
    except KeyboardInterrupt:
        common.log_warning("[warning] interrupted by user")
    except paramiko.ssh_exception.NoValidConnectionsError:
        common.log_error("[error] no valid connections")
    except (paramiko.SSHException, OSError, RuntimeError, ValueError) as err:
        common.log_error(f"[error] {str(err)}")
        raise
    finally:
        cleanup_local_temp(local_repo)
        sync.disconnect()
