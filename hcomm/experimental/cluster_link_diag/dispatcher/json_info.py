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

import json
import os
import re
import shlex
from types import SimpleNamespace

import paramiko

import dispatcher_common as common

DEFAULT_INFO_JSON = "./control_json/910b2_info.json"
CONFIG_ENV = "DISP_PROBE_CONFIG"
CONFIG_NAMES = (
    "info_json",
    "all_info",
    "deploy_info",
    "default_ssh_port",
    "default_timeout",
    "host_to_user_pair",
    "host_to_su_pswd",
    "host_to_key_filename",
    "control_topo",
    "controller_ip",
    "controller_user",
    "controller_pswd",
    "controller",
    "to_path",
    "from_path",
    "abs_to_path",
    "abs_from_path",
)
CONFIG_STATE = {
    "info_json": None,
    "all_info": {},
    "deploy_info": {},
    "default_ssh_port": 22,
    "default_timeout": 5,
    "host_to_user_pair": {},
    "host_to_su_pswd": {},
    "host_to_key_filename": {},
    "control_topo": [],
    "controller_ip": None,
    "controller_user": None,
    "controller_pswd": "",
    "controller": {},
    "to_path": "",
    "from_path": "",
    "abs_to_path": "",
    "abs_from_path": "",
    "loaded_config_path": None,
}


def __getattr__(name):
    if name in CONFIG_STATE:
        return CONFIG_STATE[name]
    if name == "_loaded_config_path":
        return CONFIG_STATE["loaded_config_path"]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def normalize_host_entries(raw_hosts):
    normalized_user_pair = {}
    normalized_su_pswd = {}
    normalized_key_filename = {}
    for host, entry in raw_hosts.items():
        if not isinstance(entry, dict):
            raise ValueError(f"deploy.host_to_user_pair.{host} must be object")

        if "User" in entry or "Password" in entry or "key_filename" in entry:
            user = entry.get("User", "root")
            password = entry.get("Password", "")
            normalized_user_pair[host] = {user: password}
            normalized_su_pswd[host] = entry.get("su_password", password)
            key_filename = entry.get("key_filename", "")
            if key_filename:
                normalized_key_filename[host] = {user: key_filename}
        else:
            normalized_user_pair[host] = entry
    return normalized_user_pair, normalized_su_pswd, normalized_key_filename


def _read_password_from_host_entry(entry):
    if entry.get("password_env"):
        env_name = entry["password_env"]
        if not isinstance(env_name, str) or len(env_name) == 0:
            raise ValueError("password_env must be non-empty string")
        return os.environ.get(env_name, "")
    return entry.get("password", entry.get("Password", ""))


def _translate_control_topo_hosts(control_topo_value, id_to_ip):
    if control_topo_value is None:
        return None
    if isinstance(control_topo_value, str):
        return id_to_ip.get(control_topo_value, control_topo_value)
    if isinstance(control_topo_value, list):
        return [_translate_control_topo_hosts(item, id_to_ip) for item in control_topo_value]
    if isinstance(control_topo_value, dict):
        return {
            id_to_ip.get(host_id, host_id): _translate_control_topo_hosts(children, id_to_ip)
            for host_id, children in control_topo_value.items()
        }
    raise ValueError("deploy.control_topo must be string, array, object, or null")


def normalize_v2_hosts(raw_hosts):
    if not isinstance(raw_hosts, list) or len(raw_hosts) == 0:
        raise ValueError("hosts must be a non-empty array")

    id_to_ip = {}
    seen_ips = set()
    normalized_user_pair = {}
    normalized_su_pswd = {}
    normalized_key_filename = {}

    for index, entry in enumerate(raw_hosts):
        field = f"hosts[{index}]"
        if not isinstance(entry, dict):
            raise ValueError(f"{field} must be object")
        host_id = entry.get("id")
        host_ip = entry.get("ip")
        if not isinstance(host_id, str) or len(host_id) == 0:
            raise ValueError(f"{field}.id must be non-empty string")
        if not isinstance(host_ip, str) or len(host_ip) == 0:
            raise ValueError(f"{field}.ip must be non-empty string")
        if host_id in id_to_ip:
            raise ValueError(f"duplicate host id: {host_id}")
        if host_ip in seen_ips:
            raise ValueError(f"duplicate host ip: {host_ip}")

        id_to_ip[host_id] = host_ip
        seen_ips.add(host_ip)
        user = entry.get("user", entry.get("User", "root"))
        password = _read_password_from_host_entry(entry)
        normalized_user_pair[host_ip] = {user: password}
        normalized_su_pswd[host_ip] = entry.get("su_password", password)
        key_filename = entry.get("ssh_key", entry.get("key_filename", ""))
        if key_filename:
            normalized_key_filename[host_ip] = {user: key_filename}

    return id_to_ip, normalized_user_pair, normalized_su_pswd, normalized_key_filename


def _read_json_config(path):
    info_stat = os.stat(path)
    if info_stat.st_mode & 0o077:
        common.log_error(f"[security][warning] {path} is readable or writable by group/others; run: chmod 600 {path}")
    with open(path, "r") as f:
        return json.load(f)


def _config_get(loaded_deploy_info, key, default=None, required=False):
    if key in loaded_deploy_info:
        return loaded_deploy_info[key]
    if required:
        raise KeyError(f"deploy.{key}")
    return default


def _make_loaded_config_values(config_context):
    host_to_user_pair_value = config_context["host_to_user_pair"]
    controller_ip_value = config_context["controller_ip"]
    controller_user_value = config_context["controller_user"]
    return {
        "host_to_user_pair": host_to_user_pair_value,
        "host_to_su_pswd": config_context["host_to_su_pswd"],
        "host_to_key_filename": config_context["host_to_key_filename"],
        "control_topo": config_context["control_topo"],
        "controller_ip": controller_ip_value,
        "controller_user": controller_user_value,
        "controller_pswd": host_to_user_pair_value[controller_ip_value][controller_user_value],
        "controller": {controller_ip_value: controller_user_value},
    }


def _load_v2_config(loaded_all_info, loaded_deploy_info):
    id_to_ip, loaded_host_to_user_pair, loaded_host_to_su_pswd, loaded_host_to_key_filename = normalize_v2_hosts(
        loaded_all_info.get("hosts")
    )
    loaded_control_topo = _config_get(loaded_deploy_info, "control_topo", [])
    loaded_control_topo = (
        list(loaded_host_to_user_pair.keys())
        if loaded_control_topo is None or len(loaded_control_topo) == 0
        else _translate_control_topo_hosts(loaded_control_topo, id_to_ip)
    )
    loaded_controller_id = loaded_all_info.get("controller")
    if not isinstance(loaded_controller_id, str) or len(loaded_controller_id) == 0:
        raise ValueError("controller must be a non-empty host id string for schema_version=2")
    if loaded_controller_id not in id_to_ip:
        raise ValueError(f"controller references unknown host id: {loaded_controller_id}")
    loaded_controller_ip = id_to_ip[loaded_controller_id]
    loaded_controller_user = next(iter(loaded_host_to_user_pair[loaded_controller_ip]))
    return _make_loaded_config_values({
        "host_to_user_pair": loaded_host_to_user_pair,
        "host_to_su_pswd": loaded_host_to_su_pswd,
        "host_to_key_filename": loaded_host_to_key_filename,
        "control_topo": loaded_control_topo,
        "controller_ip": loaded_controller_ip,
        "controller_user": loaded_controller_user,
    })


def _merge_v1_host_groups(loaded_deploy_info, loaded_host_to_user_pair, loaded_host_to_su_pswd, loaded_key_filename):
    if "hosts_to_user_pair" not in loaded_deploy_info:
        return
    for item in _config_get(loaded_deploy_info, "hosts_to_user_pair"):
        value = item["user_pair"]
        for key in item["hosts"]:
            extra_user_pair, extra_su_pswd, extra_key_filename = normalize_host_entries({key: value})
            loaded_host_to_user_pair.update(extra_user_pair)
            loaded_host_to_su_pswd.update(extra_su_pswd)
            loaded_key_filename.update(extra_key_filename)


def _load_v1_config(loaded_deploy_info):
    raw_host_to_user_pair = _config_get(loaded_deploy_info, "host_to_user_pair", required=True)
    loaded_host_to_user_pair, loaded_host_to_su_pswd, loaded_host_to_key_filename = normalize_host_entries(
        raw_host_to_user_pair
    )
    _merge_v1_host_groups(
        loaded_deploy_info,
        loaded_host_to_user_pair,
        loaded_host_to_su_pswd,
        loaded_host_to_key_filename,
    )
    loaded_control_topo = _config_get(loaded_deploy_info, "control_topo", [])
    loaded_control_topo = (
        list(loaded_host_to_user_pair.keys())
        if loaded_control_topo is None or len(loaded_control_topo) == 0
        else loaded_control_topo
    )
    loaded_controller = _config_get(loaded_deploy_info, "controller", {})
    loaded_controller_ip = None if not loaded_controller else next(iter(loaded_controller))
    loaded_controller_ip = (
        next(iter(loaded_host_to_user_pair))
        if loaded_controller_ip is None or len(loaded_controller_ip) == 0
        else loaded_controller_ip
    )
    loaded_controller_user = (
        next(iter(loaded_host_to_user_pair[loaded_controller_ip]))
        if loaded_controller_ip not in loaded_controller or len(loaded_controller[loaded_controller_ip]) == 0
        else loaded_controller[loaded_controller_ip]
    )
    return _make_loaded_config_values({
        "host_to_user_pair": loaded_host_to_user_pair,
        "host_to_su_pswd": loaded_host_to_su_pswd,
        "host_to_key_filename": loaded_host_to_key_filename,
        "control_topo": loaded_control_topo,
        "controller_ip": loaded_controller_ip,
        "controller_user": loaded_controller_user,
    })


def _complete_config_values(values, loaded_deploy_info):
    values["host_to_su_pswd"].update(_config_get(loaded_deploy_info, "host_to_su_pswd", {}))
    values["host_to_key_filename"].update(_config_get(loaded_deploy_info, "host_to_key_filename", {}))
    for host, pairs in values["host_to_user_pair"].items():
        if host not in values["host_to_su_pswd"] or len(values["host_to_su_pswd"][host]) == 0:
            values["host_to_su_pswd"][host] = next(iter(pairs.values()))
    values["to_path"] = _config_get(loaded_deploy_info, "to_path", required=True)
    values["from_path"] = _config_get(loaded_deploy_info, "from_path", ".")
    values["abs_to_path"] = values["to_path"].replace(
        "~", f"/home/{values['controller_user']}" if values["controller_user"] != "root" else "/root"
    )
    values["abs_from_path"] = values["from_path"].replace("~", os.path.expanduser("~"), 1)
    return values


def _build_config_values(path):
    loaded_all_info = _read_json_config(path)
    loaded_schema_version = loaded_all_info.get("schema_version", 1)
    if loaded_schema_version not in (1, 2):
        raise ValueError(f"unsupported schema_version: {loaded_schema_version}")
    loaded_deploy_info = loaded_all_info.get("deploy", {})
    values = _load_v2_config(loaded_all_info, loaded_deploy_info) if loaded_schema_version == 2 else _load_v1_config(
        loaded_deploy_info
    )
    values.update({
        "info_json": path,
        "all_info": loaded_all_info,
        "deploy_info": loaded_deploy_info,
        "default_ssh_port": _config_get(loaded_deploy_info, "default_ssh_port", 22),
        "default_timeout": _config_get(loaded_deploy_info, "default_timeout", 5),
    })
    return _complete_config_values(values, loaded_deploy_info)


def _make_config_namespace(values):
    return SimpleNamespace(
        info_json=values["info_json"],
        all_info=values["all_info"],
        deploy_info=values["deploy_info"],
        default_ssh_port=values["default_ssh_port"],
        default_timeout=values["default_timeout"],
        host_to_user_pair=values["host_to_user_pair"],
        host_to_su_pswd=values["host_to_su_pswd"],
        host_to_key_filename=values["host_to_key_filename"],
        control_topo=values["control_topo"],
        controller_ip=values["controller_ip"],
        controller_user=values["controller_user"],
        controller_pswd=values["controller_pswd"],
        controller=values["controller"],
        to_path=values["to_path"],
        from_path=values["from_path"],
        abs_to_path=values["abs_to_path"],
        abs_from_path=values["abs_from_path"],
    )


def _publish_config_values(values, path):
    for name in CONFIG_NAMES:
        CONFIG_STATE[name] = values[name]
    CONFIG_STATE["loaded_config_path"] = path


def load_config(path=None):
    path = path or os.environ.get(CONFIG_ENV, DEFAULT_INFO_JSON)
    try:
        values = _build_config_values(path)
    except Exception as err:
        raise RuntimeError(f"解析json文件错误: {err}") from err
    _publish_config_values(values, path)
    return _make_config_namespace(values)


def ensure_config_loaded(path=None):
    expected_path = path or os.environ.get(CONFIG_ENV, DEFAULT_INFO_JSON)
    if CONFIG_STATE["loaded_config_path"] != expected_path:
        return load_config(expected_path)
    config_values = {}
    for name in CONFIG_NAMES:
        config_values[name] = CONFIG_STATE[name]
    return SimpleNamespace(**config_values)


def ssh_cmd(user, host, cmd, pswd=None):
    # 现在无法合理地使用sshpass,因此pswd传入也没有意义,现在已经转向paramiko
    return f'ssh {user}@{host} "{cmd}"'


# 由于正则表达式难以理解,只需要知道上面的替代关系即可
def _match_command_separator(cmd, index):
    separators = ["||", "&&", "|", "&", ";"]
    for sep in separators:
        if cmd[index:index + len(sep)] != sep:
            continue
        if sep != "&":
            return sep
        prev_char = cmd[index - 1] if index > 0 else ""
        next_char = cmd[index + 1] if index + 1 < len(cmd) else ""
        if prev_char not in [">", "<"] and next_char != ">":
            return sep
    return None


def _append_token(tokens, current):
    if current:
        tokens.append("".join(current).strip())
    return []


def _update_quote_state(char, in_single_quote, in_double_quote):
    if char == "'" and not in_double_quote:
        return True, in_double_quote, True
    if char == '"' and not in_single_quote:
        return in_single_quote, True, True
    return in_single_quote, in_double_quote, False


def _consume_escape(cmd, index, state):
    char = cmd[index]
    if char == "\\" and not state["escape_next"]:
        state["escape_next"] = True
        state["current"].append(char)
        return index + 1, True
    if state["escape_next"]:
        state["current"].append(char)
        state["escape_next"] = False
        return index + 1, True
    return index, False


def _consume_quoted_char(cmd, index, state):
    if not state["in_single_quote"] and not state["in_double_quote"]:
        return index, False
    char = cmd[index]
    if char == "'" and state["in_single_quote"]:
        state["in_single_quote"] = False
    elif char == '"' and state["in_double_quote"]:
        state["in_double_quote"] = False
    state["current"].append(char)
    return index + 1, True


def _consume_quote_start(cmd, index, state):
    in_single, in_double, quote_changed = _update_quote_state(
        cmd[index], state["in_single_quote"], state["in_double_quote"]
    )
    state["in_single_quote"] = in_single
    state["in_double_quote"] = in_double
    if quote_changed:
        state["current"].append(cmd[index])
        return index + 1, True
    return index, False


def _consume_separator(cmd, index, state):
    found_separator = _match_command_separator(cmd, index)
    if not found_separator:
        return index, False
    state["current"] = _append_token(state["tokens"], state["current"])
    state["tokens"].append(found_separator)
    return index + len(found_separator), True


def _consume_command_char(cmd, index, state):
    for consumer in (_consume_escape, _consume_quoted_char, _consume_quote_start, _consume_separator):
        next_index, consumed = consumer(cmd, index, state)
        if consumed:
            return next_index
    state["current"].append(cmd[index])
    return index + 1


def split_ignoring_quotes(cmd):
    """分割命令，但忽略引号内的分隔符"""
    state = {
        "tokens": [],
        "current": [],
        "in_single_quote": False,
        "in_double_quote": False,
        "escape_next": False,
    }
    i = 0
    while i < len(cmd):
        i = _consume_command_char(cmd, i, state)
    _append_token(state["tokens"], state["current"])
    return state["tokens"]


def split_token_parts(token):
    parts = []
    current_part = []
    in_quote = False
    quote_char = None
    for char in token:
        if char in ['"', "'"] and not in_quote:
            in_quote = True
            quote_char = char
            current_part.append(char)
        elif char == quote_char and in_quote:
            in_quote = False
            current_part.append(char)
        elif char == " " and not in_quote:
            if current_part:
                parts.append("".join(current_part))
                current_part = []
        else:
            current_part.append(char)
    if current_part:
        parts.append("".join(current_part))
    return parts


def sudo_cmd(cmd, pswd):
    tokens = split_ignoring_quotes(cmd)
    processed_tokens = []
    for token in tokens:
        # 检查是否为分隔符
        if token in ["||", "&&", "|", "&", ";"]:
            processed_tokens.append(token)
        else:
            parts = split_token_parts(token)
            if parts and parts[0] == "sudo" and len(parts) > 1:
                new_cmd = f"sudo -S -p '' {' '.join(parts[1:])}"
                processed_tokens.append(new_cmd)
            else:
                processed_tokens.append(token)
    # 拼接最终命令
    return " ".join(processed_tokens)


def su_cmd(cmd, pswd):
    tokens = split_ignoring_quotes(cmd)
    processed_tokens = []
    for token in tokens:
        # 检查是否为分隔符
        if token in ["||", "&&", "|", "&", ";"]:
            processed_tokens.append(token)
        else:
            parts = split_token_parts(token)
            if parts and parts[0] == "su" and len(parts) > 1:
                # 获取su后面的命令部分
                su_command = " ".join(parts[1:])
                new_cmd = f"su - root -c {shlex.quote(su_command)}"
                processed_tokens.append(new_cmd)
            else:
                processed_tokens.append(token)
    # 拼接最终命令
    return " ".join(processed_tokens)


def clean_ip(ip):
    if ip == "localhost":
        return ip
    # 匹配 IPv4 地址（允许后面紧跟非数字字符）
    ip_segment = r"(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)"
    pattern = rf"{ip_segment}\.{ip_segment}\.{ip_segment}\.({ip_segment})(?!\d)"

    match = re.search(pattern, ip)
    if match:
        return match.group(0)
    return ""


def get_key_filename(host, user):
    ensure_config_loaded()
    key_filename = ""
    key_filenames = CONFIG_STATE["host_to_key_filename"]
    if host in key_filenames:
        key_config = key_filenames[host]
    else:
        key_config = key_filenames.get(clean_ip(host), "")

    if isinstance(key_config, dict):
        key_filename = key_config.get(user, "")
    else:
        key_filename = key_config

    if not key_filename:
        return None

    return key_filename.replace(
        "~", f"/home/{user}" if user != "root" else "/root"
    )


def setup_ssh_host_key_policy(client):
    client.load_system_host_keys()
    known_hosts = os.path.expanduser("~/.ssh/known_hosts")
    if os.path.exists(known_hosts):
        client.load_host_keys(known_hosts)
    if os.environ.get("DISP_PROBE_SSH_AUTO_ADD_HOST_KEY") == "1":
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    else:
        client.set_missing_host_key_policy(paramiko.RejectPolicy())


def connect_ssh_client(connection):
    ensure_config_loaded()
    setup_ssh_host_key_policy(connection.client)
    key_filename = get_key_filename(connection.host_name, connection.user)
    kwargs = {
        "hostname": connection.host_ip,
        "username": connection.user,
        "timeout": CONFIG_STATE["default_timeout"],
        "allow_agent": True,
        "look_for_keys": True,
    }
    if connection.pswd:
        kwargs["password"] = connection.pswd
    if key_filename:
        kwargs["key_filename"] = key_filename
    if connection.sock is not None:
        kwargs["sock"] = connection.sock
    connection.client.connect(**kwargs)


def make_cmd_abs(cmd, user):
    ensure_config_loaded()
    base_path = CONFIG_STATE["to_path"]
    project_abs_path = base_path.replace('~/', ('/root' if user == "root" else '/home/' + user) + '/') + '/'
    return cmd.replace('./', project_abs_path)


if __name__ == "__main__":
    load_config()
    common.log_info(sudo_cmd("whoami && sudo whoami && whoami", CONFIG_STATE["controller_pswd"]))
    common.log_info(su_cmd("whoami && su 'whoami && whoami' && su whoami", CONFIG_STATE["controller_pswd"]))
