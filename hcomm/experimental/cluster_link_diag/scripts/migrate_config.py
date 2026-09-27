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

import argparse
import json
import logging
import sys


def _v1_hosts(deploy):
    hosts = []
    for ip, entry in deploy.get("host_to_user_pair", {}).items():
        if not isinstance(entry, dict):
            raise ValueError(f"deploy.host_to_user_pair.{ip} must be object")
        host = {"id": ip, "ip": ip}
        if "User" in entry or "Password" in entry or "key_filename" in entry:
            host["user"] = entry.get("User", "root")
            if entry.get("key_filename"):
                host["ssh_key"] = entry["key_filename"]
            if entry.get("Password"):
                host["password_env"] = f"DISP_PROBE_PASS_{ip.replace('.', '_').replace('-', '_')}"
        else:
            user = next(iter(entry), "root")
            host["user"] = user
        hosts.append(host)
    if not hosts:
        raise ValueError("missing deploy.host_to_user_pair")
    return hosts


def _controller_id(deploy, hosts):
    controller = deploy.get("controller", {})
    if isinstance(controller, dict) and controller:
        controller_ip = next(iter(controller))
        if controller_ip:
            return controller_ip
    return hosts[0]["id"]


def migrate_v1_to_v2(config):
    if config.get("schema_version", 1) == 2:
        return config

    deploy = config.get("deploy", {})
    hosts = _v1_hosts(deploy)
    topology = config.get("probe_topo", {}).get("tracert", {})
    pingpong = config.get("probe_controller", {}).get("pingpong", {})
    probe_scope = config.get("probe_scope", {})
    if not isinstance(probe_scope, dict) or not probe_scope:
        raise ValueError("missing probe_scope")

    scope = {}
    for host_id, devices in probe_scope.items():
        scope[host_id] = {"devices": devices}

    new_deploy = {
        "to_path": deploy.get("to_path", "/root/disp_probe"),
        "default_ssh_port": deploy.get("default_ssh_port", 22),
        "default_timeout": deploy.get("default_timeout", 5),
    }
    if deploy.get("from_path") is not None:
        new_deploy["from_path"] = deploy["from_path"]
    if deploy.get("control_topo"):
        new_deploy["control_topo"] = deploy["control_topo"]

    return {
        "schema_version": 2,
        "controller": _controller_id(deploy, hosts),
        "deploy": new_deploy,
        "hosts": hosts,
        "probe": {
            "scope": scope,
            "topology": topology,
            "pingpong": pingpong,
        },
    }


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    parser = argparse.ArgumentParser(description="Migrate disp_probe JSON config from schema v1 to v2")
    parser.add_argument("input", help="input config JSON")
    parser.add_argument("output", nargs="?", help="output config JSON; defaults to stdout")
    args = parser.parse_args()

    with open(args.input, "r") as f:
        migrated = migrate_v1_to_v2(json.load(f))

    text = json.dumps(migrated, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        with open(args.output, "w") as f:
            f.write(text)
    else:
        logging.info("%s", text.rstrip())


if __name__ == "__main__":
    main()
