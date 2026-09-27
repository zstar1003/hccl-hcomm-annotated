#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# 根据device_links.yaml配置文件，生成server/pod粒度的topo.json和rootinfo.json
# 功能：
#   1. 解析yaml文件，获取device端口分配表和连接关系
#   2. 生成topo.json：描述server/pod内所有NPU设备的连接关系
#   3. 生成rootinfo.json：为每个device的portGroup分配EID和端口信息
#   4. EID分配规则：按照IP地址分配规则(AA.BB.CC.DD)分配IP，再转换为EID
#      - AA = 192 (superpod段，单server/pod固定192)
#      - BB = 1   (server段，单server/pod固定1，排除192.0.0.0~192.0.0.24段)
#      - Host IP: 192.1.0.0 (固定为AA.BB.0.0，仅供host使用)
#      - Device port IPs: 192.1.0.1, 192.1.0.2, ... 依次递增 (从0.1开始，d2h端口也从此段分配)
#      - IP→EID: IPv4地址放入16字节EID末尾(网络字节序)，前12字节为0
#
# 支持两种端口分配格式:
#   V1 (pin_map): {pin_map: {port_num_per_die, die_list: [{die_id, peer2peer, peer2net, h2d}], port_group}}
#   V2 (device_ports_allocate_map): [{die_id, pin_map: [0,2,2,1,...]}]
#
# 支持三种连接描述方法 (links列表中可共存):
#   fullmesh: link_mode + connections/device_groups
#   enum:     link_mode + device_to_device_links/device_to_switch_links
#   matrix:   link_mode + device_to_device_links(矩阵)/device_to_switch_links

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

error_exit() {
    echo -e "${RED}[错误] $1${NC}" >&2
    exit 1
}

success_msg() {
    echo -e "${GREEN}[成功] $1${NC}"
}

info_msg() {
    echo -e "${BLUE}[信息] $1${NC}"
}

warn_msg() {
    echo -e "${YELLOW}[警告] $1${NC}"
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "${SCRIPT_DIR}")"

SERVER_CONFIG_DIR="${BASE_DIR}/config/server_or_pod"
DEFAULT_OUTPUT_DIR="${BASE_DIR}/config/network/server_or_pod"

resolve_path() {
    local input="$1"
    local default_dir="$2"
    if [[ "$input" = /* ]]; then
        if [[ ! -f "$input" ]]; then
            error_exit "文件不存在: $input"
        fi
        realpath "$input"
    else
        local full_path="${default_dir}/${input}"
        if [[ ! -f "$full_path" ]]; then
            error_exit "文件不存在: $full_path (默认路径: ${default_dir})"
        fi
        realpath "$full_path"
    fi
}

show_usage() {
    cat << EOF
用法: $(basename "$0") [OPTIONS] <server_topo_yaml>

参数:
    server_topo_yaml   yaml配置文件路径或文件名 (支持V1/V2格式)
                       绝对路径: 直接使用
                       文件名/相对路径: 在 ${SERVER_CONFIG_DIR} 下查找

选项:
    -o, --output DIR         指定输出目录 (默认: ${DEFAULT_OUTPUT_DIR})
    -n, --name NAME          指定输出文件名前缀 (默认: 从yaml文件name字段获取)
    -b, --bare               输出文件不带前缀: topo.json 和 hccl_rootinfo.json
    -h, --help               显示帮助信息

说明:
    解析yaml配置文件，生成topo.json和rootinfo.json

    端口分配格式 (V1 - pin_map):
      pin_map:
        port_num_per_die: 9
        die_list: [{die_id, peer2peer, peer2net, h2d}]
        port_group: [{layer, port: [{die_id, port}]}]

    端口分配格式 (V2 - device_ports_allocate_map):
      device_ports_allocate_map: [{die_id, pin_map: [0,2,2,1,...]}]
      port_group: [{layer, ports: ["0/1", ...]}] 或 [{layer, port: [{die_id, port}]}]

    连接描述方法 (links列表中可共存):
      fullmesh: link_mode + connections/device_groups
      enum:     link_mode + device_to_device_links/device_to_switch_links
      matrix:   link_mode + device_to_device_links(矩阵)/device_to_switch_links

示例:
    $(basename "$0") ascend950_server_topo_normal.yaml
    $(basename "$0") /absolute/path/to/device_links.yaml
    $(basename "$0") -o /tmp/output ascend950_server_topo_normal.yaml -n ascend950_demo

EOF
}

OUTPUT_DIR=""
OUTPUT_NAME=""
BARE_MODE=0

YAML_INPUT=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_usage
            exit 0
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -n|--name)
            OUTPUT_NAME="$2"
            shift 2
            ;;
        -b|--bare)
            BARE_MODE=1
            shift
            ;;
        *)
            YAML_INPUT="$1"
            shift
            ;;
    esac
done

if [[ -z "$YAML_INPUT" ]]; then
    error_exit "请指定yaml配置文件路径\n$(show_usage)"
fi

YAML_PATH="$(resolve_path "$YAML_INPUT" "$SERVER_CONFIG_DIR")"

if [[ -z "$OUTPUT_DIR" ]]; then
    YAML_BASENAME=$(basename "$YAML_PATH" .yaml)
    OUTPUT_DIR="${DEFAULT_OUTPUT_DIR}/${YAML_BASENAME}"
fi

echo "=========================================="
echo "  Server/Pod拓扑描述文件生成工具"
echo "=========================================="
echo ""
echo "输入文件: $YAML_PATH"
echo "输出目录: $OUTPUT_DIR"
echo ""

mkdir -p "$OUTPUT_DIR"

export YAML_PATH
export OUTPUT_DIR
export OUTPUT_NAME
export BARE_MODE

python3 << 'PYTHON_SCRIPT'
import json
import os
import sys
import socket

YAML_PATH = os.environ.get('YAML_PATH', '')
OUTPUT_DIR = os.environ.get('OUTPUT_DIR', '')
OUTPUT_NAME = os.environ.get('OUTPUT_NAME', '')
BARE_MODE = os.environ.get('BARE_MODE', '0') == '1'

try:
    import yaml
except ImportError:
    print("[错误] 需要PyYAML库，请安装: pip install pyyaml")
    sys.exit(1)

with open(YAML_PATH, 'r') as f:
    config = yaml.safe_load(f)

soc_version = config.get('soc_version', 'unknown')
device_num = config.get('device_num', 0)
name = config.get('name', 'topo')

if OUTPUT_NAME:
    out_name = OUTPUT_NAME
else:
    out_name = name

if BARE_MODE:
    topo_filename = 'topo.json'
    rootinfo_filename = 'hccl_rootinfo.json'
else:
    topo_filename = f'{out_name}_topo.json'
    rootinfo_filename = f'{out_name}_rootinfo.json'

if device_num == 0:
    print("[错误] device_num为0或未定义")
    sys.exit(1)

print(f"[信息] 芯片类型: {soc_version}")
print(f"[信息] Device数量: {device_num}")
print(f"[信息] 输出文件名前缀: {out_name}")
print("")

# ============================================================
# 1. 解析端口分配表 (自动检测V1/V2格式, 支持group_id扩展)
# ============================================================
pin_map_raw = config.get('pin_map', {})
dpam_raw = config.get('device_ports_allocate_map', [])
dp_raw = config.get('device_ports', [])

# device_group解析: 构建 device_id → group_id 映射
device_group_raw = config.get('device_group', [])
device_to_group = {}  # device_id → group_id
group_to_devices = {}  # group_id → [device_list]
_default_group_id = 1

if device_group_raw:
    print(f"[信息] 检测到device_group配置, 共{len(device_group_raw)}个分组")
    _seen_group_ids = set()
    for dg in device_group_raw:
        gid = dg.get('group_id', _default_group_id)
        # 校验: group_id不能重复
        if gid in _seen_group_ids:
            print(f"[错误] device_group中存在重复的group_id={gid}，请检查配置文件")
            sys.exit(1)
        _seen_group_ids.add(gid)
        dev_list = dg.get('device_list', [])
        group_to_devices[gid] = dev_list
        for dev_id in dev_list:
            device_to_group[int(dev_id)] = gid
        print(f"       group_id={gid}: devices={dev_list}")
else:
    # 没有device_group配置, 默认所有device为一组
    group_to_devices[_default_group_id] = list(range(device_num))
    for dev_id in range(device_num):
        device_to_group[dev_id] = _default_group_id

die_list = []  # 默认die_list (向后兼容)
port_num_per_die = 0
port_group_raw = None

# group_id → die_list 映射
group_die_lists = {}

if dpam_raw:
    print("[信息] 检测到V2端口分配格式 (device_ports_allocate_map)")
    _current_group_id = _default_group_id
    _temp_entries = []

    for d in dpam_raw:
        if isinstance(d, dict):
            # 检查是否是group_id元数据条目
            if 'group_id' in d and 'die_id' not in d and 'pin_map' not in d:
                # 保存之前的entries到对应的group
                if _temp_entries:
                    group_die_lists[_current_group_id] = _temp_entries
                _current_group_id = d['group_id']
                _temp_entries = []
                continue
            # 兼容旧的id字段
            if 'id' in d and 'die_id' not in d and 'pin_map' not in d:
                if _temp_entries:
                    group_die_lists[_current_group_id] = _temp_entries
                _current_group_id = d['id']
                _temp_entries = []
                continue
            # die条目
            die_id = d.get('die_id', 0)
            pin_map_vals = d.get('pin_map', d.get('port', []))
        elif isinstance(d, (list, tuple)):
            die_id = len(_temp_entries)
            pin_map_vals = list(d)
        else:
            continue

        p2p = [i for i, v in enumerate(pin_map_vals) if v == 1]
        p2n = [i for i, v in enumerate(pin_map_vals) if v == 2]
        h2d = [i for i, v in enumerate(pin_map_vals) if v == 3]

        _temp_entries.append({
            'die_id': die_id,
            'peer2peer': p2p,
            'peer2net': p2n,
            'h2d': h2d
        })

    # 保存最后一组
    if _temp_entries:
        group_die_lists[_current_group_id] = _temp_entries

    # 默认die_list (向后兼容)
    if _default_group_id in group_die_lists:
        die_list = group_die_lists[_default_group_id]
    elif group_die_lists:
        first_gid = min(group_die_lists.keys())
        die_list = group_die_lists[first_gid]

    # 打印信息
    for gid, entries in group_die_lists.items():
        dev_list_info = group_to_devices.get(gid, '全部(默认)')
        print(f"[信息] 端口分配表 group_id={gid}, 适用device: {dev_list_info}")
        for d in entries:
            print(f"       die{d['die_id']}: PEER2PEER={d['peer2peer']}, PEER2NET={d['peer2net']}, H2D={d['h2d']}")

    if group_die_lists:
        first_entries = group_die_lists[min(group_die_lists.keys())]
        if first_entries:
            all_ports = set()
            for d in first_entries:
                all_ports.update(d['peer2peer'])
                all_ports.update(d['peer2net'])
                all_ports.update(d['h2d'])
            if all_ports:
                port_num_per_die = max(all_ports) + 1
elif dp_raw:
    print("[信息] 检测到V2端口分配格式 (device_ports)")
    for d in dp_raw:
        if isinstance(d, dict):
            die_id = d.get('die_id', 0)
            pin_map_vals = d.get('pin_map', d.get('port', []))
        elif isinstance(d, (list, tuple)):
            die_id = len(die_list)
            pin_map_vals = list(d)
        else:
            continue

        p2p = [i for i, v in enumerate(pin_map_vals) if v == 1]
        p2n = [i for i, v in enumerate(pin_map_vals) if v == 2]
        h2d = [i for i, v in enumerate(pin_map_vals) if v == 3]

        die_list.append({
            'die_id': die_id,
            'peer2peer': p2p,
            'peer2net': p2n,
            'h2d': h2d
        })

    if dp_raw:
        first = dp_raw[0]
        if isinstance(first, dict):
            port_num_per_die = len(first.get('pin_map', first.get('port', [])))
    # dp_raw格式: 将die_list保存到默认group
    if die_list:
        group_die_lists[_default_group_id] = die_list
elif pin_map_raw and isinstance(pin_map_raw, dict):
    print("[信息] 检测到V1端口分配格式 (pin_map)")
    port_num_per_die = pin_map_raw.get('port_num_per_die', 0)
    die_list_raw = pin_map_raw.get('die_list', [])
    port_group_raw = pin_map_raw.get('port_group', None)

    for d in die_list_raw:
        die_id = d.get('die_id', 0)
        p2p = d.get('peer2peer', [])
        p2n = d.get('peer2net', [])
        h2d = d.get('h2d', [])
        die_list.append({
            'die_id': die_id,
            'peer2peer': p2p,
            'peer2net': p2n if isinstance(p2n, list) and (not p2n or isinstance(p2n[0], int)) else [],
            'h2d': h2d
        })
    # V1格式: 将die_list保存到默认group
    if die_list:
        group_die_lists[_default_group_id] = die_list

has_h2d = any(len(d['h2d']) > 0 for d in die_list)

print(f"[信息] 端口分配表: port_num_per_die={port_num_per_die}, die数={len(die_list)}")
for d in die_list:
    print(f"       die{d['die_id']}: PEER2PEER={d['peer2peer']}, PEER2NET={d['peer2net']}, H2D={d['h2d']}")
if len(group_die_lists) > 1:
    print(f"[信息] 多端口分配表已启用, 共{len(group_die_lists)}个不同的分组")

# ============================================================
# 2. 解析port_group (端口绑定关系, 支持group_id扩展)
# ============================================================
if port_group_raw is None:
    port_group_raw = config.get('port_group', [])

# group_id → {layer: entries} 映射
group_port_groups = {}
_current_group_id = _default_group_id
_temp_pg_entries = {}
_temp_d2h_layers = set()

for pg in port_group_raw:
    # 检查是否是group_id元数据条目
    if isinstance(pg, dict) and 'group_id' in pg and 'layer' not in pg:
        # 保存之前的entries到对应的group
        if _temp_pg_entries:
            group_port_groups[_current_group_id] = {'entries': _temp_pg_entries, 'd2h_layers': _temp_d2h_layers}
        _current_group_id = pg['group_id']
        _temp_pg_entries = {}
        _temp_d2h_layers = set()
        continue

    layer = pg.get('layer', 0)
    ports = pg.get('ports', pg.get('port', []))
    pg_entries = []

    if isinstance(ports, list) and ports and isinstance(ports[0], str):
        for port_str in ports:
            if port_str == 'd2h':
                _temp_d2h_layers.add(layer)
                continue
            parts = port_str.split('/')
            if len(parts) == 2:
                pg_entries.append({
                    'die_id': int(parts[0]),
                    'port': [int(parts[1])]
                })
    elif isinstance(ports, dict):
        pg_entries.append(ports)
    elif isinstance(ports, list):
        for p in ports:
            if isinstance(p, str):
                if p == 'd2h':
                    _temp_d2h_layers.add(layer)
                    continue
                parts = p.split('/')
                if len(parts) == 2:
                    pg_entries.append({
                        'die_id': int(parts[0]),
                        'port': [int(parts[1])]
                    })
            elif isinstance(p, dict):
                pg_entries.append(p)

    # 每个YAML port_group条目保持为独立分组(append而非extend)
    # 数据结构: {layer: [[group1_entries], [group2_entries], ...]}
    # 每个子列表对应一个portGroup，对应一个独立的EID
    if layer not in _temp_pg_entries:
        _temp_pg_entries[layer] = []
    _temp_pg_entries[layer].append(pg_entries)

# 保存最后一组
if _temp_pg_entries:
    group_port_groups[_current_group_id] = {'entries': _temp_pg_entries, 'd2h_layers': _temp_d2h_layers}

# 向后兼容: parsed_port_groups 和 d2h_layers
parsed_port_groups = {}
d2h_layers = set()
if _default_group_id in group_port_groups:
    parsed_port_groups = group_port_groups[_default_group_id]['entries']
    d2h_layers = group_port_groups[_default_group_id]['d2h_layers']
elif group_port_groups:
    first_gid = min(group_port_groups.keys())
    parsed_port_groups = group_port_groups[first_gid]['entries']
    d2h_layers = group_port_groups[first_gid]['d2h_layers']

if group_port_groups:
    print(f"[信息] port_group定义: 共{len(group_port_groups)}个分组")
    for gid, data in group_port_groups.items():
        dev_list_info = group_to_devices.get(gid, '全部(默认)')
        print(f"       group_id={gid}, 适用device: {dev_list_info}")
        for layer, groups in data['entries'].items():
            for gi, group in enumerate(groups):
                port_strs = []
                for entry in group:
                    pg_die_id = entry.get('die_id', 0)
                    pg_ports = entry.get('port', [])
                    if isinstance(pg_ports, int):
                        pg_ports = [pg_ports]
                    port_strs.extend([f"{pg_die_id}/{p}" for p in pg_ports])
                print(f"         layer {layer}, portGroup[{gi}]: {port_strs}")

# ============================================================
# 2b. 解析port_list (topo.json各netlayer的port列表, 支持group_id扩展)
# ============================================================
# port_list与port_group的区别:
#   - port_list: 用于topo.json中PEER2NET edge的local_a_ports
#   - port_group: 用于rootinfo.json中的EID分配
#   一般拓扑二者等价, 但复杂拓扑(如a5 1dpod64)可能不同
port_list_raw = config.get('port_list', [])

# group_id → {layer: entries} 映射
group_port_lists = {}
_current_group_id = _default_group_id
_temp_pl_entries = {}
_temp_d2h_layers_pl = set()

for pg in port_list_raw:
    # 检查是否是group_id元数据条目
    if isinstance(pg, dict) and 'group_id' in pg and 'layer' not in pg:
        # 保存之前的entries到对应的group
        if _temp_pl_entries:
            group_port_lists[_current_group_id] = {'entries': _temp_pl_entries, 'd2h_layers': _temp_d2h_layers_pl}
        _current_group_id = pg['group_id']
        _temp_pl_entries = {}
        _temp_d2h_layers_pl = set()
        continue

    layer = pg.get('layer', 0)
    ports = pg.get('ports', pg.get('port', []))
    pg_entries = []

    if isinstance(ports, list) and ports and isinstance(ports[0], str):
        for port_str in ports:
            if port_str == 'd2h':
                _temp_d2h_layers_pl.add(layer)
                continue
            parts = port_str.split('/')
            if len(parts) == 2:
                pg_entries.append({
                    'die_id': int(parts[0]),
                    'port': [int(parts[1])]
                })
    elif isinstance(ports, dict):
        pg_entries.append(ports)
    elif isinstance(ports, list):
        for p in ports:
            if isinstance(p, str):
                if p == 'd2h':
                    _temp_d2h_layers_pl.add(layer)
                    continue
                parts = p.split('/')
                if len(parts) == 2:
                    pg_entries.append({
                        'die_id': int(parts[0]),
                        'port': [int(parts[1])]
                    })
            elif isinstance(p, dict):
                pg_entries.append(p)

    if layer not in _temp_pl_entries:
        _temp_pl_entries[layer] = []
    _temp_pl_entries[layer].extend(pg_entries)

# 保存最后一组
if _temp_pl_entries:
    group_port_lists[_current_group_id] = {'entries': _temp_pl_entries, 'd2h_layers': _temp_d2h_layers_pl}

# 向后兼容: parsed_port_lists 和 d2h_layers_pl
parsed_port_lists = {}
d2h_layers_pl = set()
if _default_group_id in group_port_lists:
    parsed_port_lists = group_port_lists[_default_group_id]['entries']
    d2h_layers_pl = group_port_lists[_default_group_id]['d2h_layers']
elif group_port_lists:
    first_gid = min(group_port_lists.keys())
    parsed_port_lists = group_port_lists[first_gid]['entries']
    d2h_layers_pl = group_port_lists[first_gid]['d2h_layers']

if group_port_lists:
    print(f"[信息] port_list定义: 共{len(group_port_lists)}个分组")
    for gid, data in group_port_lists.items():
        dev_list_info = group_to_devices.get(gid, '全部(默认)')
        print(f"       group_id={gid}, 适用device: {dev_list_info}")
        for layer, entries in data['entries'].items():
            port_strs = []
            for entry in entries:
                pg_die_id = entry.get('die_id', 0)
                pg_ports = entry.get('port', [])
                if isinstance(pg_ports, int):
                    pg_ports = [pg_ports]
                port_strs.extend([f"{pg_die_id}/{p}" for p in pg_ports])
            print(f"         layer {layer}: {port_strs}")

# ============================================================
# EID分配函数 (IP→EID转换)
# ============================================================
URMA_EID_LEN = 16

def ip_to_eid_string(ip, family=socket.AF_INET):
    if family == socket.AF_INET:
        ip_bytes = socket.inet_pton(socket.AF_INET, ip)
        eid = bytearray(URMA_EID_LEN)
        eid[12:16] = ip_bytes
    elif family == socket.AF_INET6:
        ip_bytes = socket.inet_pton(socket.AF_INET6, ip)
        eid = bytearray(URMA_EID_LEN)
        eid[0:16] = ip_bytes
    else:
        raise ValueError(f"不支持的地址族: {family}")
    hex_str = ''.join(f'{b:02x}' for b in eid)
    return f'0x{hex_str}'

def allocate_ip(aa, bb, counter):
    cc = (counter >> 8) & 0xFF
    dd = counter & 0xFF
    return f"{aa}.{bb}.{cc}.{dd}"

# ============================================================
# 端口工具函数
# ============================================================

def parse_port_set(port_data):
    result = []
    if isinstance(port_data, (list, tuple)):
        for item in port_data:
            if isinstance(item, (list, tuple, set)):
                result.append(sorted([int(x) for x in item]))
            else:
                result.append([int(item)])
    elif isinstance(port_data, (int, str)):
        result.append([int(port_data)])
    return result

def get_die_list_for_device(device_id):
    """获取device对应的端口分配表die_list (支持group_id扩展)"""
    gid = device_to_group.get(device_id, _default_group_id)
    return group_die_lists.get(gid, die_list)

def get_peer2peer_portgroups(die_id, device_id=None):
    dl = get_die_list_for_device(device_id) if device_id is not None else die_list
    for d in dl:
        if d['die_id'] == die_id:
            return parse_port_set(d['peer2peer'])
    return []

def get_p2n_portgroup_for_layer(layer, device_id=None):
    """获取PEER2NET端口的port定义: 优先使用port_list(topo.json), 否则使用port_group
    返回的是该layer下所有portGroup的扁平列表 (用于topo.json)"""
    gid = device_to_group.get(device_id, _default_group_id) if device_id is not None else _default_group_id
    # 优先使用port_list (port_list不区分分组)
    if gid in group_port_lists and layer in group_port_lists[gid]['entries']:
        return group_port_lists[gid]['entries'][layer]
    if layer in parsed_port_lists:
        return parsed_port_lists[layer]
    # 回退到port_group: 扁平化所有分组
    groups = None
    if gid in group_port_groups and layer in group_port_groups[gid]['entries']:
        groups = group_port_groups[gid]['entries'][layer]
    elif layer in parsed_port_groups:
        groups = parsed_port_groups[layer]
    if groups:
        flat = []
        for group in groups:
            flat.extend(group)
        return flat
    return None

def get_p2n_portgroup_groups(layer, device_id=None):
    """获取EID分配用的port_group分组列表 (仅用于rootinfo.json)
    返回: [[group1_entries], [group2_entries], ...]
    每个子列表是一个独立的portGroup，对应一个独立的EID"""
    gid = device_to_group.get(device_id, _default_group_id) if device_id is not None else _default_group_id
    if gid in group_port_groups and layer in group_port_groups[gid]['entries']:
        return group_port_groups[gid]['entries'][layer]
    if layer in parsed_port_groups:
        return parsed_port_groups[layer]
    return None

def port_group_to_strs(pg_entries):
    port_strs = []
    for entry in pg_entries:
        pg_die_id = entry.get('die_id', 0)
        pg_ports = entry.get('port', [])
        if isinstance(pg_ports, int):
            pg_ports = [pg_ports]
        for p in pg_ports:
            port_strs.append(f"{pg_die_id}/{p}")
    return port_strs

def get_peer2net_ports(die_id, device_id=None):
    dl = get_die_list_for_device(device_id) if device_id is not None else die_list
    for d in dl:
        if d['die_id'] == die_id:
            p2n = d['peer2net']
            if isinstance(p2n, list):
                flat = []
                for item in p2n:
                    if isinstance(item, list):
                        flat.extend([int(x) for x in item])
                    else:
                        flat.append(int(item))
                return sorted(flat)
            return []
    return []

def get_h2d_ports(die_id, device_id=None):
    dl = get_die_list_for_device(device_id) if device_id is not None else die_list
    for d in dl:
        if d['die_id'] == die_id:
            h2d = d['h2d']
            if isinstance(h2d, list):
                flat = []
                for item in h2d:
                    if isinstance(item, list):
                        flat.extend([int(x) for x in item])
                    else:
                        flat.append(int(item))
                return sorted(flat)
            return []
    return []

# ============================================================
# 3. 解析links，生成edge列表 (支持fullmesh/enum/matrix)
# ============================================================
links = config.get('links', [])

def expand_range_field(data, field_name, list_field_name=None):
    if list_field_name is None:
        list_field_name = field_name
    raw = data.get(field_name, None)
    range_raw = data.get(f'{field_name}_range', None)
    result = []
    if raw is not None:
        if isinstance(raw, list):
            result = [int(x) for x in raw]
        elif isinstance(raw, (int, float)):
            result = [int(raw)]
    if range_raw is not None:
        if isinstance(range_raw, list) and len(range_raw) == 2:
            result.extend(list(range(int(range_raw[0]), int(range_raw[1]) + 1)))
        elif isinstance(range_raw, list):
            result.extend([int(x) for x in range_raw])
    return result

def expand_fullmesh(devices):
    edges = []
    for i in range(len(devices)):
        for j in range(i + 1, len(devices)):
            edges.append((devices[i], devices[j]))
    return edges

all_edges = []
switch_link_devices = []
topo_instance_counter = 0

for link_entry in links:
    link_mode = link_entry.get('link_mode', 'enum')
    link_type = link_entry.get('link_type', '1DMESH')
    die_id = link_entry.get('die_id', None)
    protocol = link_entry.get('protocol', ['UB_CTP', 'UB_MEM'])
    if isinstance(protocol, str):
        protocol = [protocol]
    net_layer = link_entry.get('net_layer', 0)

    if link_mode == 'fullmesh':
        connections = link_entry.get('connections', [])
        device_groups = link_entry.get('device_groups', None)

        if connections:
            for conn in connections:
                conn_die_id = conn.get('die_id', die_id if die_id is not None else 0)
                devices = expand_range_field(conn, 'devices')
                conn_edges = expand_fullmesh(devices)
                for (a, b) in conn_edges:
                    all_edges.append({
                        'net_layer': net_layer,
                        'link_type': 'PEER2PEER',
                        'topo_type': link_type,
                        'topo_instance_id': topo_instance_counter,
                        'local_a': a,
                        'local_b': b,
                        'src_die_id': conn_die_id,
                        'dst_die_id': conn_die_id,
                        'protocols': protocol,
                        'position': 'DEVICE'
                    })
                topo_instance_counter += 1
        elif device_groups:
            for group in device_groups:
                group_edges = expand_fullmesh(group)
                for (a, b) in group_edges:
                    all_edges.append({
                        'net_layer': net_layer,
                        'link_type': 'PEER2PEER',
                        'topo_type': link_type,
                        'topo_instance_id': topo_instance_counter,
                        'local_a': a,
                        'local_b': b,
                        'src_die_id': die_id if die_id is not None else 0,
                        'dst_die_id': die_id if die_id is not None else 0,
                        'protocols': protocol,
                        'position': 'DEVICE'
                    })
                topo_instance_counter += 1

    elif link_mode == 'enum':
        d2d_links = link_entry.get('device_to_device_links', [])
        for conn in d2d_links:
            s_die = conn.get('src_die_id', die_id if die_id is not None else 0)
            d_die = conn.get('dst_die_id', die_id if die_id is not None else 0)
            s_locals = expand_range_field(conn, 'src_local_id')
            d_locals = expand_range_field(conn, 'dst_local_id')

            for s_local in s_locals:
                for d_local in d_locals:
                    all_edges.append({
                        'net_layer': net_layer,
                        'link_type': 'PEER2PEER',
                        'topo_type': link_type,
                        'topo_instance_id': topo_instance_counter,
                        'local_a': s_local,
                        'local_b': d_local,
                        'src_die_id': s_die,
                        'dst_die_id': d_die,
                        'protocols': protocol,
                        'position': 'DEVICE'
                    })
                    topo_instance_counter += 1

        d2s_links = link_entry.get('device_to_switch_links', [])
        for sw_link in d2s_links:
            sw_die_id = sw_link.get('die_id', die_id if die_id is not None else 0)
            sw_devices = expand_range_field(sw_link, 'devices')
            for dev in sw_devices:
                switch_link_devices.append({
                    'device_id': dev,
                    'die_id': sw_die_id
                })

    elif link_mode == 'matrix':
        matrix_data = link_entry.get('device_to_device_links', [])
        if matrix_data:
            matrix_die_id = 0
            matrix_rows = []

            for item in matrix_data:
                if isinstance(item, dict):
                    matrix_die_id = item.get('die_id', 0)
                elif isinstance(item, list):
                    matrix_rows.append(item)

            for i, row in enumerate(matrix_rows):
                for j in range(i + 1, len(row)):
                    if j < len(row) and row[j] == 1:
                        all_edges.append({
                            'net_layer': net_layer,
                            'link_type': 'PEER2PEER',
                            'topo_type': link_type,
                            'topo_instance_id': topo_instance_counter,
                            'local_a': i,
                            'local_b': j,
                            'src_die_id': matrix_die_id,
                            'dst_die_id': matrix_die_id,
                            'protocols': protocol,
                            'position': 'DEVICE'
                        })
            topo_instance_counter += 1

        d2s_links = link_entry.get('device_to_switch_links', [])
        for sw_link in d2s_links:
            sw_die_id = sw_link.get('die_id', die_id if die_id is not None else 0)
            sw_devices = expand_range_field(sw_link, 'devices')
            for dev in sw_devices:
                switch_link_devices.append({
                    'device_id': dev,
                    'die_id': sw_die_id
                })
    else:
        print(f"  [警告] 不支持的link_mode: {link_mode}，跳过")

print(f"[信息] 解析连接关系: 生成 {len(all_edges)} 条PEER2PEER edge")

# ============================================================
# 4. 为每条PEER2PEER edge分配端口
# ============================================================
device_port_usage = {}
for dev_id in range(device_num):
    device_port_usage[dev_id] = {}

def get_next_p2p_port(device_id, die_id):
    key = (device_id, die_id)
    if key not in device_port_usage[device_id]:
        device_port_usage[device_id][key] = 0
    portgroups = get_peer2peer_portgroups(die_id, device_id)
    if not portgroups:
        return None
    idx = device_port_usage[device_id][key]
    pg_idx = idx % len(portgroups)
    port_in_pg = idx // len(portgroups)
    pg = portgroups[pg_idx]
    if port_in_pg < len(pg):
        port = pg[port_in_pg]
    else:
        port = pg[-1]
    device_port_usage[device_id][key] = idx + 1
    return f"{die_id}/{port}"

device_p2p_needed = [0] * device_num
for edge in all_edges:
    device_p2p_needed[edge['local_a']] += 1
    device_p2p_needed[edge['local_b']] += 1

for dev_id in range(device_num):
    dev_dl = get_die_list_for_device(dev_id)
    total_p2p = sum(len(d['peer2peer']) for d in dev_dl)
    if device_p2p_needed[dev_id] > total_p2p:
        print(f"[警告] device {dev_id} 需要{device_p2p_needed[dev_id]}个P2P端口，但只有{total_p2p}个可用")

for edge in all_edges:
    a = edge['local_a']
    b = edge['local_b']
    src_die = edge.pop('src_die_id', None)
    dst_die = edge.pop('dst_die_id', None)
    dev_dl_a = get_die_list_for_device(a)
    dev_dl_b = get_die_list_for_device(b)

    if src_die is None:
        src_die = next((d['die_id'] for d in dev_dl_a if d['peer2peer']), dev_dl_a[0]['die_id'] if dev_dl_a else 0)
    if dst_die is None:
        dst_die = next((d['die_id'] for d in dev_dl_b if d['peer2peer']), dev_dl_b[0]['die_id'] if dev_dl_b else 0)

    port_a = get_next_p2p_port(a, src_die)
    port_b = get_next_p2p_port(b, dst_die)

    edge['local_a_ports'] = [port_a] if port_a else []
    edge['local_b_ports'] = [port_b] if port_b else []

print(f"[信息] 端口分配完成")

# ============================================================
# 5. 生成topo.json
# ============================================================
peer_list = [{'local_id': i} for i in range(device_num)]

topo_edges = []

include_tiid = not bool(d2h_layers)

for edge in all_edges:
    edge_dict = {
        'link_type': edge['link_type'],
        'local_a': edge['local_a'],
        'local_a_ports': edge['local_a_ports'],
        'local_b': edge['local_b'],
        'local_b_ports': edge['local_b_ports'],
        'net_layer': edge['net_layer'],
        'position': edge['position'],
        'protocols': edge['protocols'],
        'topo_type': edge['topo_type']
    }
    if include_tiid:
        edge_dict['topo_instance_id'] = edge['topo_instance_id']
    topo_edges.append(edge_dict)

p2n_topo_instance_id = topo_instance_counter

# 收集所有已知端口 (用于ungrouped端口检测)
# port_group新结构: {layer: [[group1_entries], [group2_entries], ...]}
all_pg_port_strs = set()
for gid, data in group_port_groups.items():
    for layer, groups in data['entries'].items():
        if layer not in data['d2h_layers']:
            for group in groups:
                for entry in group:
                    pg_die_id = entry.get('die_id', 0)
                    pg_ports = entry.get('port', [])
                    if isinstance(pg_ports, int):
                        pg_ports = [pg_ports]
                    for p in pg_ports:
                        all_pg_port_strs.add(f"{pg_die_id}/{p}")

# 将port_list中定义的端口也加入"已知端口"集合
for gid, data in group_port_lists.items():
    for layer, entries in data['entries'].items():
        if layer not in data['d2h_layers']:
            for entry in entries:
                pg_die_id = entry.get('die_id', 0)
                pg_ports = entry.get('port', [])
                if isinstance(pg_ports, int):
                    pg_ports = [pg_ports]
                for p in pg_ports:
                    all_pg_port_strs.add(f"{pg_die_id}/{p}")

ungrouped_p2n_ports = []
for d in die_list:
    for p in get_peer2net_ports(d['die_id']):
        ps = f"{d['die_id']}/{p}"
        if ps not in all_pg_port_strs:
            ungrouped_p2n_ports.append(ps)
ungrouped_p2n_ports.sort()

# p2n_layers: 合并所有group的port_group和port_list中的layer (排除d2h)
_all_p2n_layers = set()
_all_d2h_layers = set()
for gid, data in group_port_groups.items():
    _all_p2n_layers.update(data['entries'].keys())
    _all_d2h_layers.update(data['d2h_layers'])
for gid, data in group_port_lists.items():
    _all_p2n_layers.update(data['entries'].keys())
    _all_d2h_layers.update(data['d2h_layers'])
p2n_layers = sorted([l for l in _all_p2n_layers if l not in _all_d2h_layers])
first_p2n_layer = p2n_layers[0] if p2n_layers else None

for dev_id in range(device_num):
    dev_dl = get_die_list_for_device(dev_id)
    dev_gid = device_to_group.get(dev_id, _default_group_id)
    dev_pl = group_port_lists.get(dev_gid, {}).get('entries', parsed_port_lists)
    for layer in p2n_layers:
        # topo.json的PEER2NET端口: 优先使用port_list, 否则使用port_group
        pg_def = get_p2n_portgroup_for_layer(layer, dev_id)
        if pg_def:
            port_strs = port_group_to_strs(pg_def)
        else:
            port_strs = []
            for d in dev_dl:
                port_strs.extend([f"{d['die_id']}/{p}" for p in get_peer2net_ports(d['die_id'], dev_id)])

        # 仅当该layer没有port_list定义时，才追加ungrouped端口
        if layer == first_p2n_layer and ungrouped_p2n_ports and layer not in dev_pl:
            combined = list(port_strs) + ungrouped_p2n_ports
            combined.sort(key=lambda x: (int(x.split('/')[0]), int(x.split('/')[1])) if '/' in x else (999, 0))
            port_strs = combined

        if port_strs:
            if len(port_strs) == 1:
                protocols = ['UB_CTP']
            else:
                protocols = ['UB_CTP', 'UB_MEM']
            p2n_edge = {
                'link_type': 'PEER2NET',
                'local_a': dev_id,
                'local_a_ports': list(port_strs),
                'net_layer': layer,
                'position': 'DEVICE',
                'protocols': protocols,
                'topo_type': 'CLOS'
            }
            if include_tiid:
                p2n_edge['topo_instance_id'] = p2n_topo_instance_id
            topo_edges.append(p2n_edge)

    # per-device h2d检查 (支持group_id扩展)
    dev_has_h2d = any(len(d['h2d']) > 0 for d in dev_dl)
    if dev_has_h2d:
        dev_d2h_layers = group_port_groups.get(dev_gid, {}).get('d2h_layers', d2h_layers)
        if dev_d2h_layers:
            d2h_net_layer = min(dev_d2h_layers)
        else:
            d2h_net_layer = max(p2n_layers) + 1 if p2n_layers else 1
        d2h_edge = {
            'link_type': 'PEER2NET',
            'local_a': dev_id,
            'local_a_ports': ['d2h'],
            'net_layer': d2h_net_layer,
            'position': 'HOST',
            'protocols': ['ROCE'] if dev_d2h_layers else ['UB_CTP', 'UB_MEM'],
            'topo_type': 'CLOS'
        }
        if include_tiid:
            d2h_edge['topo_instance_id'] = p2n_topo_instance_id
        topo_edges.append(d2h_edge)

hardware_type = config.get('hardware_type', None)
if not hardware_type:
    desc = config.get('description', '')
    if 'UBX' in desc.upper():
        hardware_type = 'UBX'
    elif 'UB' in desc.upper():
        hardware_type = 'UB'
    else:
        name_str = name.upper()
        if 'UBX' in name_str:
            hardware_type = 'UBX'
        elif 'UB' in name_str:
            hardware_type = 'UB'
        else:
            hardware_type = soc_version.upper() if soc_version != 'unknown' else 'UNKNOWN'

topo_json = {
    'edge_count': len(topo_edges),
    'edge_list': topo_edges,
    'hardware_type': hardware_type,
    'peer_count': device_num,
    'peer_list': peer_list,
    'version': '2.0'
}

topo_path = os.path.join(OUTPUT_DIR, topo_filename)
with open(topo_path, 'w') as f:
    json.dump(topo_json, f, indent=4, ensure_ascii=False)
print(f"\n[成功] 生成topo.json: {topo_path}")
print(f"       peer_list: {device_num}个device, edge_list: {len(topo_edges)}条edge")

# ============================================================
# 6. 生成rootinfo.json
# ============================================================
IP_AA = 192
IP_BB = 1

DEVICE_PORT_START = 1

host_ip = f"{IP_AA}.{IP_BB}.0.0"
host_eid = ip_to_eid_string(host_ip)

port_counter = DEVICE_PORT_START
total_eid_count = 0

# rootinfo EID分配: p2n_layers使用所有group的port_group中定义的layer
_all_pg_layers = set()
_all_pg_d2h_layers = set()
for gid, data in group_port_groups.items():
    _all_pg_layers.update(data['entries'].keys())
    _all_pg_d2h_layers.update(data['d2h_layers'])
p2n_layers_eid = sorted([l for l in _all_pg_layers if l not in _all_pg_d2h_layers])

rank_list = []
for dev_id in range(device_num):
    level_list = []
    dev_dl = get_die_list_for_device(dev_id)
    dev_gid = device_to_group.get(dev_id, _default_group_id)

    # P2P端口: 按device对应的端口分配表生成 (支持group_id扩展)
    dev_has_p2p = any(len(d['peer2peer']) > 0 for d in dev_dl)
    if dev_has_p2p:
        p2p_port_strs = []
        for d in dev_dl:
            for p in sorted(d['peer2peer']):
                p2p_port_strs.append(f"{d['die_id']}/{p}")
        p2p_rank_addr_list = []
        for port_str in p2p_port_strs:
            ip_addr = allocate_ip(IP_AA, IP_BB, port_counter)
            eid_str = ip_to_eid_string(ip_addr)
            port_counter += 1
            p2p_rank_addr_list.append({
                'addr_type': 'EID',
                'addr': eid_str,
                'ports': [port_str],
                'plane_id': 'plane_0'
            })
            total_eid_count += 1
        level_list.append({
            'net_layer': 0,
            'net_instance_id': 'sp_0_srv_0',
            'net_type': 'TOPO_FILE_DESC',
            'net_attr': '',
            'rank_addr_list': p2p_rank_addr_list
        })

    # PEER2NET: EID分配使用port_group (非port_list)
    # 每个portGroup条目独立创建一个EID
    for layer in p2n_layers_eid:
        pg_groups = get_p2n_portgroup_groups(layer, dev_id)
        rank_addr_list = []
        if pg_groups:
            for group in pg_groups:
                port_strs = []
                for pg_entry in group:
                    pg_die_id = pg_entry.get('die_id', 0)
                    pg_ports = pg_entry.get('port', [])
                    if isinstance(pg_ports, int):
                        pg_ports = [pg_ports]
                    for p in pg_ports:
                        port_strs.append(f"{pg_die_id}/{p}")

                ip_addr = allocate_ip(IP_AA, IP_BB, port_counter)
                eid_str = ip_to_eid_string(ip_addr)
                port_counter += 1

                rank_addr_list.append({
                    'addr_type': 'EID',
                    'addr': eid_str,
                    'ports': port_strs,
                    'plane_id': 'plane_0'
                })
                total_eid_count += 1
        else:
            port_strs = []
            for d in dev_dl:
                port_strs.extend([f"{d['die_id']}/{p}" for p in get_peer2net_ports(d['die_id'], dev_id)])

            ip_addr = allocate_ip(IP_AA, IP_BB, port_counter)
            eid_str = ip_to_eid_string(ip_addr)
            port_counter += 1

            rank_addr_list.append({
                'addr_type': 'EID',
                'addr': eid_str,
                'ports': port_strs,
                'plane_id': 'plane_0'
            })
            total_eid_count += 1

        if rank_addr_list:
            level_list.append({
                'net_layer': layer,
                'net_instance_id': 'sp_0_srv_0',
                'net_type': 'TOPO_FILE_DESC',
                'net_attr': '',
                'rank_addr_list': rank_addr_list
            })

    # per-device h2d检查 (支持group_id扩展)
    dev_has_h2d = any(len(d['h2d']) > 0 for d in dev_dl)
    if dev_has_h2d:
        dev_d2h_layers = group_port_groups.get(dev_gid, {}).get('d2h_layers', d2h_layers)
        if dev_d2h_layers:
            d2h_net_layer = min(dev_d2h_layers)
        else:
            d2h_net_layer = max(p2n_layers_eid) + 1 if p2n_layers_eid else 1
        ip_addr = allocate_ip(IP_AA, IP_BB, port_counter)
        eid_str = ip_to_eid_string(ip_addr)
        port_counter += 1
        level_list.append({
            'net_layer': d2h_net_layer,
            'net_instance_id': 'sp_0_srv_0',
            'net_type': 'TOPO_FILE_DESC',
            'net_attr': '',
            'rank_addr_list': [{
                'addr_type': 'EID',
                'addr': eid_str,
                'ports': ['d2h'],
                'plane_id': 'plane_0'
            }]
        })
        total_eid_count += 1

    rank_list.append({
        'device_id': dev_id,
        'local_id': dev_id,
        'level_list': level_list
    })

rootinfo_json = {
    'rank_count': device_num,
    'rank_list': rank_list,
    'topo_file_path': f'{topo_path}',
    'version': '2.0'
}

rootinfo_path = os.path.join(OUTPUT_DIR, rootinfo_filename)
with open(rootinfo_path, 'w') as f:
    json.dump(rootinfo_json, f, indent=4, ensure_ascii=False)
print(f"[成功] 生成rootinfo.json: {rootinfo_path}")
print(f"       rank_count: {device_num}, EID总数: {total_eid_count}")
print(f"       Host IP: {host_ip} → EID: {host_eid}")
last_cc = ((port_counter - 1) >> 8) & 0xFF
last_dd = (port_counter - 1) & 0xFF
print(f"       Device IP: {IP_AA}.{IP_BB}.0.1 ~ {IP_AA}.{IP_BB}.{last_cc}.{last_dd}")

PYTHON_SCRIPT

echo ""
echo "=========================================="
echo "  生成完成"
echo "=========================================="
echo ""
success_msg "topo.json 和 rootinfo.json 生成完成！"
