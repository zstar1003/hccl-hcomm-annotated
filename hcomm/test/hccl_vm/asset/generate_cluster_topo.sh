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

# 根据集群拓扑配置文件，生成集群组网目录结构
# 功能：
#   a. 解析集群YAML文件，在输出目录下创建所有超节点组网目录
#   b. 支持两种集群配置格式：
#      - 新格式(扁平)：集群YAML直接包含server_list，每个server通过super_pod_id归属超节点
#      - 旧格式(三层)：集群YAML通过super_node_list引用超节点YAML，超节点YAML中定义server_list
#   c. 支持两种方式获取server拓扑文件：
#      - 如果server_topo是yaml文件，调用generate_topo_rootinfo.sh生成topo.json和rootinfo.json
#      - 如果server_topo是目录名，从topo_bank/server_topo_bank目录拷贝
#   d. 支持id_range扩展：集群配置中super_node_list的id_range，server_list的id_range
#   e. 调用allocate_eid.sh脚本，统一分配server和device ports的IP地址/EID
#   f. 生成servers_info.json文件，记录所有server的host IP和soc_version信息

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

CLUSTER_CONFIG_DIR="${BASE_DIR}/config/cluster"
SUPERPOD_CONFIG_DIR="${BASE_DIR}/config/super_pod"
SERVER_CONFIG_DIR="${BASE_DIR}/config/server_or_pod"
SERVER_TOPO_BANK_DIR="${BASE_DIR}/topo_bank/server_topo_bank"
GENERATE_TOPO_SCRIPT="${SCRIPT_DIR}/generate_server_topo.sh"
ALLOCATE_EID_SCRIPT="${SCRIPT_DIR}/allocate_eid.sh"
DEFAULT_OUTPUT_DIR="${BASE_DIR}/config/network/cluster"

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
用法: $(basename "$0") [OPTIONS] <集群配置文件>

参数:
    集群配置文件       配置文件路径或文件名
                       绝对路径: 直接使用
                       文件名/相对路径: 在 ${CLUSTER_CONFIG_DIR} 下查找

选项:
    -o, --output DIR   指定输出目录 (默认: ${DEFAULT_OUTPUT_DIR})
    -h, --help         显示帮助信息

示例:
    $(basename "$0") ascend950_cluster_4_server_normal.yaml
    $(basename "$0") /absolute/path/to/cluster_config.yaml
    $(basename "$0") -o /tmp/my_output ascend950_cluster_4_server_normal.yaml

说明:
    1. 解析集群YAML文件，自动检测配置格式（新格式/旧格式）
    2. 新格式：server_list直接在集群YAML中，通过super_pod_id归属超节点
    3. 旧格式：通过super_node_list引用超节点YAML，超节点YAML中定义server_list
    4. 在输出目录下为每个超节点创建目录
    5. 在超节点目录下为每个server创建目录
    6. 根据server_topo配置生成或拷贝topo.json和rootinfo.json
    7. 调用allocate_eid.sh统一分配IP/EID地址
    8. 生成servers_info.json文件

EOF
}

CLUSTER_YAML_INPUT=""
OUTPUT_DIR=""

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
        *)
            CLUSTER_YAML_INPUT="$1"
            shift
            ;;
    esac
done

if [[ -z "$CLUSTER_YAML_INPUT" ]]; then
    error_exit "请指定集群配置文件\n$(show_usage)"
fi

CLUSTER_YAML_PATH="$(resolve_path "$CLUSTER_YAML_INPUT" "$CLUSTER_CONFIG_DIR")"

if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="${DEFAULT_OUTPUT_DIR}"
fi

echo "=========================================="
echo "  昇腾集群组网拓扑生成工具"
echo "=========================================="
echo ""
echo "集群配置文件: $CLUSTER_YAML_PATH"
echo "输出根目录: $OUTPUT_DIR"
echo ""

CLUSTER_BASENAME=$(basename "$CLUSTER_YAML_PATH" .yaml)

OUTPUT_DIR="${OUTPUT_DIR}/${CLUSTER_BASENAME}"

echo "集群输出目录: $OUTPUT_DIR"
echo ""

rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

export CLUSTER_YAML_PATH
export OUTPUT_DIR
export CLUSTER_CONFIG_DIR
export SUPERPOD_CONFIG_DIR
export SERVER_CONFIG_DIR
export SERVER_TOPO_BANK_DIR
export GENERATE_TOPO_SCRIPT
export ALLOCATE_EID_SCRIPT

python3 << 'PYTHON_SCRIPT'
import json
import os
import re
import shutil
import subprocess
import sys
import socket

try:
    import yaml
except ImportError:
    print("[错误] 需要PyYAML库，请安装: pip install pyyaml")
    sys.exit(1)

CLUSTER_YAML_PATH = os.environ.get('CLUSTER_YAML_PATH', '')
OUTPUT_DIR = os.environ.get('OUTPUT_DIR', '')
SUPERPOD_CONFIG_DIR = os.environ.get('SUPERPOD_CONFIG_DIR', '')
SERVER_CONFIG_DIR = os.environ.get('SERVER_CONFIG_DIR', '')
SERVER_TOPO_BANK_DIR = os.environ.get('SERVER_TOPO_BANK_DIR', '')
GENERATE_TOPO_SCRIPT = os.environ.get('GENERATE_TOPO_SCRIPT', '')
ALLOCATE_EID_SCRIPT = os.environ.get('ALLOCATE_EID_SCRIPT', '')

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

def expand_id_range(entry):
    ids = []
    if 'id' in entry:
        raw = entry['id']
        if isinstance(raw, list):
            ids.extend([int(x) for x in raw])
        else:
            ids.append(int(raw))
    if 'id_range' in entry:
        rng = entry['id_range']
        if isinstance(rng, list) and len(rng) == 2:
            ids.extend(list(range(int(rng[0]), int(rng[1]) + 1)))
        elif isinstance(rng, list):
            ids.extend([int(x) for x in rng])
    return ids

def find_superpod_yaml(topo_name):
    if not topo_name:
        return None
    if os.path.isfile(topo_name):
        return topo_name
    candidates = [
        os.path.join(SUPERPOD_CONFIG_DIR, topo_name),
        os.path.join(SUPERPOD_CONFIG_DIR, topo_name if topo_name.endswith('.yaml') else topo_name + '.yaml'),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None

def find_server_topo(topo_name):
    if not topo_name:
        return None, None
    yaml_candidates = [
        os.path.join(SERVER_CONFIG_DIR, topo_name),
        os.path.join(SERVER_CONFIG_DIR, topo_name if topo_name.endswith('.yaml') else topo_name + '.yaml'),
    ]
    for c in yaml_candidates:
        if os.path.isfile(c):
            return 'yaml', c
    dir_candidates = [
        os.path.join(SERVER_TOPO_BANK_DIR, topo_name),
    ]
    for c in dir_candidates:
        if os.path.isdir(c):
            return 'bank', c
    return None, None

def generate_server_topo(yaml_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    cmd = ['bash', GENERATE_TOPO_SCRIPT, yaml_path, '-o', output_dir, '-b']
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [错误] 生成server拓扑失败: {result.stderr}")
        return False
    return True

def copy_server_topo_from_bank(bank_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    copied_files = []
    for f in os.listdir(bank_dir):
        src = os.path.join(bank_dir, f)
        if os.path.isfile(src):
            shutil.copy2(src, output_dir)
            copied_files.append(f)
    return copied_files

# ============================================================
# Step A: 解析集群YAML文件，自动检测格式
# ============================================================
print("[信息] Step A: 解析集群配置文件...")

with open(CLUSTER_YAML_PATH, 'r') as f:
    cluster_config = yaml.safe_load(f)

cluster_name = cluster_config.get('name', 'unknown')
super_node_num = cluster_config.get('super_node_num', 0)
server_list_raw = cluster_config.get('server_list', [])
super_node_list_raw = cluster_config.get('super_node_list', [])

is_new_format = bool(server_list_raw and not super_node_list_raw)

print(f"  集群名称: {cluster_name}")
print(f"  超节点数量: {super_node_num}")
print(f"  配置格式: {'新格式(扁平)' if is_new_format else '旧格式(三层)'}")

if super_node_num == 0 and not is_new_format:
    print("[错误] 集群配置文件中未定义超节点 (super_node_num=0)")
    sys.exit(1)

# ============================================================
# Step B: 构建超节点→server映射，创建目录结构
# ============================================================
# sp_servers: {sp_id: [{'id': srv_id, 'server_topo': ..., 'soc_version': ...}, ...]}
sp_servers = {}

if is_new_format:
    # 新格式：server_list直接在集群YAML中，通过super_pod_id归属超节点
    server_num = cluster_config.get('server_num', 0)
    print(f"  Server数量: {server_num}")

    for entry in server_list_raw:
        sp_id = entry.get('super_pod_id', 0)
        ids = expand_id_range(entry)
        srv_topo = entry.get('server_topo', '')
        soc_version = entry.get('soc_version', '')
        for sid in ids:
            if sp_id not in sp_servers:
                sp_servers[sp_id] = []
            sp_servers[sp_id].append({
                'id': sid,
                'server_topo': srv_topo,
                'soc_version': soc_version
            })

    actual_server_count = sum(len(v) for v in sp_servers.values())
    actual_super_node_count = len(sp_servers)
    print(f"  展开后: {actual_super_node_count} 个超节点, {actual_server_count} 个server")

    if server_num > 0 and actual_server_count != server_num:
        print(f"[错误] server_num({server_num})与server_list展开后的实际server数量({actual_server_count})不一致")
        sys.exit(1)

    if super_node_num > 0 and actual_super_node_count != super_node_num:
        print(f"[错误] super_node_num({super_node_num})与server_list中实际超节点数量({actual_super_node_count})不一致")
        sys.exit(1)
else:
    # 旧格式：通过super_node_list引用超节点YAML
    expanded_super_nodes = []
    for entry in super_node_list_raw:
        ids = expand_id_range(entry)
        topo = entry.get('superpod_topo', '')
        for sid in ids:
            expanded_super_nodes.append({'id': sid, 'superpod_topo': topo})

    print(f"  展开后超节点数: {len(expanded_super_nodes)}")

    if super_node_num > 0 and len(expanded_super_nodes) != super_node_num:
        print(f"[错误] super_node_num({super_node_num})与super_node_list展开后的实际超节点数量({len(expanded_super_nodes)})不一致")
        sys.exit(1)

    for sp_entry in expanded_super_nodes:
        sp_id = sp_entry['id']
        sp_topo_name = sp_entry['superpod_topo']

        sp_yaml_path = find_superpod_yaml(sp_topo_name)
        if not sp_yaml_path:
            print(f"  [错误] 超节点拓扑配置文件不存在: {sp_topo_name}")
            continue

        with open(sp_yaml_path, 'r') as f:
            sp_config = yaml.safe_load(f)

        sp_name = sp_config.get('name', 'unknown')
        sp_form_type = sp_config.get('form_type', 'server')
        server_num = sp_config.get('server_num', 0)
        sp_server_list_raw = sp_config.get('server_list', [])

        print(f"  [信息] 超节点{sp_id}: {sp_name}, 形态: {sp_form_type}, Server数量: {server_num}")

        expanded_servers = []
        for srv_entry in sp_server_list_raw:
            ids = expand_id_range(srv_entry)
            srv_topo = srv_entry.get('server_topo', '')
            soc_version = srv_entry.get('soc_version', '')
            for sid in ids:
                expanded_servers.append({
                    'id': sid,
                    'server_topo': srv_topo,
                    'soc_version': soc_version
                })

        sp_servers[sp_id] = expanded_servers
        print(f"  [信息] 展开后Server数: {len(expanded_servers)}")

        if server_num > 0 and len(expanded_servers) != server_num:
            print(f"[错误] 超节点{sp_id}的server_num({server_num})与server_list展开后的实际server数量({len(expanded_servers)})不一致")
            sys.exit(1)

print("")

# ============================================================
# Step C: 创建目录结构，生成/拷贝server拓扑文件
# ============================================================
for sp_id in sorted(sp_servers.keys()):
    sp_dir = os.path.join(OUTPUT_DIR, f'superpod{sp_id}')
    os.makedirs(sp_dir, exist_ok=True)
    print(f"[成功] 创建超节点目录: superpod{sp_id}")

    for srv in sp_servers[sp_id]:
        srv_id = srv['id']
        srv_topo_name = srv['server_topo']
        srv_soc_version = srv['soc_version']

        srv_dir = os.path.join(sp_dir, f'server{srv_id}')

        topo_type, topo_path = find_server_topo(srv_topo_name)

        if topo_type == 'yaml':
            if generate_server_topo(topo_path, srv_dir):
                print(f"  [成功] server{srv_id}: 生成拓扑 ({srv_topo_name})")
            else:
                print(f"  [错误] server{srv_id}: 生成拓扑失败 ({srv_topo_name})")
        elif topo_type == 'bank':
            copied = copy_server_topo_from_bank(topo_path, srv_dir)
            print(f"  [成功] server{srv_id}: 拷贝 {len(copied)} 个文件 ({srv_topo_name})")
        else:
            print(f"  [错误] server{srv_id}: 找不到拓扑配置 ({srv_topo_name})")

    print("")

# ============================================================
# Step D: 调用allocate_eid.sh分配EID/IP地址
# ============================================================
print("[信息] Step D: 调用allocate_eid.sh分配EID/IP地址...")

if os.path.isfile(ALLOCATE_EID_SCRIPT):
    result = subprocess.run(['bash', ALLOCATE_EID_SCRIPT, '-d', OUTPUT_DIR], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [警告] EID分配失败: {result.stderr}")
    else:
        print(result.stdout)
else:
    print(f"  [警告] allocate_eid.sh脚本不存在: {ALLOCATE_EID_SCRIPT}")

# ============================================================
# Step E: 生成servers_info.json
# ============================================================
print("[信息] Step E: 生成servers_info.json...")

def extract_soc_version(filename):
    match = re.match(r'(ascend\d+)', filename)
    if match:
        return match.group(1)
    return 'unknown'

server_ip_list = []

for sp_id in sorted(sp_servers.keys()):
    sp_idx = sp_id
    for srv in sp_servers[sp_id]:
        srv_idx = srv['id']
        srv_soc_version = srv['soc_version']

        if not srv_soc_version:
            srv_dir = os.path.join(OUTPUT_DIR, f'superpod{sp_id}', f'server{srv_idx}')
            soc_version = 'unknown'
            if os.path.isdir(srv_dir):
                for f in os.listdir(srv_dir):
                    if 'rootinfo' in f and f.endswith('.json'):
                        soc_version = extract_soc_version(f)
                        break
            if soc_version == 'unknown':
                topo_json_path = os.path.join(srv_dir, 'topo.json')
                if os.path.isfile(topo_json_path):
                    try:
                        with open(topo_json_path, 'r') as tf:
                            topo_data = json.load(tf)
                        soc_version = topo_data.get('soc_version', 'unknown')
                    except Exception:
                        pass
            srv_soc_version = soc_version

        aa = 192 + sp_idx
        bb = srv_idx + 1
        host_ip = f"{aa}.{bb}.0.0"
        host_eid = ip_to_eid_string(host_ip)

        server_ip_list.append({
            "soc_version": srv_soc_version,
            "addr": host_eid
        })

servers_info = {
    "version": "2.0",
    "server_count": len(server_ip_list),
    "server_ip_list": server_ip_list
}

output_path = os.path.join(OUTPUT_DIR, 'servers_info.json')
with open(output_path, 'w') as f:
    json.dump(servers_info, f, indent=4, ensure_ascii=False)

print(f"  [成功] 生成servers_info.json: {output_path}")
print(f"         服务器数量: {len(server_ip_list)}")
for i, entry in enumerate(server_ip_list):
    print(f"         [{i}] {entry['soc_version']}: {entry['addr']}")

PYTHON_SCRIPT

echo ""
echo "=========================================="
echo "  生成完成"
echo "=========================================="
echo ""
echo "输出目录: $OUTPUT_DIR"
echo ""

TOTAL_SUPERPODS=$(find "$OUTPUT_DIR" -maxdepth 1 -type d -name "superpod*" | wc -l | tr -d ' ')
TOTAL_SERVERS=$(find "$OUTPUT_DIR" -mindepth 2 -maxdepth 2 -type d -name "server*" | wc -l | tr -d ' ')
TOTAL_FILES=$(find "$OUTPUT_DIR" -type f | wc -l | tr -d ' ')

echo "超节点目录数: $TOTAL_SUPERPODS"
echo "Server目录数: $TOTAL_SERVERS"
echo "文件总数: $TOTAL_FILES"

if [[ -f "${OUTPUT_DIR}/servers_info.json" ]]; then
    echo "servers_info.json: 已生成"
fi

echo ""

echo "目录结构:"
if command -v tree &>/dev/null; then
    tree -L 3 "$OUTPUT_DIR" 2>/dev/null || find "$OUTPUT_DIR" -type f | head -30
else
    find "$OUTPUT_DIR" -type f | head -30
fi

echo ""
success_msg "集群拓扑目录生成完成！"
