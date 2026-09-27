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

# 统一分配EID/IP地址
# 功能：
#   1. 遍历output_temp目录下所有rootinfo相关json文件
#   2. 根据IP地址分配规则，为每个server的portGroup分配IP地址
#   3. 将IP地址转换为EID字符串，修改rootinfo文件中的addr字段
#
# IP地址分配规则 (AA.BB.CC.DD):
#   AA = 192 + superpod_index  (超节点段，从192开始)
#   BB = server_index + 1      (服务器段，从1开始，排除192.0.0.0~192.0.0.24段)
#   CC:DD = device port计数器  (设备段，从0.1开始)
#   Host IP: AA.BB.0.0         (每个server固定一个host IP，仅供host使用)
#   Device port IPs: AA.BB.0.1, AA.BB.0.2, ... (依次递增，d2h端口也从此段分配)
#
# IP→EID转换规则 (等价于C++ IpToEidString):
#   IPv4地址(4字节)放入16字节EID的末尾(网络字节序)，前12字节为0
#   输出格式: "0x" + 32位十六进制字符串
#   示例: 192.0.0.1 → 0x000000000000000000000000c0000001

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
OUTPUT_DIR="${BASE_DIR}/config/network/cluster"

show_usage() {
    cat << EOF
用法: $(basename "$0") [OPTIONS]

选项:
    -d, --dir DIR     指定集群输出目录 (默认: ${OUTPUT_DIR})
    -h, --help        显示帮助信息

说明:
    遍历output_temp目录下所有rootinfo JSON文件，按照IP地址分配规则
    统一分配IP地址，转换为EID字符串后写入addr字段。

    IP地址分配规则 (AA.BB.CC.DD):
      AA = 192 + superpod_index  (超节点段)
      BB = server_index + 1      (服务器段，从1开始，排除192.0.0.0~192.0.0.24段)
      CC:DD = device port计数器  (设备段，从0.1开始)
      Host IP: AA.BB.0.0         (每个server固定一个host IP，仅供host使用)
      Device port IPs: AA.BB.0.1, AA.BB.0.2, ... (依次递增，d2h端口也从此段分配)

    IP→EID转换:
      IPv4地址放入16字节EID末尾(网络字节序)，前12字节为0
      示例: 192.0.0.1 → 0x000000000000000000000000c0000001

示例:
    $(basename "$0")
    $(basename "$0") -d /tmp/my_output

EOF
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_usage
            exit 0
            ;;
        -d|--dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            error_exit "未知参数: $1\n$(show_usage)"
            ;;
    esac
done

if [[ ! -d "$OUTPUT_DIR" ]]; then
    error_exit "输出目录不存在: $OUTPUT_DIR\n请先运行 generate_cluster_topo.sh 生成集群拓扑目录"
fi

echo "=========================================="
echo "  昇腾集群EID统一分配工具"
echo "=========================================="
echo ""
echo "输出目录: $OUTPUT_DIR"
echo ""

export OUTPUT_DIR

python3 << 'PYTHON_SCRIPT'
import json
import os
import re
import sys
import struct
import socket

OUTPUT_DIR = os.environ.get('OUTPUT_DIR', '')
if not OUTPUT_DIR:
    print("[错误] OUTPUT_DIR 环境变量未设置")
    sys.exit(1)

URMA_EID_LEN = 16

def ip_to_eid_string(ip, family=socket.AF_INET):
    """
    将IP地址转换为EID字符串。
    等价于C++函数:
        std::string IpToEidString(const std::string& ip, int family = AF_INET)

    转换逻辑:
        1. 将IPv4地址解析为4字节(网络字节序)
        2. 构造16字节EID: 前12字节为0，后4字节为IPv4地址
        3. 输出 "0x" + 32位十六进制字符串

    示例:
        ip_to_eid_string("192.0.0.1") -> "0x000000000000000000000000c0000001"
        ip_to_eid_string("192.0.0.16") -> "0x000000000000000000000000c0000010"
        ip_to_eid_string("193.3.0.5") -> "0x000000000000000000000000c1030005"
    """
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

def find_rootinfo_files(output_dir):
    rootinfo_files = []
    for root, dirs, files in os.walk(output_dir):
        for f in files:
            if 'rootinfo' in f and f.endswith('.json'):
                rootinfo_files.append(os.path.join(root, f))
    return rootinfo_files

def extract_indices(filepath):
    match = re.search(r'superpod(\d+)/server(\d+)', filepath)
    if match:
        return (int(match.group(1)), int(match.group(2)))
    return (-1, -1)

def allocate_ip(aa, bb, counter):
    cc = (counter >> 8) & 0xFF
    dd = counter & 0xFF
    return f"{aa}.{bb}.{cc}.{dd}"

rootinfo_files = find_rootinfo_files(OUTPUT_DIR)

if not rootinfo_files:
    print("[警告] 未找到任何rootinfo JSON文件")
    sys.exit(0)

rootinfo_files.sort(key=extract_indices)

print(f"[信息] 找到 {len(rootinfo_files)} 个rootinfo文件")
print(f"[信息] IP→EID转换规则: IPv4地址放入16字节EID末尾(网络字节序)")
print("")

total_addrs = 0

for filepath in rootinfo_files:
    sp_idx, srv_idx = extract_indices(filepath)

    if sp_idx < 0 or srv_idx < 0:
        print(f"  [警告] 无法从路径提取索引，跳过: {filepath}")
        continue

    aa = 192 + sp_idx
    bb = srv_idx + 1

    host_ip = f"{aa}.{bb}.0.0"
    host_eid = ip_to_eid_string(host_ip)

    with open(filepath, 'r') as f:
        data = json.load(f)

    DEVICE_PORT_START = 1
    port_counter = DEVICE_PORT_START
    addr_count = 0
    first_device_eid = None
    last_device_eid = None

    for rank in data.get('rank_list', []):
        for level in rank.get('level_list', []):
            for addr_entry in level.get('rank_addr_list', []):
                ip_addr = allocate_ip(aa, bb, port_counter)
                eid_str = ip_to_eid_string(ip_addr)
                addr_entry['addr'] = eid_str
                if first_device_eid is None:
                    first_device_eid = eid_str
                last_device_eid = eid_str
                port_counter += 1
                addr_count += 1

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print(f"  [成功] superpod{sp_idx}/server{srv_idx}: "
          f"分配 {addr_count} 个EID")
    print(f"         Host IP: {host_ip} → EID: {host_eid}")
    if first_device_eid:
        last_cc = ((port_counter - 1) >> 8) & 0xFF
        last_dd = (port_counter - 1) & 0xFF
        print(f"         Device IP: {aa}.{bb}.0.1 ~ {aa}.{bb}.{last_cc}.{last_dd}")
        print(f"         Device EID: {first_device_eid} ~ {last_device_eid}")

    total_addrs += addr_count

print("")
print(f"[信息] 共处理 {len(rootinfo_files)} 个文件，分配 {total_addrs} 个EID")

PYTHON_SCRIPT

echo ""
echo "=========================================="
echo "  分配完成"
echo "=========================================="
echo ""
success_msg "EID统一分配完成！"
