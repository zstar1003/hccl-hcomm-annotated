/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "rank_info_types.h"
#include "npu_nic_affinity.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "securec.h"
#include "hal.h"
#include "topo_addr_info_log.h"
#include "topo_addr_info_perf.h"

#define MAX_PORTS_STR_LEN (512)

char* AddrToString(const Addr* addr)
{
    const size_t max_buffer_size = 102400;
    char* buf = (char*)malloc(max_buffer_size);
    int ret = 0;
    if (buf == NULL) {
        return NULL;
    }
    (void)memset_s(buf, max_buffer_size, 0, max_buffer_size);
    char ports[MAX_PORTS_STR_LEN] = {0};
    for (int i = 0; i < addr->port_count; i++) {
        if (i > (MAX_PORT_NUM - 1)) {
            break;
        }
        char port[MAX_PORT_LEN] = {0};
        ret = sprintf_s(port, sizeof(port), "\"%s\"", addr->ports[i]);
        if (ret < 0) {
            break;
        }
        errno_t r = strcat_s(ports, MAX_PORTS_STR_LEN, port);
        if (r != 0) {
            break;
        }
        if (i != addr->port_count - 1) {
            r = strcat_s(ports, MAX_PORTS_STR_LEN, ",");
            if (r != 0) {
                break;
            }
        }
    }
    ret = sprintf_s(
        buf, max_buffer_size, "{\"addr_type\": \"%s\", \"addr\": \"%s\", \"plane_id\": \"%s\", \"ports\": [%s]}",
        addr->addr_type, addr->addr, addr->plane_id, ports);
    if (ret < 0) {
        free(buf);
        buf = NULL;
    }
    return buf;
}

void NetLayerInit(NetLayer* layer, int level, const char* layer_id)
{
    (void)memset_s(layer, sizeof(NetLayer), 0, sizeof(NetLayer));
    layer->net_layer = level;
    layer->addr_count = 0;
    (void)strcpy_s(layer->net_instance_id, sizeof(layer->net_instance_id), layer_id);
}

void NetLayerSetNetType(NetLayer* layer, const char* net_type)
{
    (void)strcpy_s(layer->net_type, sizeof(layer->net_type), net_type);
}

void NetLayerAddAddr(NetLayer* layer, const Addr* addr)
{
    if (layer->addr_count >= MAX_ADDR_NUM) {
        return;
    }
    (void)memcpy_s(&layer->rank_addr_list[layer->addr_count], sizeof(Addr), addr, sizeof(Addr));
    layer->addr_count++;
}

void NetLayerSetAddrAt(NetLayer* layer, const Addr* addr, int index)
{
    if (index < 0 || index >= MAX_ADDR_NUM) {
        return;
    }
    (void)memcpy_s(&layer->rank_addr_list[index], sizeof(Addr), addr, sizeof(Addr));
    layer->addr_count++;
}

char* NetLayerToString(const NetLayer* layer)
{
    const size_t max_buffer_size = 102400;
    char* buf = (char*)malloc(max_buffer_size);
    if (buf == NULL) {
        return NULL;
    }
    char* addr_list = (char*)malloc(max_buffer_size);
    if (addr_list == NULL) {
        free(buf);
        return NULL;
    }
    (void)memset_s(addr_list, max_buffer_size, 0, max_buffer_size);
    (void)memset_s(buf, max_buffer_size, 0, max_buffer_size);
    for (int i = 0; i < layer->addr_count; i++) {
        char* addr = AddrToString(&layer->rank_addr_list[i]);
        if (addr == NULL) {
            continue;
        }
        if (strcat_s(addr_list, max_buffer_size, addr) != 0) {
            free(addr);
            break;
        }
        if (i != layer->addr_count - 1) {
            if (strcat_s(addr_list, max_buffer_size, ",") != 0) {
                free(addr);
                break;
            }
        }
        free(addr);
    }
    int ret = sprintf_s(
        buf, max_buffer_size,
        "{\"net_layer\": %d, \"net_instance_id\": \"%s\", \"net_type\": \"%s\",\"net_attr\":\"%s\","
        "\"rank_addr_list\": [%s]}",
        layer->net_layer, layer->net_instance_id, layer->net_type, layer->net_attr, addr_list);
    free(addr_list);
    if (ret < 0) {
        free(buf);
        buf = NULL;
    }
    return buf;
}

char* RankListToString(const RootInfo* rootinfo)
{
    const size_t max_buffer_size = 102400;
    char* buf = (char*)malloc(max_buffer_size);
    if (buf == NULL) {
        return NULL;
    }
    (void)memset_s(buf, max_buffer_size, 0, max_buffer_size);
    for (int i = 0; i < rootinfo->rank_count; i++) {
        char* rank = RankToString(&rootinfo->ranks[i]);
        errno_t ret = strcat_s(buf, max_buffer_size, rank);
        if (ret != 0) {
            free(rank);
            break;
        }
        if (i != rootinfo->rank_count - 1) {
            if (strcat_s(buf, max_buffer_size, ",") != 0) {
                free(rank);
                break;
            }
        }
        free(rank);
    }
    return buf;
}

void RankInit(Rank* rank, int deviceId, int localId)
{
    (void)memset_s(rank, sizeof(Rank), 0, sizeof(Rank));
    rank->device_id = deviceId;
    rank->local_id = localId;
    rank->level_count = 0;
}

void RankAddNetLayer(Rank* rank, const NetLayer* layer)
{
    if (rank->level_count >= MAX_NET_LEVEL_NUM) {
        return;
    }
    (void)memcpy_s(&rank->level_list[rank->level_count], sizeof(NetLayer), layer, sizeof(NetLayer));
    rank->level_count++;
}

char* RootInfoToString(const RootInfo* rootinfo)
{
    const size_t max_buffer_size = 102400;
    char* buf = (char*)malloc(max_buffer_size);
    if (buf == NULL) {
        return NULL;
    }
    (void)memset_s(buf, max_buffer_size, 0, max_buffer_size);
    char* rank_list = RankListToString(rootinfo);
    int ret = sprintf_s(
        buf, max_buffer_size,
        "{\"version\": \"%s\", \"topo_file_path\": \"%s\","
        "\"rank_count\": %d, \"rank_list\": [%s]}",
        rootinfo->version, rootinfo->topo_file_path, rootinfo->rank_count, rank_list);
    free(rank_list);
    if (ret < 0) {
        free(buf);
        buf = NULL;
    }
    return buf;
}

void RootInfoInit(RootInfo* rootinfo)
{
    (void)memset_s(rootinfo, sizeof(RootInfo), 0, sizeof(RootInfo));
    (void)strcpy_s(rootinfo->version, sizeof(rootinfo->version), "2.0");
    rootinfo->rank_count = 0;
}

void RootInfoAddRank(RootInfo* rootinfo, const Rank* rank)
{
    if (rootinfo->rank_count >= MAX_RANK_NUM) {
        return;
    }
    (void)memcpy_s(&rootinfo->ranks[rootinfo->rank_count], sizeof(Rank), rank, sizeof(Rank));
    rootinfo->rank_count++;
}

void AddrInit(Addr* addr) { (void)memset_s(addr, sizeof(Addr), 0, sizeof(Addr)); }

void AddrSetEID(Addr* addr, const dcmi_urma_eid_t* eid)
{
    for (int i = 0; i < DCMI_URMA_EID_SIZE; i++) {
        int ret = sprintf_s(&addr->addr[i * 2], MAX_NET_ADDR_LEN - (i * 2), "%02x", eid->raw[i]);
        if (ret < 0) {
            return;
        }
    }
    (void)strcpy_s(addr->addr_type, sizeof(addr->addr_type), "EID");
}

void AddrSetIP(Addr* addr, const char* ip)
{
    (void)strcpy_s(addr->addr, sizeof(addr->addr), ip);
    (void)strcpy_s(addr->addr_type, sizeof(addr->addr_type), "IPV4");
}

void AddrSetPlaneId(Addr* addr, const char* plane_id)
{
    (void)strcpy_s(addr->plane_id, sizeof(addr->plane_id), plane_id);
}

int AddrAddPort(Addr* addr, const char* port)
{
    if (addr->port_count >= (MAX_PORT_NUM - 1)) {
        return -1;
    }
    return strcpy_s(addr->ports[addr->port_count++], MAX_PORT_LEN, port);
}

char* RankToString(const Rank* rank)
{
    const size_t max_buf_size = 102400;
    char* buf = (char*)malloc(max_buf_size);
    if (buf == NULL) {
        return NULL;
    }
    char* level_list = (char*)malloc(max_buf_size);
    if (level_list == NULL) {
        free(buf);
        return NULL;
    }
    (void)memset_s(buf, max_buf_size, 0, max_buf_size);
    (void)memset_s(level_list, max_buf_size, 0, max_buf_size);
    for (int i = 0; i < rank->level_count; ++i) {
        char* layer = NetLayerToString(&rank->level_list[i]);
        if (strcat_s(level_list, max_buf_size, layer) != 0) {
            free(layer);
            break;
        }
        if (i != (rank->level_count - 1)) {
            if (strcat_s(level_list, max_buf_size, ", ") != 0) {
                free(layer);
                break;
            }
        }
        free(layer);
    }
    int ret = sprintf_s(
        buf, max_buf_size, "{\"device_id\": %d, \"local_id\": %d, \"level_list\": [%s]}", rank->device_id,
        rank->local_id, level_list);
    free(level_list);
    if (ret < 0) {
        free(buf);
        buf = NULL;
    }
    return buf;
}

TopoAddrResult ProcessLayerRoce(int npu_id, NetLayer* layer)
{
    TOPO_PERF_BEGIN(ProcessLayerRoce);
#define PROC_ROCE_LEVEL (3)
    char ip[MAX_ADDR_LEN] = {0};
    TopoAddrResult ret = GetRoceIpFromXml(npu_id, ip, sizeof(ip));
    if (ret != TOPO_SUCCESS || ip[0] == '\0') {
        TOPO_INFO("ProcessLayerRoce: GetRoceIpFromXml failed for npu_id=%d, ret=%d", npu_id, ret);
        TOPO_PERF_END(ProcessLayerRoce);
        return (ret != TOPO_SUCCESS) ? ret : TOPO_ERR_NOT_FOUND;
    }
    NetLayerInit(layer, PROC_ROCE_LEVEL, "cluster");
    NetLayerSetNetType(layer, NET_TYPE_CLOS);
    Addr addr;
    (void)memset_s(&addr, sizeof(Addr), 0x00, sizeof(Addr));
    AddrSetIP(&addr, ip);
    (void)strcpy_s(addr.plane_id, MAX_PLANE_ID_LEN, "plane0");
    AddrAddPort(&addr, "d2h");
    NetLayerAddAddr(layer, &addr);
    TOPO_PERF_END(ProcessLayerRoce);
    return TOPO_SUCCESS;
}
