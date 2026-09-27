/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __RANK_INFO_TYPES_H__
#define __RANK_INFO_TYPES_H__

#include "hal.h"
#include "topo_addr_info_err.h"

#ifdef __cplusplus
extern "C" {
#endif

#define MAX_TOPO_PATH_LEN (256)
#define MAX_ADDR_LEN (64)
#define MAX_INSTANCE_ID_LEN (64)
#define MAX_PLANE_ID_LEN (64)
#define MAX_PORT_LEN (16)
#define MAX_PORT_NUM (32)
#define MAX_NET_TYPE_LEN (16)
#define MAX_NET_LEVEL_NUM (8) // 最大网络层级数

#define MAX_ADDR_TYPE_LEN (8) // 网络地址类型最大长度，例如：EID, IPV4
#define MAX_RANK_NUM (8)
#define MAX_VERSION_LEN (32)
#define MAX_STATUS_LEN (32)
#define MAX_ADDR_NUM (32)     // 最大地址个数
#define MAX_NET_ADDR_LEN (33) // 保留结束符

#define NET_TYPE_TOPO_FILE_DESC "TOPO_FILE_DESC"
#define NET_TYPE_MESH "TOPO_FILE_DESC"
#define NET_TYPE_CLOS "CLOS"

#define UDIE_0 (0)
#define UDIE_1 (1)

#define RET_OK (0)
#define RET_NOK (-1)

typedef struct stAddr {
    char addr_type[MAX_ADDR_TYPE_LEN];
    char addr[MAX_NET_ADDR_LEN];
    int port_count;
    char ports[MAX_PORT_NUM][MAX_PORT_LEN];
    char plane_id[MAX_PLANE_ID_LEN];
} Addr;

typedef struct stNetLayer {
    int net_layer;
    char net_instance_id[MAX_INSTANCE_ID_LEN];
    char net_type[MAX_NET_TYPE_LEN];
    char net_attr[MAX_NET_ADDR_LEN];
    int addr_count;
    Addr rank_addr_list[MAX_ADDR_NUM];
} NetLayer;

typedef struct stRank {
    int device_id;
    int local_id;
    int level_count;
    NetLayer level_list[MAX_NET_LEVEL_NUM];
} Rank;

typedef struct stRootInfo {
    char version[MAX_VERSION_LEN];
    char status[MAX_STATUS_LEN];
    char topo_file_path[MAX_TOPO_PATH_LEN];
    int rank_count;
    Rank ranks[MAX_RANK_NUM];
} RootInfo;

char* AddrToString(const Addr* addr);
void AddrInit(Addr* addr);
void AddrSetEID(Addr* addr, const dcmi_urma_eid_t* eid);
void AddrSetIP(Addr* addr, const char* ip_addr);
void AddrSetPlaneId(Addr* addr, const char* plane_id);
int AddrAddPort(Addr* addr, const char* port);

void NetLayerInit(NetLayer* net_layer, int level, const char* layer_id);
void NetLayerSetNetInstanceId(NetLayer* net_layer, const char* net_instance_id);
void NetLayerAddAddr(NetLayer* net_layer, const Addr* addr);
void NetLayerSetAddrAt(NetLayer* net_layer, const Addr* addr, int index);
void NetLayerSetNetType(NetLayer* layer, const char* net_type);
char* NetLayerToString(const NetLayer* net_layer);

void RankInit(Rank* rank, int device_id, int local_id);
void RankAddNetLayer(Rank* rank, const NetLayer* layer);
char* RankToString(const Rank* rank);

void RootInfoInit(RootInfo* root_info);
void RootInfoAddRank(RootInfo* root_info, const Rank* rank);
char* RootInfoToString(const RootInfo* root_info);

TopoAddrResult ProcessLayerRoce(int npu_id, NetLayer* layer);

#ifdef __cplusplus
}
#endif

#endif // __RANK_INFO_TYPES_H__
