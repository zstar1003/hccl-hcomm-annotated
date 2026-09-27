/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "product_pod.h"
#include <string.h>
#include <stdlib.h>
#include <syslog.h>
#include "rank_info_types.h"
#include "hal.h"
#include "topo.h"
#include "eid_util.h"
#include "securec.h"
#include "topo_addr_info_log.h"

#define MAX_POD_ROOTINFO_LEN (2048)
#define PRODUCT_MESH_LEVEL (0)
#define PRODUCT_CLOS_LEVEL (1)
#define MAX_MESH_PORT_ID (9)
#define NPU_NUM (8)

int PodGetRootinfoLen(size_t* len)
{
    *len = MAX_POD_ROOTINFO_LEN;
    return 0;
}

typedef struct rule {
    unsigned int mainboardId;
    int level; // 网络层级  0 mesh, 1 scaleup, 2 UBOE/UB_RTP, 3 ROCE
    int dieId; // io die id NPU有两个iodie, 0和1
    int ueId;  // UBEntity ID， 用于定义UB实体的功能
    int ports[MAX_PORT_NUM];
    int portNum;
    int npus[MAX_PORT_NUM];
    int npuNum;
    char planeId[MAX_PLANE_ID_LEN];
} UBEntityRule;

/**
 * @brief PoD形态的通信规划
 * 每个NPU出8个口上scaleup网络，一个iodie出6口到平面0， 另一个iodie出2口到平面1
 * 其中前4个NPU:  iodie 0出6口， iodie 1出2口
 * 其中后4个NPU:  iodie 0出2口， iodie 1出6口
 * 这个结构体用于定义这些规则
 */
static const UBEntityRule g_ubrules[] = {
    // NPU 0-3 的1die使用6个端口，连平面0，平面0只和平面0通信
    {MAIN_BOARD_ID_POD, PRODUCT_CLOS_LEVEL, 1, 2, {0, 1, 2, 3, 5, 6}, 6, {0, 1, 2, 3}, 4, "plane_pg_0"},
    // NPU 0-3 的0die使用2个端口，连平面1，平面1只和平面1通信
    {MAIN_BOARD_ID_POD, PRODUCT_CLOS_LEVEL, 0, 2, {1, 2}, 2, {0, 1, 2, 3}, 4, "plane_pg_1"},
    // NPU 4-7 的0 die使用6个端口
    {MAIN_BOARD_ID_POD, PRODUCT_CLOS_LEVEL, 0, 2, {0, 1, 2, 3, 4, 5}, 6, {4, 5, 6, 7}, 4, "plane_pg_0"},
    // NPU 4-7 的1 die使用2个端口
    {MAIN_BOARD_ID_POD, PRODUCT_CLOS_LEVEL, 1, 2, {1, 2}, 2, {4, 5, 6, 7}, 4, "plane_pg_1"},
    // NPU 0-3 的1die使用6个端口
    {MAIN_BOARD_ID_POD_2D, PRODUCT_CLOS_LEVEL, 1, 2, {0, 1, 2, 3, 5, 6}, 6, {0, 1, 2, 3}, 4, "plane_pg_0"},
    // NPU 0-3 的0die使用2个端口
    {MAIN_BOARD_ID_POD_2D, PRODUCT_CLOS_LEVEL, 0, 2, {1, 2}, 2, {0, 1, 2, 3}, 4, "plane_pg_1"},
    // NPU 4-7 的0 die使用6个端口
    {MAIN_BOARD_ID_POD_2D, PRODUCT_CLOS_LEVEL, 0, 2, {0, 1, 2, 3, 4, 5}, 6, {4, 5, 6, 7}, 4, "plane_pg_0"},
    // NPU 4-7 的1 die使用2个端口
    {MAIN_BOARD_ID_POD_2D, PRODUCT_CLOS_LEVEL, 1, 2, {1, 2}, 2, {4, 5, 6, 7}, 4, "plane_pg_1"},
};

static int ProcessLayerMesh(int npu_id, NetLayer* layer, UEList* ueList, struct dcmi_spod_info* spod_info)
{
    char net_instance_id[MAX_INSTANCE_ID_LEN] = {0};
    (void)sprintf_s(
        net_instance_id, sizeof(net_instance_id), "sp%ld_srv%ld", spod_info->super_pod_id, spod_info->server_index);
    NetLayerInit(layer, PRODUCT_MESH_LEVEL, net_instance_id);
    NetLayerSetNetType(layer, NET_TYPE_MESH);

    const int meshDieId = npu_id % 8 < 4 ? 0 : 1;
    int meshEntityId = UBGetMaxEntityId(ueList, meshDieId);
    for (unsigned int i = 0; i < ueList->ueNum; i++) {
        int fe = UBEntityGetId(&ueList->ueList[i]);
        int dieId = UBEntityGetDieId(&ueList->ueList[i]);
        if (fe != meshEntityId || dieId != meshDieId) {
            continue;
        }
        for (unsigned int j = 0; j < ueList->ueList[i].eidNum; ++j) {
            int phyPortId = UrmaEidGetPortId(&ueList->ueList[i].eidList[j].eid);
            if (phyPortId > MAX_MESH_PORT_ID) {
                continue;
            }
            Addr addr;
            (void)memset_s(&addr, sizeof(Addr), 0x00, sizeof(Addr));
            AddrSetEID(&addr, &ueList->ueList[i].eidList[j].eid);
            int portId = UrmaEidGetPortId(&ueList->ueList[i].eidList[j].eid);
            char port[MAX_PORT_LEN] = {0};
            (void)sprintf_s(port, MAX_PORT_LEN, "%d/%d", dieId, portId);
            AddrAddPort(&addr, port);
            AddrSetPlaneId(&addr, "plane_0");
            NetLayerAddAddr(layer, &addr);
        }
    }
    return 0;
}

/**
 * @brief 根据NPU ID, 网络层级, 主板ID, iodie ID, UE ID获取对应的UB实体规则
 * @param npu_id NPU ID
 * @param level 网络层级
 * @param mainBoardId 主板ID
 * @param die iodie ID
 * @param feId UE ID
 * @return const UBEntityRule* 对应的UB实体规则
 */
static const UBEntityRule* GetUBRule(int npu_id, int level, unsigned int mainBoardId, int die, int feId)
{
    for (size_t i = 0; i < sizeof(g_ubrules) / sizeof(UBEntityRule); ++i) {
        if (g_ubrules[i].level != level) {
            continue;
        }
        int isNpuInRule = 0;
        for (int j = 0; j < g_ubrules[i].npuNum; ++j) {
            if (g_ubrules[i].npus[j] == npu_id) {
                isNpuInRule = 1;
                break;
            }
        }
        if (isNpuInRule == 0) {
            continue;
        }
        if (g_ubrules[i].level != level) {
            continue;
        }
        if (mainBoardId == g_ubrules[i].mainboardId && die == g_ubrules[i].dieId && feId == g_ubrules[i].ueId) {
            return &g_ubrules[i];
        }
    }
    return NULL;
}

static int ProcessLayerClos(
    int npu_id, unsigned int mainBoardId, NetLayer* layer, UEList* ueList, struct dcmi_spod_info* spod_info)
{
    char net_instance_id[MAX_INSTANCE_ID_LEN] = {0};
    (void)sprintf_s(net_instance_id, sizeof(net_instance_id), "superpod_%ld", spod_info->super_pod_id);
    NetLayerInit(layer, PRODUCT_CLOS_LEVEL, net_instance_id);
    NetLayerSetNetType(layer, NET_TYPE_CLOS);

    for (unsigned int i = 0; i < ueList->ueNum; ++i) {
        int fe = UBEntityGetId(&ueList->ueList[i]);
        int portGroupIdx = UBEntityGetPortGroupIdx(&ueList->ueList[i]);
        if (portGroupIdx < 0) {
            continue;
        }
        int die = UrmaEidGetDieId(&ueList->ueList[i].eidList[portGroupIdx].eid);
        const UBEntityRule* rule = GetUBRule((npu_id % NPU_NUM), PRODUCT_CLOS_LEVEL, mainBoardId, die, fe);
        if (rule == NULL) {
            continue;
        }
        Addr addr;
        (void)memset_s(&addr, sizeof(Addr), 0x00, sizeof(Addr));
        AddrSetEID(&addr, &ueList->ueList[i].eidList[portGroupIdx].eid);

        int portNum = 0;
        for (unsigned int j = 0; j < ueList->ueList[i].eidNum; ++j) {
            if (UrmaEidIsPortGroup(&ueList->ueList[i].eidList[j].eid)) {
                continue;
            }
            int portId = UrmaEidGetPortId(&ueList->ueList[i].eidList[j].eid);
            char port[MAX_PORT_LEN] = {0};
            (void)sprintf_s(port, MAX_PORT_LEN, "%d/%d", die, portId);
            AddrAddPort(&addr, port);
            portNum++;
        }

        // CCU需使用相同的6口建联，需要把6口的portgroup排在前面
        const int primaryPortNum = 6;
        if (portNum >= primaryPortNum) {
            AddrSetPlaneId(&addr, "plane_0");
            NetLayerSetAddrAt(layer, &addr, 0);
        } else {
            AddrSetPlaneId(&addr, "plane_1");
            NetLayerSetAddrAt(layer, &addr, 1);
        }
    }
    return 0;
}

int PodGetRootinfo(int npu_id, unsigned mainboard_id, void* buf, size_t* len)
{
    if (buf == NULL || len == NULL) {
        return RET_NOK;
    }
    RootInfo rootinfo;
    Rank rank;
    NetLayer layerMesh;
    NetLayer layerClos;
    NetLayer layerRoce;
    RootInfoInit(&rootinfo);

    TopoGetFilePath(mainboard_id, TOPO_TYPE_IGNORE, rootinfo.topo_file_path, MAX_TOPO_PATH_LEN);
    struct dcmi_spod_info spod_info;
    UEList ueList;
    HalGetUBEntityList(npu_id, &ueList);
    hal_get_spod_info(npu_id, &spod_info);
    // local id在PoD框内取值从0-63， 本OS只能看到device_id, 需要加上server_index * 8构成完整的0-63
    int localId = (npu_id % NPU_NUM) + ((spod_info.server_index % NPU_NUM) * NPU_NUM);
    RankInit(&rank, npu_id, localId);

    if (ProcessLayerMesh(npu_id, &layerMesh, &ueList, &spod_info) == 0) {
        RankAddNetLayer(&rank, &layerMesh);
    }
    if (ProcessLayerClos(npu_id, mainboard_id, &layerClos, &ueList, &spod_info) == 0) {
        RankAddNetLayer(&rank, &layerClos);
    }

    /* L3 RoCE 层：当 XML 提供 IP 时追加 */
    if (ProcessLayerRoce(npu_id, &layerRoce) == 0) {
        RankAddNetLayer(&rank, &layerRoce);
    }

    RootInfoAddRank(&rootinfo, &rank);
    char* rootinfo_buf = RootInfoToString(&rootinfo);
    if (rootinfo_buf == NULL) {
        TOPO_ERR("NPU phy id %d RootInfoToString failed, MainBoardId %u", npu_id, mainboard_id);
        return -1;
    }
    errno_t ret = strcpy_s(buf, *len, rootinfo_buf);
    if (ret != EOK) {
        TOPO_ERR(
            "NPU phy id %d strcpy_s failed MainBoardId %u space %ld actually size %ld", npu_id, mainboard_id, *len,
            strlen(rootinfo_buf));
    }
    (*len) = strlen(rootinfo_buf) + 1;
    free(rootinfo_buf);
    return ret;
}
