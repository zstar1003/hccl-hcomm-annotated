/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "product_server.h"
#include <string.h>
#include <stdlib.h>
#include <syslog.h>
#include "rank_info_types.h"
#include "hal.h"
#include "topo.h"
#include "eid_util.h"
#include "securec.h"
#include "topo_addr_info_log.h"

#define MAX_SERVER_ROOTINFO_LEN (4096)
#define PRODUCT_MESH_LEVEL (0)
#define PRODUCT_CLOS_LEVEL (1)
#define IP_ADDR_LEN (32)
#define MAX_MAIN_BOARD_ID_NUM (16)
#define NPU_NUM_PER_BOARD (8)
#define NET_LAYER_ROCE (3)

/* 用于识别FE*/
#define MAX_UE_ID_IN_LEVEL (4)
#define MAX_LEVEL_NUM (4)
#define SERVER_NPU_NUM (8)
#define CLOS_LEVEL3 \
    {.level = NET_LAYER_ROCE, .netType = NET_TYPE_CLOS, .ueNum = 0, .instanceIdFunc = GetNetInstanceIdForCluster}
#define UBOE_CLUSTER_PLANE_ID "plane_uboe"
#define UB_RTP_CLUSTER_PLANE_ID "plane_ub_rtp"

enum UbEntityType {
    UE_TYPE_MESH = 0,
    UE_TYPE_CLOS = 1,
    UE_TYPE_UBOE = 2,
    UE_TYPE_UB_RTP = 3,
    UE_TYPE_CLOS_PORTS = 4,
};

typedef struct stUEInfo {
    int dieId;
    int feId;
    enum UbEntityType type;
    char ports[256];
} UEInfo;

typedef int (*GetNetInstanceIdFunc)(
    int npu_id, const struct dcmi_spod_info* spodInfo, char* netInstanceId, int netInstanceIdLen);

typedef struct stLevelInfo {
    int level;
    int ueNum;
    UEInfo ueList[MAX_UE_ID_IN_LEVEL];
    char netType[16];
    GetNetInstanceIdFunc instanceIdFunc;
} LevelInfo;

typedef struct _stUBRule {
    unsigned int mainBoardId[MAX_MAIN_BOARD_ID_NUM];
    unsigned int spodType;
    int levelNum;
    LevelInfo levelInfos[MAX_LEVEL_NUM];
} NetInfo;

/**
 *  获取OS级net instance id， 用于不组超节点形态
 */
int GetNetInstanceIdForOS(int npu_id, const struct dcmi_spod_info* spodInfo, char* netInstanceId, int netInstanceIdLen);

/**
 *  获取超节点中单个服务器的net instance id
 */
int GetNetInstanceIdForPod(
    int npu_id, const struct dcmi_spod_info* spodInfo, char* netInstanceId, int netInstanceIdLen);

/**
 *  获取Pod Flex机型的的net instance id
 * 该机型OS内存在2个8P mesh，8pmesh间无直连通道，因此net instance id需要分开
 */
int GetNetInstanceIdForPodFlex(
    int npu_id, const struct dcmi_spod_info* spodInfo, char* netInstanceId, int netInstanceIdLen);

/**
 *  获取超节点的net instance id
 */
int GetNetInstanceIdForSuperPod(
    int npu_id, const struct dcmi_spod_info* spodInfo, char* netInstanceId, int netInstanceIdLen);

int GetNetInstanceIdForCluster(
    int npu_id, const struct dcmi_spod_info* spodInfo, char* netInstanceId, int netInstanceIdLen);

#define MAX_UE_ID (99) // 定义一个MAX_UE_ID， mesh必须使用最大的UE
static const NetInfo g_netInfoList[] = {
    {
        .mainBoardId = {MAIN_BOARD_ID_SERVER_350L, MAIN_BOARD_INVALID},
        .spodType = TOPO_TYPE_IGNORE,
        .levelNum = 2, 
        {
            {
                .level = 0,
                .netType = NET_TYPE_TOPO_FILE_DESC,
                .ueNum = 3,
                .instanceIdFunc = GetNetInstanceIdForOS,
                .ueList = {
                    {.dieId = UDIE_1, .feId = 3, .type =UE_TYPE_MESH}, 
                    {.dieId = UDIE_1, .feId = 2, .type =UE_TYPE_CLOS_PORTS},
                    {.dieId = UDIE_1, .feId = 2, .type =UE_TYPE_CLOS}
                }
            },
            CLOS_LEVEL3,
        },
   },
   {
        .mainBoardId = {MAIN_BOARD_ID_SERVER_8PMESH, MAIN_BOARD_INVALID},
        .spodType = TOPO_TYPE_IGNORE,
        .levelNum = 3, 
        {
            {  // level 0 为OS内通信
                .level = 0,
                .netType = NET_TYPE_TOPO_FILE_DESC,
                .ueNum = 1,
                .instanceIdFunc = GetNetInstanceIdForPod,
                .ueList = {
                    {.dieId = UDIE_1, .feId = MAX_UE_ID, .type = UE_TYPE_MESH},
                }
            },
            { // level 1 为超平面
                .level = 1,
                .netType = NET_TYPE_CLOS,
                .ueNum = 1,
                .instanceIdFunc = GetNetInstanceIdForSuperPod,
                .ueList = { // 8口scaleup
                    {.dieId = UDIE_0, .feId = 3, .type = UE_TYPE_CLOS},
                },
            },
            CLOS_LEVEL3,
        },
   },
   {
        .mainBoardId = {MAIN_BOARD_ID_SERVER_8PMESH_UBOE, MAIN_BOARD_INVALID},
        .spodType = TOPO_TYPE_IGNORE,
        .levelNum = 4, 
        {
            {
                .level = 0, .netType = NET_TYPE_TOPO_FILE_DESC, .ueNum = 1,
                .instanceIdFunc = GetNetInstanceIdForPod,
                .ueList = { {.dieId = UDIE_1, .feId = MAX_UE_ID, .type = UE_TYPE_MESH}, }
            },
            {
                .level = 1, .netType = NET_TYPE_CLOS, .ueNum = 1,
                .instanceIdFunc = GetNetInstanceIdForSuperPod,
                .ueList = { {.dieId = UDIE_0, .feId = 3, .type =UE_TYPE_CLOS} },
            },
            { 
                .level = 2, .netType = NET_TYPE_CLOS, .ueNum = 1, .instanceIdFunc = GetNetInstanceIdForCluster,
                .ueList = { {.dieId = UDIE_0, .feId = 0, .type =UE_TYPE_UBOE, .ports = "1/8"} } 
            },
            CLOS_LEVEL3,
        },
   },
   {// 由两个8 NPU服务器组成一个16NPU的小超节点，共16个NPU组成fullmesh组网
        .mainBoardId = {MAIN_BOARD_ID_SERVER_8PMESH_UBOE, MAIN_BOARD_INVALID},
        .spodType = TOPO_TYPE_SERVER_16FM,  // 两个服务器组16p fullmesh
        .levelNum = 3, 
        {
            {
                .level = 0, .netType = NET_TYPE_TOPO_FILE_DESC, .ueNum = 2,
                .instanceIdFunc = GetNetInstanceIdForPod,
                .ueList = { 
                    {.dieId = UDIE_0, .feId = MAX_UE_ID, .type = UE_TYPE_MESH}, 
                    {.dieId = UDIE_1, .feId = MAX_UE_ID, .type = UE_TYPE_MESH}
                }
            },
            { // 支持UBOE出口
                .level = 2, .netType = NET_TYPE_CLOS, .ueNum = 1, .instanceIdFunc = GetNetInstanceIdForCluster,
                .ueList = { {.dieId = UDIE_0, .feId = 0, .type =UE_TYPE_UBOE, .ports = "1/8"} },
            },
            CLOS_LEVEL3,
        },
   },
   {
        .mainBoardId = { MAIN_BOARD_ID_SERVER_8PMESH_NOSP_UBOE, MAIN_BOARD_INVALID}, // 这种形态无超平面，但是有UBOE
        .spodType = TOPO_TYPE_IGNORE,
        .levelNum = 3, 
        {
            {
                .level = 0,
                .netType = NET_TYPE_TOPO_FILE_DESC,
                .ueNum = 1,
                .instanceIdFunc = GetNetInstanceIdForOS,
                .ueList = {
                    {.dieId = UDIE_1, .feId = MAX_UE_ID, .type = UE_TYPE_MESH},
                }
            },
            { 
                .level = 2, .netType = NET_TYPE_CLOS, .ueNum = 1, .instanceIdFunc = GetNetInstanceIdForCluster,
                .ueList = { {.dieId = UDIE_0, .feId = 0, .type =UE_TYPE_UBOE, .ports = "1/8"} } 
            },
            CLOS_LEVEL3,
        },
   },
   {
        .mainBoardId = {MAIN_BOARD_ID_SERVER_8PMESH_NOSP, MAIN_BOARD_INVALID}, // 无超平面，无UBOE
        .spodType = TOPO_TYPE_IGNORE,
        .levelNum = 2,
        {
            {
                .level = 0, .netType = NET_TYPE_TOPO_FILE_DESC, .ueNum = 1,
                .instanceIdFunc = GetNetInstanceIdForOS,
                .ueList = { {.dieId = UDIE_1, .feId = MAX_UE_ID, .type = UE_TYPE_MESH} }
            },
            CLOS_LEVEL3,
        },
   },
   {
        .mainBoardId = {MAIN_BOARD_ID_SERVER_TYPE1, MAIN_BOARD_INVALID},
        .spodType = TOPO_TYPE_IGNORE,
        .levelNum = 3, 
        {
            {
                .level = 0,
                .ueNum = 1,
                .ueList = {
                    {.dieId = 1, .feId = MAX_UE_ID, .type = UE_TYPE_MESH},
                }
            },
            {
                .level = 1,
                .ueNum = 2,
                .ueList = {
                    {.dieId = 0, .feId = 3, .type =UE_TYPE_CLOS },
                    {.dieId = 1, .feId = 2, .type =UE_TYPE_CLOS },
                }
            },
            CLOS_LEVEL3,
        },
   },
   {
        .mainBoardId = {MAIN_BOARD_ID_POD_FLEX, MAIN_BOARD_ID_POD_FLEX_RTP, MAIN_BOARD_INVALID},
        .spodType = TOPO_TYPE_IGNORE,
        .levelNum = 4, 
        {
            {
                .level = 0,
                .ueNum = 1,
                .netType = NET_TYPE_TOPO_FILE_DESC,
                .instanceIdFunc = GetNetInstanceIdForPodFlex,
                .ueList = {
                    {.dieId = 0, .feId = MAX_UE_ID, .type = UE_TYPE_MESH},
                }
            },
            {
                .level = 1,
                .ueNum = 1,
                .netType = NET_TYPE_CLOS,
                .instanceIdFunc = GetNetInstanceIdForSuperPod,
                .ueList = {
                    {.dieId = 1, .feId = 2, .type =UE_TYPE_CLOS },
                }
            },
            { 
                .level = 2, .netType = NET_TYPE_CLOS, .ueNum = 1, .instanceIdFunc = GetNetInstanceIdForCluster,
                .ueList = { {.dieId = UDIE_0, .feId = 0, .type = UE_TYPE_UB_RTP, .ports = "0/8"} } 
            },
            CLOS_LEVEL3, // level2 and level3 will not exist at same time
        },
   },
   {
        .mainBoardId = {MAIN_BOARD_ID_SERVER_550EL_100, MAIN_BOARD_INVALID},
        .spodType = TOPO_TYPE_IGNORE,
        .levelNum = 3, 
        {
            {
                .level = 0,
                .ueNum = 1,
                .netType = NET_TYPE_CLOS,
                .instanceIdFunc = GetNetInstanceIdForPod,
                .ueList = {
                    {.dieId = 1, .feId = 2, .type = UE_TYPE_CLOS},
                }
            },
            {
                .level = 1,
                .ueNum = 1,
                .netType = NET_TYPE_CLOS,
                .instanceIdFunc = GetNetInstanceIdForSuperPod,
                .ueList = {
                    {.dieId = 1, .feId = 2, .type = UE_TYPE_CLOS },
                }
            },
            CLOS_LEVEL3,
        },
   },
   {
        .mainBoardId = {MAIN_BOARD_ID_SERVER_550EL_200, MAIN_BOARD_INVALID},
        .spodType = TOPO_TYPE_IGNORE,
        .levelNum = 3, 
        {
            {
                .level = 0,
                .ueNum = 2,
                .netType = NET_TYPE_CLOS,
                .instanceIdFunc = GetNetInstanceIdForPod,
                .ueList = {
                    {.dieId = 0, .feId = 2, .type = UE_TYPE_CLOS},
                    {.dieId = 1, .feId = 2, .type = UE_TYPE_CLOS},
                }
            },
            {
                .level = 1,
                .ueNum = 2,
                .netType = NET_TYPE_CLOS,
                .instanceIdFunc = GetNetInstanceIdForSuperPod,
                .ueList = {
                    {.dieId = 0, .feId = 2, .type = UE_TYPE_CLOS},
                    {.dieId = 1, .feId = 2, .type = UE_TYPE_CLOS},
                }
            },
            CLOS_LEVEL3,
        },
   },
};

int ServerGetRootinfoLen(size_t* len)
{
    *len = MAX_SERVER_ROOTINFO_LEN;
    return 0;
}

int GetNetInstanceIdForOS(int npu_id, const struct dcmi_spod_info* spodInfo, char* netInstanceId, int netInstanceIdLen)
{
    (void)npu_id;
    (void)spodInfo;
    return get_server_id(netInstanceId, netInstanceIdLen);
}

int GetNetInstanceIdForPod(int npu_id, const struct dcmi_spod_info* spodInfo, char* netInstanceId, int netInstanceIdLen)
{
    (void)npu_id;
    return sprintf_s(netInstanceId, netInstanceIdLen, "sp_%u_srv_%u", spodInfo->super_pod_id, spodInfo->server_index);
}

int GetNetInstanceIdForPodFlex(
    int npu_id, const struct dcmi_spod_info* spodInfo, char* netInstanceId, int netInstanceIdLen)
{
    return sprintf_s(
        netInstanceId, netInstanceIdLen, "sp_%u_srv_%u_board%d", spodInfo->super_pod_id, spodInfo->server_index,
        npu_id / NPU_NUM_PER_BOARD);
}

int GetNetInstanceIdForSuperPod(
    int npu_id, const struct dcmi_spod_info* spodInfo, char* netInstanceId, int netInstanceIdLen)
{
    (void)npu_id;
    return sprintf_s(netInstanceId, netInstanceIdLen, "sp_%u", spodInfo->super_pod_id);
}

int GetNetInstanceIdForCluster(
    int npu_id, const struct dcmi_spod_info* spodInfo, char* netInstanceId, int netInstanceIdLen)
{
    (void)npu_id;
    (void)spodInfo;
    return sprintf_s(netInstanceId, netInstanceIdLen, "cluster");
}

static int IsMainBoardIdInList(unsigned int mainBoardId, const NetInfo* netInfo)
{
    for (int i = 0; i < MAX_MAIN_BOARD_ID_NUM; ++i) {
        if (netInfo->mainBoardId[i] == MAIN_BOARD_INVALID) {
            break;
        }
        if (netInfo->mainBoardId[i] == mainBoardId) {
            return 1;
        }
    }
    return 0;
}

static const NetInfo* GetNetInfo(unsigned int mainBoardId, unsigned int spodType)
{
    // 优先匹配满足mainboard id和 super pod type的情况
    for (size_t i = 0; i < sizeof(g_netInfoList) / sizeof(g_netInfoList[0]); ++i) {
        if (IsMainBoardIdInList(mainBoardId, &g_netInfoList[i]) && g_netInfoList[i].spodType == spodType) {
            return &g_netInfoList[i];
        }
    }
    //  忽略spod type，只匹配mainboard id
    for (size_t i = 0; i < sizeof(g_netInfoList) / sizeof(g_netInfoList[0]); ++i) {
        if (IsMainBoardIdInList(mainBoardId, &g_netInfoList[i])) {
            return &g_netInfoList[i];
        }
    }
    return NULL;
}

/*
 * @brief 处理mesh entity的地址, 将UBEntity中的地址加到layer中
 * 将UB entity中的所有地址按每个端口填入
 * @param layer: 网络层
 * @param ue: ue entity
 * @param spod_info: spod信息
 * @return int: 0 成功
 */
static int LayerAddMesh(const UBEntity* ue, NetLayer* layer, int type)
{
    for (unsigned int j = 0; j < ue->eidNum; ++j) {
        if (UrmaEidIsPortGroup(&ue->eidList[j].eid)) {
            continue;
        }
        Addr addr;
        (void)memset_s(&addr, sizeof(Addr), 0x00, sizeof(Addr));
        AddrSetEID(&addr, &ue->eidList[j].eid);
        char port[MAX_PORT_LEN] = {0};
        char planeId[MAX_PLANE_ID_LEN] = {0};
        int portId = UrmaEidGetPortId(&ue->eidList[j].eid);
        int dieId = UrmaEidGetDieId(&ue->eidList[j].eid);
        // topo中端口从0开始编，CNA中需要规避全0，从1开始
        (void)sprintf_s(port, MAX_PORT_LEN, "%d/%d", dieId, portId);
        if (type == UE_TYPE_MESH) {
            (void)sprintf_s(planeId, sizeof(planeId), "plane_%d", dieId);
        } else if (type == UE_TYPE_CLOS_PORTS) {
            (void)sprintf_s(planeId, sizeof(planeId), "plane_clos_%d_%d", dieId, portId);
        }
        AddrAddPort(&addr, port);
        AddrSetPlaneId(&addr, planeId);
        NetLayerAddAddr(layer, &addr);
    }
    return 0;
}

static int LayerAddClos(const UBEntity* ue, NetLayer* layer)
{
    int portGroupIdx = UBEntityGetPortGroupIdx(ue);
    if (portGroupIdx < 0) {
        return -1;
    }
    int dieId = UrmaEidGetDieId(&ue->eidList[portGroupIdx].eid);
    Addr addr;
    (void)memset_s(&addr, sizeof(Addr), 0x00, sizeof(Addr));
    AddrSetEID(&addr, &ue->eidList[portGroupIdx].eid);
    for (unsigned int i = 0; i < ue->eidNum; ++i) {
        if (UrmaEidIsPortGroup(&ue->eidList[i].eid)) {
            continue;
        }
        char port[MAX_PORT_LEN] = {0};
        (void)sprintf_s(port, MAX_PORT_LEN, "%d/%d", dieId, UrmaEidGetPortId(&ue->eidList[i].eid));
        AddrAddPort(&addr, port);
    }

    char planeId[MAX_PLANE_ID_LEN] = {0};
    (void)sprintf_s(planeId, sizeof(planeId), "plane_clos_%d", dieId);
    AddrSetPlaneId(&addr, planeId);
    NetLayerAddAddr(layer, &addr);
    return 0;
}

static int LayerAddUBOE(const UBEntity* ue, const UEInfo* ueInfo, NetLayer* layer)
{
#define INVLID_IP_PREFIX "254" // 该IP是有协议栈无DHCP等场景自动生成的本地私有地址前缀
    Addr addr;
    (void)memset_s(&addr, sizeof(Addr), 0x00, sizeof(Addr));
    int result = -1;
    for (unsigned int i = 0; i < ue->eidNum; ++i) {
        char cna[IP_ADDR_LEN] = {0};
        UrmaEid2CNA(&ue->eidList[i].eid, cna, sizeof(cna));
        if (strncmp(cna, INVLID_IP_PREFIX, strlen(INVLID_IP_PREFIX)) == 0) {
            continue;
        }
        AddrSetIP(&addr, cna);
        char port[MAX_PORT_LEN] = {0};
        int ret = sprintf_s(port, MAX_PORT_LEN, "%s", ueInfo->ports);
        if (ret < 0) {
            TOPO_ERR("set uboe port failed, ret = %d", ret);
            break;
        }
        AddrAddPort(&addr, port);
        result = 0;
    }
    if (result != 0) {
        TOPO_ERR("Find UBOE address failed, UE eid num = %d", ue->eidNum);
        return result;
    }
    AddrSetPlaneId(&addr, UBOE_CLUSTER_PLANE_ID);
    NetLayerAddAddr(layer, &addr);
    return result;
}

static int LayerAddUbRtp(const UBEntity* ue, const UEInfo* ueInfo, NetLayer* layer)
{
    Addr addr;
    (void)memset_s(&addr, sizeof(Addr), 0x00, sizeof(Addr));
    int result = -1;
    for (unsigned int i = 0; i < ue->eidNum; ++i) {
        AddrSetEID(&addr, &ue->eidList[i].eid);
        char port[MAX_PORT_LEN] = {0};
        int ret = sprintf_s(port, MAX_PORT_LEN, "%s", ueInfo->ports);
        if (ret < 0) {
            TOPO_ERR("set ub rtp port failed, ret = %d", ret);
            break;
        }
        AddrAddPort(&addr, port);
        result = 0;
    }
    if (result != 0) {
        TOPO_ERR("Find UB RTP EID failed, UE eid num = %d", ue->eidNum);
        return result;
    }
    AddrSetPlaneId(&addr, UB_RTP_CLUSTER_PLANE_ID);
    NetLayerAddAddr(layer, &addr);
    return result;
}

/**
 * @brief 根据dieId和feId从UEList中获取UBEntity
 * @param ueList: ue entity列表
 * @param dieId: die id
 * @param ueId: ue id
 * @param type: UB类型
 * @return UBEntity*: ue entity
 */
static const UBEntity* GetUBEntityByFilter(const UEList* ueList, int dieId, int ueId, int type)
{
    if (type == UE_TYPE_UBOE) {
        for (unsigned int i = 0; i < ueList->ueNum; i++) {
            if (UrmaEidIsUBOE(&ueList->ueList[i].eidList[0].eid)) {
                return &ueList->ueList[i];
            }
        }
        return NULL;
    }
    if (type == UE_TYPE_UB_RTP) {
        for (unsigned int i = 0; i < ueList->ueNum; i++) {
            if (UrmaEidIsUbRtp(&ueList->ueList[i].eidList[0].eid)) {
                return &ueList->ueList[i];
            }
        }
        return NULL;
    }
    int maxFe = 0;
    const UBEntity* ubEntity = NULL;
    for (unsigned int i = 0; i < ueList->ueNum; i++) {
        if (UrmaEidIsUBOE(&ueList->ueList[i].eidList[0].eid)) {
            continue;
        }
        int die = UrmaEidGetDieId(&ueList->ueList[i].eidList[0].eid);
        int fe = UBEntityGetId(&ueList->ueList[i]);
        if (die == dieId && fe == ueId) { // MAX_UE_ID足够大，这里不会匹配到
            return &ueList->ueList[i];
        }
        //  获取本iodie上UBEntity ID最大的UB Entity
        if (die == dieId && fe > maxFe) {
            maxFe = fe;
            ubEntity = &ueList->ueList[i];
        }
    }
    if (ueId == MAX_UE_ID) {
        return ubEntity;
    }
    return NULL;
}

/**
 * @brief 根据配置从ueList中过滤出ub entity, 并处理其中的地址添加到netlayer中
 * @param layer: 网络层信息
 * @param ueList: 全量的ue entity数组
 * @param levelInfo:  当前处理的层级信息
 * @param spodInfo: spod信息
 * @return int: 0 成功
 */
static int ProcessLayer(
    int npuId, NetLayer* layer, UEList* ueList, const LevelInfo* levelInfo, const struct dcmi_spod_info* spodInfo)
{
    /* RoCE 层（level 3）走独立路径，不查 UEList */
    if (levelInfo->level == NET_LAYER_ROCE) {
        return ProcessLayerRoce(npuId, layer);
    }

    char net_instance_id[MAX_INSTANCE_ID_LEN] = {0};
    levelInfo->instanceIdFunc(npuId, spodInfo, net_instance_id, sizeof(net_instance_id));
    NetLayerInit(layer, levelInfo->level, net_instance_id);
    NetLayerSetNetType(layer, levelInfo->netType);

    int ret = -1;
    for (int i = 0; i < levelInfo->ueNum; ++i) {
        int fe = levelInfo->ueList[i].feId;
        int die = levelInfo->ueList[i].dieId;
        int type = levelInfo->ueList[i].type;
        const UBEntity* ue = GetUBEntityByFilter(ueList, die, fe, type);
        if (ue == NULL) {
            TOPO_INFO(
                "NPU %d UB Entity not found for NetLayer %d, dieId = %d, UB Entity Id = %d, type = %d", npuId,
                levelInfo->level, die, fe, type);
            continue;
        }
        if (ue->eidNum == 0) {
            // UBOE, UBRTP not configured yet
            TOPO_INFO("NPU %d die %d UB Entity id %d has no EID", npuId, die, fe);
            continue;
        }
        if (type == UE_TYPE_MESH || type == UE_TYPE_CLOS_PORTS) {
            ret = LayerAddMesh(ue, layer, type);
        } else if (type == UE_TYPE_CLOS) {
            ret = LayerAddClos(ue, layer);
        } else if (type == UE_TYPE_UBOE) {
            ret = LayerAddUBOE(ue, &levelInfo->ueList[i], layer);
        } else if (type == UE_TYPE_UB_RTP) {
            ret = LayerAddUbRtp(ue, &levelInfo->ueList[i], layer);
        }
        if (ret != 0) {
            TOPO_INFO(
                "Unable to add UB Entity to NetLayer %d  dieId = %d feId = %d type = %d, maybe not configured",
                levelInfo->level, die, fe, type);
            break;
        }
    }
    return ret;
}

int ServerGetRootinfo(int npu_id, unsigned mainboard_id, void* buf, size_t* len)
{
    if (buf == NULL || len == NULL) {
        return RET_NOK;
    }
    struct dcmi_spod_info spod_info;
    UEList ueList;
    HalGetUBEntityList(npu_id, &ueList);
    hal_get_spod_info(npu_id, &spod_info);

    RootInfo rootinfo;
    Rank rank;
    RootInfoInit(&rootinfo);
    int localId = npu_id;
    if (spod_info.super_pod_type == TOPO_TYPE_SERVER_16FM) {
        localId = npu_id + (spod_info.server_index * SERVER_NPU_NUM);
    }
    RankInit(&rank, npu_id, localId);
    TopoGetFilePath(mainboard_id, spod_info.super_pod_type, rootinfo.topo_file_path, MAX_TOPO_PATH_LEN);

    const NetInfo* netInfo = GetNetInfo(mainboard_id, spod_info.super_pod_type);
    if (netInfo == NULL) {
        TOPO_ERR(
            "NPU phy id %d Get NetInfo Failed, MainBoardId %u, super pod type %d", npu_id, mainboard_id,
            spod_info.super_pod_type);
        return -1;
    }

    for (int i = 0; i < netInfo->levelNum; ++i) {
        NetLayer layer;
        if (ProcessLayer(npu_id, &layer, &ueList, &netInfo->levelInfos[i], &spod_info) == 0) {
            RankAddNetLayer(&rank, &layer);
        } else {
            TOPO_ERR(
                "NPU phy id %d ProcessLayer %d failed, MainBoardId %u", npu_id, netInfo->levelInfos[i].level,
                mainboard_id);
        }
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
