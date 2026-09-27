/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "product_card.h"
#include <string.h>
#include <stdlib.h>
#include <syslog.h>
#include "securec.h"
#include "rank_info_types.h"
#include "hal.h"
#include "topo.h"
#include "eid_util.h"
#include "topo_addr_info_log.h"

#define MAX_CARD_ROOTINFO_LEN (2048)
#define MAX_MESH_PORT_ID (9)
#define CARD_4P_MESH_NUM (4)
#define CARD_2P_MESH_NUM (2)

int GetCardRankInfoLen(size_t* len)
{
    (*len) = MAX_CARD_ROOTINFO_LEN;
    return 0;
}

/**
 *
 */
static int ProcessLayerMesh(int npu_id, NetLayer* layer, dcmi_urma_eid_info_t* eid_list, size_t eid_cnt)
{
    (void)npu_id;
    if (eid_cnt == 0) {
        return -1;
    }
    char server_id[MAX_INSTANCE_ID_LEN] = {0};
    char net_instance_id[MAX_INSTANCE_ID_LEN] = {0};
    get_server_id(server_id, sizeof(server_id));
    // 标卡没4个NPU一组， 可分多组， 标卡机头无单独的server id，因此使用mac地址作为server id
    // 和组ID组合起来作为mesh域的ID
    int ret = sprintf_s(net_instance_id, sizeof(net_instance_id), "%s", server_id);
    if (ret < 0) {
        return -1;
    }
    NetLayerInit(layer, 0, net_instance_id);
    NetLayerSetNetType(layer, NET_TYPE_MESH);
    for (size_t i = 0; i < eid_cnt; i++) {
        int portId = UrmaEidGetPortIdForCard(&eid_list[i].eid);
        if (portId > MAX_MESH_PORT_ID) {
            continue;
        }
        int dieId = UrmaEidGetDieIdForCard(&eid_list[i].eid);
        Addr addr;
        (void)memset_s(&addr, sizeof(Addr), 0x00, sizeof(Addr));
        AddrSetEID(&addr, &eid_list[i].eid);
        char port[MAX_PORT_LEN] = {0};
        char planeId[MAX_PLANE_ID_LEN] = {0};
        int ret1 = sprintf_s(port, MAX_PORT_LEN, "%d/%d", dieId, portId);
        int ret2 = sprintf_s(planeId, MAX_PORT_LEN, "plane_%d", dieId);
        if (ret1 < 0 || ret2 < 0) {
            break;
        }
        AddrAddPort(&addr, port);
        AddrSetPlaneId(&addr, planeId);
        NetLayerAddAddr(layer, &addr);
    }
    return 0;
}

static int ProcessLayerMesh2P(int npu_id, NetLayer* layer, dcmi_urma_eid_info_t* eid_list, size_t eid_cnt)
{
    if (eid_cnt == 0) {
        return -1;
    }
    char server_id[MAX_INSTANCE_ID_LEN] = {0};
    char net_instance_id[MAX_INSTANCE_ID_LEN] = {0};
    get_server_id(server_id, sizeof(server_id));
    // 2p标卡每2个NPU一组， 可分多组， 标卡机头无单独的server id，因此使用mac地址作为server id
    // 和组ID组合起来作为mesh域的ID
    int ret = sprintf_s(net_instance_id, sizeof(net_instance_id), "%s_%d", server_id, (npu_id / CARD_2P_MESH_NUM));
    if (ret < 0) {
        return -1;
    }
    NetLayerInit(layer, 0, net_instance_id);
    NetLayerSetNetType(layer, NET_TYPE_MESH);
    for (size_t i = 0; i < eid_cnt; i++) {
        int portId = UrmaEidGetPortIdForCard(&eid_list[i].eid);
        // 2P互联使用PortGroup EID
        if (portId <= MAX_MESH_PORT_ID) {
            continue;
        }
        Addr addr;
        (void)memset_s(&addr, sizeof(Addr), 0x00, sizeof(Addr));
        AddrSetEID(&addr, &eid_list[i].eid);
        const int ports[] = {4, 5, 6, 8}; // 2P互联固定使用4568端口
        for (int j = 0; j < 4; ++j) {
            char port[MAX_PORT_LEN] = {0};
            (void)sprintf_s(port, MAX_PORT_LEN, "0/%d", ports[j]);
            AddrAddPort(&addr, port);
        }
        AddrSetPlaneId(&addr, "plane_0");
        NetLayerAddAddr(layer, &addr);
    }
    return 0;
}

int GetCardRankInfo(int phyId, unsigned int mainboardId, void* buf, size_t* len)
{
    if (buf == NULL || len == NULL) {
        return RET_NOK;
    }
    RootInfo rootinfo;
    Rank rank;
    NetLayer layer_mesh;
    NetLayer layer_roce;
    RootInfoInit(&rootinfo);
    RankInit(&rank, phyId, phyId);
    TopoGetFilePath(mainboardId, TOPO_TYPE_IGNORE, rootinfo.topo_file_path, MAX_TOPO_PATH_LEN);

    dcmi_urma_eid_info_t eid_list[MAX_EID_NUM] = {0};
    size_t eid_cnt = MAX_EID_NUM;
    hal_get_eid_list_by_phy_id(phyId, eid_list, &eid_cnt);
    int result = -1;
    if (mainboardId == MAIN_BOARD_ID_CARD_4PMESH) {
        result = ProcessLayerMesh(phyId, &layer_mesh, eid_list, eid_cnt);
    }
    if (mainboardId == MAIN_BOARD_ID_CARD_2PMESH) {
        result = ProcessLayerMesh2P(phyId, &layer_mesh, eid_list, eid_cnt);
    }
    if (result == 0) { // 无UB互联的标卡无0层
        RankAddNetLayer(&rank, &layer_mesh);
    }
    if (ProcessLayerRoce(phyId, &layer_roce) == 0) {
        RankAddNetLayer(&rank, &layer_roce);
    }

    RootInfoAddRank(&rootinfo, &rank);
    char* rootinfo_buf = RootInfoToString(&rootinfo);
    if (rootinfo_buf == NULL) {
        TOPO_ERR("NPU phy id %d RootInfoToString failed, MainBoardId %u", phyId, mainboardId);
        return -1;
    }

    errno_t ret = strcpy_s(buf, *len, rootinfo_buf);
    if (ret != EOK) {
        TOPO_ERR(
            "NPU phy id %d strcpy_s failed MainBoardId %u space %ld actually size %ld", phyId, mainboardId, *len,
            strlen(rootinfo_buf));
    }
    (*len) = strlen(rootinfo_buf) + 1;
    free(rootinfo_buf);
    return ret;
}
