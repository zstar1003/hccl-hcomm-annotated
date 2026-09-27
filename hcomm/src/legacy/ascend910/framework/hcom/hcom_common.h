/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOM_COMMOM_H
#define HCOM_COMMOM_H

#include "hccl_comm_pub.h"
#include "hcom_common_v2.h"
#include "../common/src/topo/topoinfo_detect.h"

#include "topoinfo_struct.h"

HcclResult HcclGetDeviceId(void);

// Ge适配的类
constexpr u32 SEND_RECEIVE_TASK_NUM = 20;
constexpr u32 OP_DEFAULT_TASK_NUM = 245;
constexpr u32 AIV_DEFAULT_TASK_NUM = 4;  // 在AIV且非RDMA场景下，task数量固定为4
constexpr u32 DFX_DEFAULT_TASK_NUM = 16; // 2个计数 + 2个其它dfx + 2个memcpy_async + 预留10个
constexpr u32 DFX_PADDING_TASK_NUM = 4;
constexpr u32 MASTER_STREAM_EVENT_NUM = 3;
constexpr u32 SLAVE_STREAM_EVENT_NUM = 2;
constexpr u32 COM_STEP_NUM = 2;
constexpr s32 SERVER_NUM_ONE = 1;
constexpr s32 SERVER_NUM_EIGHT = 8;
constexpr u32 ALLREDUCE_DEFAULT_COM_STEP = 19;     // allgather + reducescatter
constexpr u32 ALLGATHER_DEFAULT_COM_STEP = 9;      // 5个通信 + 4个barrier
constexpr u32 REDUCESCATTER_DEFAULT_COM_STEP = 11; // 6个通信 + 5个barrier
constexpr u32 ALLTOALL_DEFAULT_COM_STEP = 14; // alltoall taskNum, 每个对端通信的最大task数量(图模式, NA+pairwise)
constexpr u32 TASK_NUM_DEVICE_FOUR = 4;
constexpr s32 TASK_NUM_DEVICE_ONE = 1;
constexpr u32 ALG_8P_RING_COMM_STEP = 7;
constexpr u32 PIPELINE_STREAM_EVENT_NUM = 2;
constexpr u32 MINUS_MESH_STREAM_NUM = 2;

enum class GeDeterministicOption { DISABLE = 0, ENABLE = 1, STRICT = 2 };

enum class RankInfoType {
    RANK_SIZE_IN_GROUP,
    RANK_ID_IN_GROUP,
    WORLD_RANK_ID_BY_GROUP,
    GROUP_RANK_ID_BY_WORLD,
    SERVER_NUM_IN_GROUP
};

static std::unordered_map<s32, u64> OFFLINE_BUILD_SUB_STEAM_NUM = {
    {HCCL_DEVICE_NUM_EIGHT, HCCL_SUB_STREAM_NUM_8P_RING},
    {HCCL_DEVICE_NUM_FOUR, HCCL_SUB_STREAM_NUM_4P_MESH},
    {HCCL_DEVICE_NUM_TWO, HCCL_SUB_STREAM_NUM_ZERO},
    {HCCL_DEVICE_MINNUM, HCCL_SUB_STREAM_NUM_ZERO},
};

constexpr u32 SINGLE_SERVER_NUM = 1;
using HcomOpTagInfo = struct HcomOpTagInfoCtx {
    std::map<std::string, u32> opIndex; // key: (group name) or (identifier), value: op index
};

using HcclGroupParams = struct TagHcclGroupParamsInfo {
    /* * group的基本构建信息，节点数及本节点在group中的编号、
    本节点在worldgroup中的编号、group的所有ranks */
    u32 worldRank;               /* * 用于标识world内不同节点 */
    u32 groupRank;               /* * 用于标识group内不同节点 */
    u32 serverNum;               /* * 用于标识group内服务器总数 */
    u32 totalRanks;              /* * 用于指示group内的节点总数, rank范围[0, totalRanks-1] */
    std::vector<u32> groupRanks; // 内部存储wordrankid，其下标表示groupid
    HcclCommPtr pSubComm;
    u32 refCounter = 0;
    bool destroyFlag = false;
};

using HcomInfo = struct HcomInfoTag {
    HcclCommPtr pComm;
    void* psComm;
    hccl::HcclCommParams params;
    std::unordered_map<std::string, HcclGroupParams>
        hcomGroupMap; // 每个group的信息(kname为服务器的server_id,按照服务器区分)
    std::mutex groupParamsLock;
    hccl::RankTable_t rankTable;
    s32 devId;
    bool cloudFlag;  // cloudFlag为0即实验室场景,cloudFlag为1则为云场景
    bool isHcomInit; // 标识是否为pytorch单算子通信域复用场景
    std::mutex backloggedGroupLock;
    std::map<std::string, std::vector<u32>> backloggedGroup; // 待创建的group
    std::map<std::string, std::shared_ptr<hccl::TopoInfoDetect>> hcclCommTopoInfoDetectServer;
    std::map<std::string, std::shared_ptr<hccl::TopoInfoDetect>> hcclCommTopoInfoDetectAgent;
    std::mutex groupRankNumMapLock;
    std::unordered_map<std::string, u32> groupRankNumMap; // 记录每个group的rank数量，用于topoInfo设置
    HcomInfoTag() : pComm(nullptr), devId(-1), cloudFlag(false), isHcomInit(false) {}

    ~HcomInfoTag()
    {
        pComm = nullptr;
        hcclCommTopoInfoDetectServer.clear();
        hcclCommTopoInfoDetectAgent.clear();
    }
};

HcclResult HcomSetGroupTopoInfo(const char* group, uint32_t rankSize);
void HcomUnSetGroupTopoInfo(const char* group);
HcclResult HcomGetCommByGroup(const char* group, std::shared_ptr<hccl::hcclComm>& hcclComm);
HcclResult HcomGetTopoDesc(const char* group, HcclTopoDescs* topoDescs, uint32_t topoSize);
s32 HcclGetThreadDeviceId();
void HcomGroupCallbackFuncInstall(
    HcclResult (*p1)(const std::string&, const std::vector<u32>&), bool (*p2)(HcomInfo&),
    HcclResult (*p3)(const std::string&), HcclResult (*p4)(HcomInfo&));
HcclResult DestroyFlag(const char* group, bool flag);
HcclResult HcomQueryGroupRef(const char* group, u32& groupRef);
bool HcomCheckrtMemcpyAddrAsync(const std::string& group = HCCL_WORLD_GROUP);
HcclResult HcomGetbackloggedByGroup(const char* group, std::vector<u32>& groupRanks, s32& groupSize);
HcomInfo& HcomGetCtxHomInfo(void);
HcclResult HcomInitCollComm(uint32_t rank, void** commV2, HcclCommPtr& comm);

#ifdef __cplusplus
extern "C" {
#endif
HcclResult HcomInitByFile(const char* rankTablePath, const char* identify);
#ifdef __cplusplus
}
#endif
#endif /* HCCL_COMM_PUB_H */
