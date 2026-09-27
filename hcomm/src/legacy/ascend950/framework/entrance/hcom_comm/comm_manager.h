/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOM_COMMOM_V2_H
#define HCOM_COMMOM_V2_H
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include <algorithm>

#include "hccl/base.h"
#include "hccl_common_v2.h"
#include "hccl/hccl_types.h"
#include "hccl_communicator.h"
#include "log.h"
#include <hccl/hccl_types.h>
#include "ccu_driver_handle.h"

const u64 RANKTABLE_FILE_MAX_SIZE = 1024ULL * 1024 * 1024;
constexpr s32 HOST_DEVICE_ID = -1; // device id 无效值
u64 GetFileSize(const std::string& path);

using HcclGroupParamsV2 = struct TagHcclGroupParamsInfoV2 {
    /* * group的基本构建信息，节点数及本节点在group中的编号、
    本节点在worldgroup中的编号、group的所有ranks */
    u32 worldRank;               /* * 用于标识world内不同节点 */
    u32 groupRank;               /* * 用于标识group内不同节点 */
    u32 serverNum;               /* * 用于标识group内服务器总数 */
    u32 totalRanks;              /* * 用于指示group内的节点总数, rank范围[0, totalRanks-1] */
    std::vector<u32> groupRanks; // 内部存储wordrankid，其下标表示groupid
    std::shared_ptr<Hccl::HcclCommunicator> pComm;
    u32 refCounter = 0;
    bool destroyFlag = false;
};

MAKE_ENUM(DeviceStatus, DEVICE_IDLE = 0, DEVICE_RECOVERED, DEVICE_READY);

constexpr u32 MAX_NUM_COMM_USING_MS = 1;
struct CcuStatus {
    std::vector<std::string> useMsCommIds{};
    std::vector<std::string> useSchedCommIds{};

    void RemoveCommId(const std::string& commId);
    bool IsMsAvailable(const std::string& commId) const;
    HcclResult InsertCommId(const std::string& commId, bool isUsingCcuMs, bool isUsingCcuSched);
    HcclResult InsertMsCommId(const std::string& commId);
    void InsertSchedCommId(const std::string& commId);
};

using HcclCommInfoV2 = struct HcclCommInfoCtxV2 {
    s32 devId{HOST_DEVICE_ID};
    std::shared_ptr<Hccl::HcclCommunicator> pComm{nullptr};
    Hccl::CommParams commParams;
    std::map<std::string, HcclGroupParamsV2> hcclGroupMap;
    std::mutex groupParamsLock; // 操作hcclGroupMap前加锁
    bool isUsed{false};
    DeviceStatus status{DeviceStatus::DEVICE_IDLE}; // Device状态
    u64 step{0};                                    // 新增
    CcuStatus ccuStatus;                            // 管理ccu资源使用情况

    ~HcclCommInfoCtxV2()
    {
        hcclGroupMap.clear();
        pComm = nullptr;
    }
};

extern std::mutex g_commInfoV2CtxMutex;

class CommManager {
public:
    CommManager(const CommManager& that) = delete;
    CommManager& operator=(const CommManager& that) = delete;

    static CommManager& GetInstance(s32 deviceLogicId);
    HcclCommInfoV2& GetCommInfoV2();
    void PrintChannelInfo();
    std::function<void()> GetPrintChannelInfoCallback();
    std::shared_ptr<Hccl::CcuDriverHandle> GetCcuDriver();
    void DeinitCcuDriver();
    s32 deviceLogicId{0};
    HcclResult SetCommAcceleratorV2(Hccl::HcclCommunicator* communicator, int32_t accelerator);

private:
    CommManager() = default;
    bool isCcuAvailable{true};

    std::shared_ptr<Hccl::CcuDriverHandle> ccuDriverHandle{nullptr};
    HcclCommInfoV2 commInfoV2{};
};

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus
HcclCommInfoV2& GetCommInfoV2(void);
HcclResult HcomDestroyV2(void);
HcclResult GetHcomRankListV2(u32 rankNum, const u32* rankIds, HcclGroupParamsV2& params, HcclComm globalComm);
HcclResult HcomCreateGroupImplV2(const std::string& group, u32 rankNum, const std::vector<u32>& rankIds);
HcclResult HcomDestroyGroupImplV2(const std::string& group);
HcclResult HcomGetWorldRankFromGroupRankV2(const char* group, u32 groupRank, u32* worldRank);
HcclResult HcomGetGroupRankFromWorldRankV2(u32 worldRank, const char* group, u32* groupRank);
HcclResult HcomGetRankSizeV2(const char* group, u32* rankSize);
HcclResult HcomGetCommV2(void** commV2);
HcclResult HcomGetGroupParamsV2(const char* group, void* groupParams, void** commV2);
HcclResult HcomInitByFileV2(const char* rankTablePath, const char* identify);
HcclResult HcomInitByStringV2(const char* rankTableM, const char* identify);
HcclResult CallSingletons();
HcclResult CcuResAllocAndCtxMgrInit(s32 deviceLogicId);
HcclResult HcomGetCcuTaskInfo(const std::string& group, void* tilingData, void* ccuTaskGroup);
#ifdef __cplusplus
}
#endif // __cplusplus
#endif /* HCCL_COMM_PUB_H */
