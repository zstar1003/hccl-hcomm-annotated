/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RA_RESOURCE_MANAGER_H
#define RA_RESOURCE_MANAGER_H
#include <mutex>
#include "dtype_common.h"
#include "hccl_common.h"
#include "adapter_hccp.h"
#include "interface_hccl.h"

namespace hccl {
constexpr u32 TIME_FROM_1900 = 1900;
class RdmaResourceManager {
public:
    static RdmaResourceManager& GetInstance();
    HcclResult Init();
    void SetDisableLiteThread(bool disable);
    HcclResult DeInit();
    HcclResult GetRdmaHandle(RdmaHandle& rdmaHandle);
    HcclResult GetCqeErrInfo(struct CqeErrInfo* infoList, u32* num);
    HcclResult GetCqeErrInfoByQpn(u32 qpn, struct HcclErrCqeInfo* errCqeList, u32* num);
    HcclResult GetNotifyMrInfo(struct MrInfoT& mrInfo);
    HcclResult GetResvMemPoolIdByType(u32 type, u32& poolId);

private:
    RdmaResourceManager();
    ~RdmaResourceManager();
    RdmaResourceManager(RdmaResourceManager const&) = delete;
    RdmaResourceManager(RdmaResourceManager&&) = delete;
    RdmaResourceManager& operator=(RdmaResourceManager const&) = delete;
    RdmaResourceManager& operator=(RdmaResourceManager&&) = delete;
    HcclResult InitResvMemInfo();

private:
    s32 deviceLogicId_{};
    u32 devicePhyId_{};
    NICDeployment nicDeploy_ = NICDeployment::NIC_DEPLOYMENT_DEVICE;
    HcclIpAddress ipAddr_{};
    u32 port_{MAX_VALUE_U32};
    RdmaHandle rdmaHandle_{};
    std::mutex cqeErrMapMutex_;
    std::map<u32, std::queue<struct CqeErrInfo>> cqeErrPerQP_{}; // key:qp, value:cqe_err_info queue
    MrHandle mrHandle_{};
    u64 notifyBaseVa_ = 0;
    u64 notifyTotalSize_ = 0;
    struct MrInfoT notifyMrInfo_ {};
    std::unordered_map<u32, u32> resvMemInfo_{}; // type:pollId
};

} // namespace hccl

#endif
