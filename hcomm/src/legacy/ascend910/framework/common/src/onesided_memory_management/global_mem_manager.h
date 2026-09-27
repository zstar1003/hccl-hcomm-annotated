/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <set>
#include <map>
#include <mutex>
#include <memory>
#include <unordered_set>

#include <hccl/hccl_comm.h>
#include <hccl/hccl_inner.h>
#include "hccl_mem_defs.h"
#include "global_mem_record.h"
#include "hccl_network_pub.h"
#include "hccl_socket_manager.h"

namespace hccl {

// 进程粒度的内存管理单例
class GlobalMemRegMgr {
public:
    static GlobalMemRegMgr& GetInstance(); // 获取单例
    ~GlobalMemRegMgr();

    inline bool CheckHandleIsValid(void* handle) const
    {
        return handle != nullptr && validHandlePtrSet.find(handle) != validHandlePtrSet.cend();
    }

    HcclResult Reg(const HcclMem* mem, void** memRecordHandle); // 登记一段内存
    HcclResult DeReg(void* memRecordHandle);                    // 删除一段内存记录
    HcclResult GetNetDevCtx(NicType nicType, const HcclIpAddress& ipAddr, u32 port, HcclNetDevCtx& netDevCtx);
    HcclResult InitNic();
    HcclResult DeInitNic();
    HcclResult Destroy();

private:
    HcclResult CheckOverlapAndInsert(GlobalMemRecord& memRecord, void** memRecordHandle);
    HcclResult CheckOneSidedBackupAndSetDevId(
        const HcclIpAddress& ipAddr, u32& backupDevPhyId, u32& backupDevLogicId,
        std::vector<HcclIpAddress>& localIpList, bool& isOneSidedTaskAndBackupInitA3) const;

    std::set<GlobalMemRecord> memRecordSet_{}; // 内存记录，按照type>addr>size排序
    std::unordered_set<void*> validHandlePtrSet{}; // 记录handle地址，用于校验用户传入的是否是handle的地址
    std::mutex lock_;                              // 锁保证多线程访问安全
    std::map<PortInfo, std::pair<NicType, HcclNetDevCtx>> netDevCtxMap_{};
    u32 devicePhyId_{INVALID_UINT};
    s32 deviceLogicId_{INVALID_INT};
    bool nicInited_{false};
    bool isInited_{false};
    std::unique_ptr<HcclSocketManager> socketManager_;
    std::mutex netDevCtxMtx_;
};

} // namespace hccl
