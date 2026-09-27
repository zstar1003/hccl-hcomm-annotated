/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CCU_RES_PACK_H
#define CCU_RES_PACK_H

#include "ccu_res_repo.h"
#include "ccu_types.h"
#include "ccu_res_desc.h"
#include "ccu_device_pub.h"
#include "ccu_res_specs.h"
#include <memory>
#include <unordered_map>
#include <shared_mutex>

namespace hcomm {
class CcuResPack {
public:
    explicit CcuResPack() {};
    ~CcuResPack();
    // 基于实例类型初始化
    CcuResult InitByInsType(const CcuInstanceType insType);
    // 基于资源描述符数组初始化（资源数量由 resDesc 驱动，与 Init 的 insType 路径区分）
    CcuResult InitByResDescs(const CcuResDesc* descs[], uint32_t descNum);
    CcuResult Reset();

    CcuResRepository& GetCcuResRepo();

    // 申请一个空闲级联计数器块并标记为已占用, 以调用方(CcuInstanceMgr)传入的全局唯一 handle
    // 为键登记到 handleMap_; handle 由上层统一分配(本类不再自增), 无空闲块/dieId 越界/handle 为 0
    // 时返回错误码。非幂等：重复调用会取走下一个块。
    CcuResult AcquireCascCntBlock(uint8_t dieId, HcommCcuCascCntHandle handle);
    // 返回 handleMap_ 的快照拷贝(读锁保护), 避免外部无锁遍历/拷贝期间被 AcquireCascCntBlock
    // 的写入(emplace 触发 rehash)破坏; CntXnBlock 为 POD 值类型, 拷贝成本可控。
    // 调用方仅应使用返回的临时快照, 不持有内部 map 引用。
    std::unordered_map<HcommCcuCascCntHandle, CntXnBlock> GetCascCntBlocks()
    {
        std::shared_lock<std::shared_timed_mutex> lock(handleMapMutex_);
        return handleMap_;
    }
    CcuResult GetCascCntBlock(HcommCcuCascCntHandle handle, CntXnBlock& cascCntBlock);

private:
    CcuResPack(const CcuResPack& that) = delete;
    CcuResPack& operator=(const CcuResPack& that) = delete;
    CcuResPack(CcuResPack&& that) = delete;
    CcuResPack& operator=(CcuResPack&& that) = delete;
    CcuResult AllocCascCntBlock(const CcuResDesc* descs[], uint32_t descNum);
    void CcuReleaseRes();

    int32_t userDevId_{0};
    CcuResHandle resHandle_{nullptr};
    CcuResRepository resRepo_{};
    std::mutex cascCntBlockMutex_;
    std::array<std::unordered_map<std::unique_ptr<CntXnBlock>, bool>, CCU_MAX_IODIE_NUM> cascCntBlocks_;
    mutable std::shared_timed_mutex handleMapMutex_;
    std::unordered_map<HcommCcuCascCntHandle, CntXnBlock> handleMap_;
};

} // namespace hcomm

#endif // CCU_RES_PACK_H
