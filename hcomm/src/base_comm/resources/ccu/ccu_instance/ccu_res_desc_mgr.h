/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_CCU_RES_DESC_MGR_H
#define HCOMM_CCU_RES_DESC_MGR_H

#include <memory>
#include <shared_mutex>
#include <unordered_map>

#include "ccu_res_defs.h"
#include "ccu_res_desc.h"

namespace hcomm {

class CcuResDescMgr {
public:
    CcuResDescMgr() = default;
    ~CcuResDescMgr() = default;
    CcuResDescMgr(const CcuResDescMgr&) = delete;
    CcuResDescMgr& operator=(const CcuResDescMgr&) = delete;

    CcuResult Create(uint32_t dieId, HcommCcuResDescHandle& handle);
    const CcuResDesc* Get(HcommCcuResDescHandle handle) const;
    CcuResult Destroy(HcommCcuResDescHandle handle);
    CcuResult SetResNum(HcommCcuResDescHandle handle, ResType resType, uint32_t resNum);
    CcuResult QueryResNum(HcommCcuResDescHandle handle, ResType resType, uint32_t& resNum) const;
    CcuResult QueryDieId(HcommCcuResDescHandle handle, uint32_t& dieId) const;
    CcuResult Deinit();

    // 在锁内遍历资源类型 → 查询容量 → 收集区间 → 扫描最大连续 → 写入 desc
    CcuResult QueryRemainRes(HcommCcuResDescHandle handle, int32_t userDevId) const;

private:
    using DescMap = std::unordered_map<HcommCcuResDescHandle, std::unique_ptr<CcuResDesc>>;
    using ConstDescIterator = DescMap::const_iterator;

    CcuResult FindDesc(HcommCcuResDescHandle handle, const char* funcName, ConstDescIterator& it) const;

    mutable std::shared_timed_mutex descMapMutex_;
    HcommCcuResDescHandle nextHandle_{0};
    DescMap descMap_{};
};

} // namespace hcomm

#endif // HCOMM_CCU_RES_DESC_MGR_H
