/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCD_COMM_H
#define HCCD_COMM_H

#include <vector>
#include <memory>
#include <map>
#include <mutex>
#include "hccl/base.h"
#include "hccl_common.h"
#include "mem_device_pub.h"
#include "mem_host_pub.h"
#include "adapter_pub.h"
#include "topoinfo_struct.h"
#include "memory_alloc_ring.h"
#include "hccl_operator.h"
#include "stream_pub.h"

namespace hccl {

/* * 默认的rank_table, ranklist为空数组;  后续HCCL可以用于判断是否走新分支 */
extern RankTable_t g_hcclDefaultRankTable;

class HccdImplPml;
class HccdComm {
public:
    explicit HccdComm(std::string identifier = "");
    ~HccdComm();

    HcclResult init(HcclCommParams& params, const RankTable_t& rankTable = g_hcclDefaultRankTable);
    HcclResult RegisterMemory(void* buffer, uint64_t size);
    HcclResult UnregisterMemory(void* buffer);
    HcclResult Isend(
        void* buffer, s32 count, HcclDataType dataType, u32 peerRank, s32 tag, HcclRequest& request,
        u32 userRequire) const;
    HcclResult Improbe(u32 peerRank, s32 tag, s32& flag, HcclMessage& msgHandle, HcclStatus& status) const;
    HcclResult Imrecv(void* buffer, s32 count, HcclDataType dataType, HcclMessage msg, HcclRequest& request) const;
    HcclResult HcclTest(HcclRequest hcclRequest, s32& flag, HcclStatus& compState) const;
    HcclResult GetUserRank(u32& userRank);
    const std::string& GetIdentifier();
    HcclResult GetRankSize(u32& rankSize);
    static HcclResult GetUniqueId(HcclRootInfo* uniqueId);

protected:
    /* * 禁止用户对API类的实体做拷贝构造或拷贝赋值的操作，内部有指针成员变量 */
    HccdComm(const HccdComm&) = delete;
    HccdComm& operator=(const HccdComm&) = delete;

private:
    HcclResult InitImpl();
    std::unique_ptr<HccdImplPml> impl_;
    const std::string identifier_;
};
} // namespace hccl
#endif
