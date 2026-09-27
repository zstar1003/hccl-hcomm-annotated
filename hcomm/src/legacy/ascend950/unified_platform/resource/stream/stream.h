/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_STREAM_H
#define HCCLV2_STREAM_H

#include "orion_adapter_rts.h"
#include "stream_lite.h"
namespace Hccl {

class Stream {
public:
    explicit Stream(aclrtStream ptr, bool isMaster = true);

    explicit Stream(bool deviceUsed = false, bool isMaster = true);

    Stream(const Stream& stream, bool isMaster = true) = delete;

    Stream& operator=(const Stream& stream) = delete;

    ~Stream();

    bool operator==(const Stream& rhs) const { return ptr == rhs.ptr; }

    bool operator!=(const Stream& rhs) const { return !(rhs == *this); }

    void SetStmMode(u64 stmMode);

    aclrtStream GetPtr() const;

    u32 GetId() const;

    u32 GetSqId() const;

    bool IsMaster() const;

    bool IsSelfOwned() const;

    u64 GetMode() const;

    std::vector<char> GetUniqueId() const;

    std::string Describe() const;

private:
    static constexpr int32_t HCCL_STREAM_PRIORITY_LOW = 5;
    static constexpr int32_t HCCL_STREAM_PRIORITY_HIGH = 5;
    static constexpr int32_t STREAM_MODE_STOP_ON_FAILURE = 1; // 配置流失败模式为遇错即停

    aclrtStream ptr;
    u32 id{0};
    bool selfOwned{false};
    u64 mode{0};
    bool devUsed{false};
    u32 sqId{0};
    u32 cqId{0};
    u32 devPhyId{0};
    bool isMaster_{true};

    void InitDevPhyId();
};

} // namespace Hccl
#endif
