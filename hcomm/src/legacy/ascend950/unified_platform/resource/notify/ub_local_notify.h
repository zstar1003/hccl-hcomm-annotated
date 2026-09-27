/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_UB_LOCAL_NOTIFY_H
#define HCCLV2_UB_LOCAL_NOTIFY_H

#include "enum_factory.h"
#include "orion_adapter_hccp.h"
#include "buffer_key.h"

#include "local_notify_v2.h"

namespace Hccl {

MAKE_ENUM(UbNotifyStatus, INIT, READY, RELEASED);

class UbLocalNotify : public BaseLocalNotify {
public:
    explicit UbLocalNotify(RdmaHandle rdmaHandle, bool devUsed = false);

    UbLocalNotify(const UbLocalNotify& that) = delete;

    UbLocalNotify& operator=(const UbLocalNotify& that) = delete;

    string Describe() const override;

    void Wait(const Stream& stream, u32 timeout) const override;

    void Post(const Stream& stream) const override;

    std::unique_ptr<Serializable> GetExchangeDto() override; // 先实现UB Notify的exchange dto，IPC/RDMA待补充

    ~UbLocalNotify() override;

private:
    RdmaHandle rdmaHandle{nullptr};
    u32 tokenValue{0};
    u64 addr{0};
    u32 size{0};
    u8 key[HRT_UB_MEM_KEY_MAX_LEN]{0};
    u32 tokenId{0};
    TokenIdHandle tokenIdHandle_{0};
    u64 memHandle{0};
    u32 keySize{0};

    HrtRaUbLocalMemRegOutParam reqReg;
    void* lmemHandle{nullptr};
    BufferKey<uintptr_t, u64> bufKey_{0, 0};

    void ReleaseResource();
};

} // namespace Hccl
#endif // !HCCLV2_UB_LOCAL_NOTIFY_H
