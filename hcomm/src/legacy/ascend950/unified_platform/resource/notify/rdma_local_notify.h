/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_RDMA_LOCAL_NOTIFY_H
#define HCCLV2_RDMA_LOCAL_NOTIFY_H

#include "orion_adapter_hccp.h"

#include "local_notify_v2.h"

namespace Hccl {

class RdmaLocalNotify : public BaseLocalNotify {
public:
    explicit RdmaLocalNotify(RdmaHandle rdmaHandle, bool devUsed = false);

    ~RdmaLocalNotify() override;

    RdmaLocalNotify(const RdmaLocalNotify& that) = delete;

    RdmaLocalNotify& operator=(const RdmaLocalNotify& that) = delete;

    void Wait(const Stream& stream, u32 timeout) const override;

    void Post(const Stream& stream) const override;

    string Describe() const override;

    std::unique_ptr<Serializable> GetExchangeDto() override;

private:
    RdmaHandle rdmaHandle;
    u64 addr{0};
    u32 size{0};
    u8 key[RDMA_MEM_KEY_MAX_LEN]{0};
    u32 lkey{0};
    u32 rkey{0};
    MrHandle mrHandle{nullptr};
};

} // namespace Hccl
#endif // !HCCLV2_RDMA_LOCAL_NOTIFY_H
