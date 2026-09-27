/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_LOCAL_NOTIFY_H
#define HCCLV2_LOCAL_NOTIFY_H

#include "rts_notify_v2.h"
#include "rma_type.h"
#include "serializable.h"

namespace Hccl {

class BaseLocalNotify {
public:
    BaseLocalNotify(const RmaType& type, bool devUsed) : type(type)
    {
        HCCL_INFO("[BaseLocalNotify] init type[%d] devUsed[%d]", type, devUsed);
        notify = std::make_unique<RtsNotify>(devUsed);
    }

    virtual ~BaseLocalNotify() = default;

    virtual string Describe() const = 0;

    virtual void Wait(const Stream& stream, u32 timeout) const = 0;

    virtual void Post(const Stream& stream) const = 0;

    const RmaType& GetType() const { return type; }

    std::vector<char> GetUniqueId() const { return notify->GetUniqueId(); }

    virtual std::unique_ptr<Serializable> GetExchangeDto()
    {
        MACRO_THROW(NotSupportException, StringFormat("not support."));
    }

    RtsNotify* GetNotify() const { return notify.get(); }

protected:
    RmaType type;

private:
    std::unique_ptr<RtsNotify> notify;
};
} // namespace Hccl
#endif // !HCCLV2_LOCAL_NOTIFY_H
