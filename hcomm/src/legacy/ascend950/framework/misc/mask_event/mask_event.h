/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_MASK_EVENT_H
#define HCCLV2_MASK_EVENT_H

#include "orion_adapter_rts.h"
#include "stream.h"

namespace Hccl {

class MaskEvent {
public:
    explicit MaskEvent();

    virtual ~MaskEvent();

    MaskEvent(const MaskEvent& maskEvent) = delete;
    MaskEvent& operator=(const MaskEvent& maskEvent) = delete;

    inline RtEvent_t GetPtr() const { return eventPtr; }

    void Record(const Stream& stream) const;
    HrtEventStatus QueryStatus() const;

private:
    RtEvent_t eventPtr;
};

} // namespace Hccl
#endif
