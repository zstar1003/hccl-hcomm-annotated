/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_H2DTLV_H
#define HCCL_H2DTLV_H

#include <memory>

#include "hccl/base.h"
#include "hccl_common.h"
#include "hccl_comm_pub.h"

namespace hccl {

constexpr unsigned int H2D_TLVBUFFERSIZE = 0;

class hcclH2dTlv {
public:
    hcclH2dTlv() = default;
    ~hcclH2dTlv() = default;
    static hcclH2dTlv& GetInstance();
    HcclResult InitHccpChannel(u32 devicePhyId);
    void DeinitHccpChannel();
    bool GetH2dTlvInitFlag();
    unsigned int GetH2dTlvBufferSize() const;
    void* GetH2dTlvHandle();

private:
    std::atomic<bool> hcclH2dTlvInitFlag_ = {false};
    void* hcclH2dTlvHandle_ = nullptr;
    unsigned int hcclH2dTlvBuffsize_ = 0;
};

} // namespace hccl
#endif /* HCCL_H2DTLV_H */
