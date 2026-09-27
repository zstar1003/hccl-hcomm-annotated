/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_AICPU_RMT_RMA_BUFFER_LITE_H_
#define HCCL_AICPU_RMT_RMA_BUFFER_LITE_H_

#include <set>
#include <vector>
#include "buffer.h"
#include "op_mode.h"
#include "coll_alg_info.h"
#include "data_buffer.h"
#include "mem_transport_lite_mgr.h"

namespace Hccl {

class RmtDataBufferMgr {
public:
    RmtDataBufferMgr(MemTransportLiteMgr* memTransportLiteMgr, CollAlgInfo* algInfo)
        : memTransportLiteMgr_(memTransportLiteMgr),
          algInfo_(algInfo) {};
    ~RmtDataBufferMgr() {};
    DataBuffer GetBuffer(const LinkData& linkData, BufferType bufferType);

private:
    MemTransportLiteMgr* memTransportLiteMgr_;
    CollAlgInfo* algInfo_;
};
} // namespace Hccl

#endif // HCCL_AICPU_RESOURCE_AI_CPU_MGRS_H_
