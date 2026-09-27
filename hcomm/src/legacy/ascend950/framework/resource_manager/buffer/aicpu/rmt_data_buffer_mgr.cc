/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "rmt_data_buffer_mgr.h"
#include "null_ptr_exception.h"
#include "log.h"
#include "data_buffer.h"
#include "exception_util.h"
namespace Hccl {

DataBuffer RmtDataBufferMgr::GetBuffer(const LinkData& linkData, BufferType bufferType)
{
    if (algInfo_ == nullptr) {
        THROW<NullPtrException>("[%s] ERROR algInfo is null", __func__);
    }
    OpMode opMode = algInfo_->GetOpMode();
    HCCL_INFO(
        "[%s] linkData[%s] bufferType[%s] opMode[%s]", __func__, linkData.Describe().c_str(),
        bufferType.Describe().c_str(), opMode.Describe().c_str());
    MemTransportLite* memTransportLite = nullptr;
    if (opMode == OpMode::OPBASE) {
        CHECK_NULLPTR(memTransportLiteMgr_, "[RmtDataBufferMgr::GetBuffer] memTransportLiteMgr_ is nullptr!");
        memTransportLite = memTransportLiteMgr_->GetOpbase(linkData);
        if (memTransportLite == nullptr) {
            THROW<NullPtrException>("[%s] ERROR GetOpbase is null", __func__);
        }
        Buffer buffer = memTransportLite->GetRmtBuffer(bufferType);
        HCCL_INFO("[%s] buffer.GetAddr()[%lu] buffer.GetSize()[%llu]", __func__, buffer.GetAddr(), buffer.GetSize());
        return DataBuffer(buffer.GetAddr(), buffer.GetSize());
    } else {
        THROW<NullPtrException>("[%s] ERROR no support offload mode", __func__);
    }
    HCCL_WARNING("[%s] linkData[%p] getDatabuffer failed", __func__, linkData);
    std::size_t bufferSize = 0;
    return DataBuffer(bufferSize);
}

} // namespace Hccl
