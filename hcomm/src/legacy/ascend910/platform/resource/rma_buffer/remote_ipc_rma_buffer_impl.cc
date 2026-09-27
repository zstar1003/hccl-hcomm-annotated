/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "remote_ipc_rma_buffer_impl.h"
#include "hccl_network.h"

namespace hccl {
RemoteIpcRmaBufferImpl::RemoteIpcRmaBufferImpl(const HcclNetDevCtx netDevCtx)
    : RmaBuffer(netDevCtx, nullptr, 0, RmaMemType::TYPE_NUM, RmaType::IPC_RMA),
      netDevCtx(netDevCtx)
{}

RemoteIpcRmaBufferImpl::~RemoteIpcRmaBufferImpl() {}

HcclResult RemoteIpcRmaBufferImpl::Deserialize(const std::string& msg)
{
    std::istringstream iss(msg);
    iss.read(reinterpret_cast<char_t*>(&memName.ipcName), sizeof(memName.ipcName));
    iss.read(reinterpret_cast<char_t*>(&memOffset), sizeof(memOffset));
    HCCL_DEBUG("[RemoteIpcRmaBufferImpl][Deserialize]ipcName[%s], memOffset[%lu]", memName.ipcName, memOffset);
    return HCCL_SUCCESS;
}

HcclResult RemoteIpcRmaBufferImpl::Open()
{
    if (memType == RmaMemType::HOST) {
        HCCL_ERROR("[RemoteIpcRmaBufferImpl][Open]remote memType[%d] not support.", memType);
        return HCCL_E_PARA;
    }

    s32 deviceLogicId = 0;
    if (netDevCtx != nullptr) {
        deviceLogicId = (static_cast<NetDevContext*>(netDevCtx))->GetLogicId();
    } else {
        CHK_RET(hrtGetDevice(&deviceLogicId));
    }
    bool firstOpened = false;
    HcclResult ret = MemNameRepository::GetInstance(deviceLogicId)
                         ->OpenIpcMem(&devAddr, size, memName.ipcName, HCCL_IPC_MEM_NAME_LEN, memOffset, firstOpened);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[RemoteIpcRmaBufferImpl][Open]errNo[0x%016llx] Open ipc mem failed. memName[%s], offset[%llu]",
            HCCL_ERROR_CODE(ret), memName.ipcName, memOffset),
        ret);
    return HCCL_SUCCESS;
}

HcclResult RemoteIpcRmaBufferImpl::Close()
{
    s32 deviceLogicId = 0;
    if (netDevCtx != nullptr) {
        deviceLogicId = (static_cast<NetDevContext*>(netDevCtx))->GetLogicId();
    } else {
        CHK_RET(hrtGetDevice(&deviceLogicId));
    }
    MemNameRepository::GetInstance(deviceLogicId)->CloseIpcMem(static_cast<const u8*>(memName.ipcName));
    HCCL_DEBUG("[RemoteIpcRmaBufferImpl][Close]memName[%s]", memName.ipcName);
    return HCCL_SUCCESS;
}
} // namespace hccl
