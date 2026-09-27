/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TYPICAL_PARAM_CHECK_H
#define TYPICAL_PARAM_CHECK_H
#include "interface_hccl.h"
namespace hccl {

inline HcclResult CheckDataType(const HcclDataType dataType)
{
    if ((dataType >= HCCL_DATA_TYPE_RESERVED) || (dataType < HCCL_DATA_TYPE_INT8)) {
        HCCL_ERROR(
            "[Check][DataType]errNo[0x%016llx] data type[%s] not supported", HCOM_ERROR_CODE(HCCL_E_NOT_SUPPORT),
            GetDataTypeEnumStr(dataType).c_str());
        return HCCL_E_NOT_SUPPORT;
    }
    return HCCL_SUCCESS;
}

inline HcclResult CheckCount(const u64 count)
{
    if (count > SYS_MAX_COUNT) {
        HCCL_ERROR(
            "[Check][Count]errNo[0x%016llx] count[%llu] is invalid(bigger than MAX count[%llu])",
            HCOM_ERROR_CODE(HCCL_E_PARA), count, SYS_MAX_COUNT);
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

inline HcclResult CheckSendRecvInfo(AscendSendRecvInfo* sendRecvInfo)
{
    CHK_PTR_NULL(sendRecvInfo);
    CHK_PTR_NULL(sendRecvInfo->localQPinfo);

    CHK_PTR_NULL(sendRecvInfo->localWindowMem);
    CHK_PTR_NULL(sendRecvInfo->remoteWindowMem);

    CHK_PTR_NULL(sendRecvInfo->localSyncMemPrepare);
    CHK_PTR_NULL(sendRecvInfo->localSyncMemDone);
    CHK_PTR_NULL(sendRecvInfo->localSyncMemAck);

    CHK_PTR_NULL(sendRecvInfo->remoteSyncMemPrepare);
    CHK_PTR_NULL(sendRecvInfo->remoteSyncMemDone);
    CHK_PTR_NULL(sendRecvInfo->remoteSyncMemAck);

    CHK_PTR_NULL(reinterpret_cast<void*>(sendRecvInfo->localWindowMem->addr));
    CHK_PTR_NULL(reinterpret_cast<void*>(sendRecvInfo->remoteWindowMem->addr));

    CHK_PTR_NULL(reinterpret_cast<void*>(sendRecvInfo->localSyncMemPrepare->addr));
    CHK_PTR_NULL(reinterpret_cast<void*>(sendRecvInfo->localSyncMemDone->addr));
    CHK_PTR_NULL(reinterpret_cast<void*>(sendRecvInfo->localSyncMemAck->addr));

    CHK_PTR_NULL(reinterpret_cast<void*>(sendRecvInfo->remoteSyncMemPrepare->addr));
    CHK_PTR_NULL(reinterpret_cast<void*>(sendRecvInfo->remoteSyncMemDone->addr));
    CHK_PTR_NULL(reinterpret_cast<void*>(sendRecvInfo->remoteSyncMemAck->addr));
    return HCCL_SUCCESS;
}

inline HcclResult CheckSendRecvLinkInfo(AscendSendRecvLinkInfo* sendRecvInfo)
{
    CHK_PTR_NULL(sendRecvInfo);
    CHK_PTR_NULL(sendRecvInfo->localQPinfo);

    CHK_PTR_NULL(sendRecvInfo->localSyncMemPrepare);
    CHK_PTR_NULL(sendRecvInfo->localSyncMemDone);
    CHK_PTR_NULL(sendRecvInfo->localSyncMemAck);

    CHK_PTR_NULL(sendRecvInfo->remoteSyncMemPrepare);
    CHK_PTR_NULL(sendRecvInfo->remoteSyncMemDone);
    CHK_PTR_NULL(sendRecvInfo->remoteSyncMemAck);
    return HCCL_SUCCESS;
}

inline HcclResult CheckSendLinkInfo(AscendSendLinkInfo* sendInfo)
{
    CHK_PTR_NULL(sendInfo);
    CHK_PTR_NULL(sendInfo->localQPinfo);
    CHK_PTR_NULL(sendInfo->localSyncMemAck);
    CHK_PTR_NULL(sendInfo->remoteNotifyValueMem);
    // 校验sendInfo->remoteNotifyValueMem长度
    const uint32_t NOTIFY_MEM_LEN = 4;
    CHK_PRT_RET(
        sendInfo->remoteNotifyValueMem->size != NOTIFY_MEM_LEN,
        HCCL_ERROR(
            "[CheckSendLinkInfo] remoteNotifyValueMem len expect:[%u], actual:[%llu].", NOTIFY_MEM_LEN,
            sendInfo->remoteNotifyValueMem->size),
        HCCL_E_PARA);
    return HCCL_SUCCESS;
}

inline HcclResult CheckParam(void* buf, uint64_t count, HcclDataType dataType, aclrtStream stream)
{
    CHK_PTR_NULL(buf);
    CHK_RET(CheckCount(count));
    CHK_RET(CheckDataType(dataType));
    CHK_PTR_NULL(stream);
    return HCCL_SUCCESS;
}
} // namespace hccl
// namespace hccl
#endif
