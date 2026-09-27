/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "dev_rdma_connection.h"
#include "not_support_exception.h"
#include "orion_adapter_rts.h"
namespace Hccl {

DevRdmaConnection::DevRdmaConnection(Socket* socket, RdmaHandle rdmaHandle, OpMode opMode)
    : RmaConnection(socket, RmaConnType::RDMA)
{
    int qpMode = 0;
    DevType devType = HrtGetDeviceType();
    if (devType == DevType::DEV_TYPE_910A2 || devType == DevType::DEV_TYPE_910A3) {
        qpMode = (opMode == OpMode::OPBASE) ? OPBASE_QP_MODE_EXT : OFFLINE_QP_MODE_EXT;
    } else if (devType == DevType::DEV_TYPE_910A) {
        qpMode = (opMode == OpMode::OPBASE) ? OPBASE_QP_MODE : OFFLINE_QP_MODE;
    } else {
        HCCL_ERROR(
            "Cannot support this device type! "
            "errNo[0x%016llx], device type[%s]",
            HCCL_ERROR_CODE(HcclResult::HCCL_E_NOT_SUPPORT), DevTypeToString(devType).c_str());
        throw NotSupportException(DevTypeToString(devType));
    }
    qpHandle = HrtRaQpCreate(rdmaHandle, QP_FLAG_RC, qpMode);
}

void DevRdmaConnection::Connect() { GetStatus(); }

RmaConnStatus DevRdmaConnection::GetStatus()
{
    if (status == RmaConnStatus::INIT) {
        if (rdmaConnStatus == RdmaConnStatus::INIT) {
            if (socket->GetStatus() == SocketStatus::OK) {
                HrtRaQpConnectAsync(qpHandle, socket->GetFdHandle());
                rdmaConnStatus = RdmaConnStatus::CONNECTING;
                CheckQpStatus(qpHandle);
            } else if (socket->GetStatus() == SocketStatus::TIMEOUT) {
                rdmaConnStatus = RdmaConnStatus::SOCKET_TIMEOUT;
                status = RmaConnStatus::CONN_INVALID;
            }
        } else if (rdmaConnStatus == RdmaConnStatus::CONNECTING) {
            CheckQpStatus(qpHandle);
        }
    }
    return status;
}

void DevRdmaConnection::CheckQpStatus(QpHandle handle)
{
    if (HrtGetRaQpStatus(handle) == 1) { // 为1时，qp 建链成功
        status = RmaConnStatus::READY;
    }
}

QpHandle DevRdmaConnection::GetHandle() { return qpHandle; }

DevRdmaConnection::~DevRdmaConnection() { HrtRaQpDestroy(qpHandle); }

unique_ptr<BaseTask> DevRdmaConnection::PrepareWrite(
    const MemoryBuffer& remoteMemBuf, const MemoryBuffer& localMemBuf, [[maybe_unused]] const SqeConfig& config)
{
    VerifySizeIsEqual(remoteMemBuf, localMemBuf, "DevRdmaConnection::PrepareWrite");

    if (localMemBuf.size == 0) {
        return nullptr;
    }
    // RDMA_WRITE: op = 0
    HRaSendWr wr(localMemBuf.addr, localMemBuf.size, remoteMemBuf.addr, 0, RA_SEND_SIGNALED);
    RaSendWrResp resp = HrtRaSendOneWr(qpHandle, wr);

    return make_unique<TaskRdmaSend>(resp.dbIndex, static_cast<u64>(resp.dbInfo));
}

string DevRdmaConnection::Describe() const
{
    return StringFormat("DevRdmaConnection[status=%s]", status.Describe().c_str());
}

} // namespace Hccl
