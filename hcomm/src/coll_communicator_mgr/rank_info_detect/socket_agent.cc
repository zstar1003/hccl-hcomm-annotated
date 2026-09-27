/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "socket_agent.h"
#include <climits>
#include "socket_exception.h"
#include "root_handle_v2.h"
#include "null_ptr_exception.h"

namespace Hccl {

void SocketAgent::SendMsg(const void* data, u64 dataLen)
{
    HCCL_INFO("[SocketAgent::%s] start, dataLen[%llu].", __func__, dataLen);

    // 发送数据长度
    CHK_PRT_THROW(
        !socket_->Send(&dataLen, sizeof(dataLen)), HCCL_ERROR("[SocketAgent::%s] Send data len failed", __func__),
        SocketException, "Unable to send data");

    // 发送数据
    CHK_PRT_THROW(
        !socket_->Send(data, dataLen), HCCL_ERROR("[SocketAgent::%s] Send data failed", __func__), SocketException,
        "Unable to send data");

    HCCL_INFO("[SocketAgent::%s] end.", __func__);
}

bool SocketAgent::RecvMsg(void* msg, u64& revMsgLen)
{
    HCCL_INFO("[SocketAgent::%s] start.", __func__);

    // 检查 msg 是否为 nullptr
    CHK_PRT_THROW(
        msg == nullptr, HCCL_ERROR("[RankInfoDetectService::%s] msg is nullptr", __func__), NullPtrException,
        "RecvMsg fail");

    // 先接收长度
    EXCEPTION_CATCH(socket_->Recv(&revMsgLen, sizeof(revMsgLen)), return false);
    CHK_PRT_RET(
        revMsgLen == 0 || revMsgLen > MAX_BUFFER_LEN,
        HCCL_ERROR("[SocketAgent::%s] Invalid length[%llu]", __func__, revMsgLen), false);

    // 再接收内容
    EXCEPTION_CATCH(socket_->Recv(msg, revMsgLen), return false);

    HCCL_INFO("[SocketAgent::%s] end, revMsgLen[%llu].", __func__, revMsgLen);
    return true;
}

} // namespace Hccl
