/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef URMA_DIRECT_TRANSPORT_H
#define URMA_DIRECT_TRANSPORT_H
#include "base_mem_transport.h"
#include "task_param.h"
#include "mc2_type.h"
#include "orion_adapter_hccp.h"
#include "dev_ub_connection.h"

namespace Hccl {

class UrmaDirectTransport : public BaseMemTransport {
public:
    UrmaDirectTransport(
        CommonLocRes& commonLocRes, Attribution& attr, const LinkData& linkData, const Socket& socket,
        RdmaHandle rdmaHandle1);
    UrmaDirectTransport(
        CommonLocRes& commonLocRes, Attribution& attr, const LinkData& linkData, const Socket& socket,
        RdmaHandle rdmaHandle1, std::function<void(u32 streamId, u32 taskId, const TaskParam& taskParam)> callback);

    TransportStatus GetStatus() override;

    HcclAiRMAWQ GetAiRMAWQ();
    HcclAiRMACQ GetAiRMACQ();

    std::string Describe() const override;

    RemoteUbRmaBuffer* GetRmtBuffer() const;

private:
    RdmaHandle rdmaHandle;

    MAKE_ENUM(UrmaStatus, INIT, SOCKET_OK, SEND_DATA, RECV_DATA, SEND_FIN, RECV_FIN, PROCESS_DATA, CONN_OK)
    UrmaStatus urmaStatus{UrmaStatus::INIT};

    using RemoteBufferVec = std::vector<std::unique_ptr<RemoteUbRmaBuffer>>;
    RemoteBufferVec rmtBufferVec; // 远端 buffer

    void SendExchangeData();
    void RecvExchangeData();

    void SendFinish();
    void RecvFinish();

    void BufferVecPack(BinaryStream& binaryStream);

    void RmtBufferVecUnpackProc(u32 locNum, BinaryStream& binaryStream, RemoteBufferVec& bufferVec);
    bool ConnVecUnpackProc(BinaryStream& binaryStream);

    std::vector<char> GetBufferUniqueId(RemoteBufferVec& bufferVec);

    bool IsResReady();
    bool IsConnsReady();
    bool RecvDataProcess();

    vector<char> sendData{};
    vector<char> recvData{};
    vector<char> sendFinishMsg{};
    vector<char> recvFinishMsg{};
};
} // namespace Hccl
#endif
