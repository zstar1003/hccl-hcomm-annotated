/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_RMA_CONN_MANAGER_H
#define HCCLV2_RMA_CONN_MANAGER_H

#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <memory>
#include "types.h"
#include "rma_connection.h"
#include "virtual_topo.h"
#include "socket.h"

namespace Hccl {

class CommunicatorImpl;

MAKE_ENUM(UboeStatus, READY, INIT, SOCKET_OK, SOCKET_TIMEOUT);

class RmaConnManager {
public:
    explicit RmaConnManager(const CommunicatorImpl& comm);
    virtual ~RmaConnManager();

    RmaConnection*
    Create(const std::string& tag, const LinkData& linkData, const HrtUbJfcMode jfcMode = HrtUbJfcMode::STARS_POLL);

    void BatchCreate(vector<LinkData>& links);

    RmaConnection* Get(const std::string& tag, const LinkData& linkData);

    std::vector<RmaConnection*> GetOpTagConns(const std::string& tag) const;

    void Release(const std::string& tag, const LinkData& linkData);

    void Destroy();

    void Clear();

private:
    void GetDeleteJettys(BatchDeleteJettyInfo& batchDeleteJettyInfo);
    void BatchDeleteJettys();
    unique_ptr<RmaConnection> CreateRdmaConn(Socket* socket, const std::string& tag, const LinkData& linkData) const;
    unique_ptr<RmaConnection> CreateUbConn(
        Socket* socket, const std::string& tag, const LinkData& linkData,
        const HrtUbJfcMode jfcMode = HrtUbJfcMode::STARS_POLL);
    bool isDestroyed{false};
    // tag -> LinkData -> RmaConnection
    std::unordered_map<std::string, std::unordered_map<LinkData, std::unique_ptr<RmaConnection>, hash<Hccl::LinkData>>>
        rmaConnectionMap;

    u32 localRank{0};
    const CommunicatorImpl* comm;

    IpAddress locAddr;
    IpAddress rmtAddr;
    UboeStatus uboeStatus{UboeStatus::INIT};
    MAKE_ENUM(UbStatus, INIT, SOCKET_OK, SEND_DATA, RECV_DATA, SEND_FIN, RECV_FIN, PROCESS_DATA, CONN_OK)
    UbStatus ubStatus{UbStatus::INIT};
    vector<char> sendData{};
    vector<char> recvData{};
    u32 exchangeDataSize{0}; // 交换的消息大小

    std::vector<RmaConnection*> GetAllConns() const;
    void RecreateAllConns();
    void BindRemoteRmaBuffers();

    bool IsSocketReady(Socket* socket, const LinkData& linkData);
    UboeStatus GetUboeSocketStatus(Socket* socket, const LinkData& linkData);
    void WaitUboeSocketReady(Socket* socket, const LinkData& linkData);
    void Ipv4Pack();
    void Ipv4UnPack(BinaryStream& binaryStream);
    void SendExchangeData(Socket* socket, const LinkData& linkData);
    void RecvExchangeData(Socket* socket, const LinkData& linkData);
    void RecvDataProcess(const LinkData& linkData);
};

} // namespace Hccl

#endif // HCCLV2_RMA_CONN_MANAGER_H
