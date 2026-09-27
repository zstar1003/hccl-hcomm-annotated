/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEM_TRANSPORT_MANAGER_H
#define MEM_TRANSPORT_MANAGER_H
#include <unordered_map>
#include <vector>
#include <string>
#include "base_mem_transport.h"
#include "virtual_topo.h"
#include "mem_transport_callback.h"
#include "mc2_type.h"

namespace Hccl {
class CommunicatorImpl;
class MemTransportManager {
public:
    explicit MemTransportManager(const CommunicatorImpl& communicator);

    virtual ~MemTransportManager();

    void BatchBuildOpbasedTransports(const vector<LinkData>& links);
    void BatchBuildOneSidedTransports(const vector<LinkData>& links);
    void BatchBuildOffloadTransports(const std::string& opTag, const vector<LinkData>& links);
    void BatchBuildUrmaDirectTransports(const vector<LinkData>& links);

    void UpdateOffloadTransports();

    BaseMemTransport* GetOpbasedTransport(const LinkData& linkData);
    BaseMemTransport* GetOneSidedTransport(const LinkData& linkData);
    BaseMemTransport* GetOffloadTransport(const std::string& opTag, const LinkData& linkData);
    BaseMemTransport* GetUrmaDirectTransport(const LinkData& linkData);

    bool IsAllTransportReady();

    void DumpNotReadyTransportsOpbased();
    void DumpNotReadyTransportsOffload(const std::string& opTag);
    void DumpNotReadyTransportsUrma();
    bool IsAllOpbasedTransportReady();
    bool IsAllOneSidedTransportReady();
    bool IsAllOffloadTransportReady(const std::string& opTag);

    std::vector<char> GetOpbasedPackedData();
    std::vector<char> GetOffloadPackedData(const std::string& opTag);
    std::vector<char> GetOneSidedPackedData();
    std::vector<char> GetPackedAllTransportData();

    std::vector<HcclAiRMAWQ> GetUrmaWqs();
    std::vector<HcclAiRMACQ> GetUrmaCqs();

    void BatchRecoverOpbasedTransports(const vector<LinkData>& links);
    void BatchRecoverOffloadTransports(const std::string& opTag, const vector<LinkData>& links);

    bool IsAllOpbasedTransportRecoveredReady();
    bool IsAllOffloadTransportRecoveredReady(const std::string& opTag);

    void Clear();

    HcclResult ClearOpTransport(const std::string& opTag);

private:
    std::vector<BaseLocalNotify*> GetNotifyVec(const LinkData& linkData) const;

    std::vector<LocalRmaBuffer*> GetBufferVec(const std::string& opTag, const LinkData& linkData, OpMode opMode) const;

    std::vector<RmaConnection*> GetConnVec(const std::string& opTag, const LinkData& linkData) const;

    BaseMemTransport* CreateOneSidedTransport(const LinkData& linkData);
    BaseMemTransport* CreateOpbasedMemTransport(const LinkData& linkData);
    BaseMemTransport* CreateOffloadMemTransport(const std::string& opTag, const LinkData& linkData);
    BaseMemTransport* CreateUrmaDirectTransport(const LinkData& linkData);

    void CreateOpbasedUbMemTransport(
        BaseMemTransport::CommonLocRes& locRes, BaseMemTransport::Attribution& attr, const LinkData& linkData,
        const Socket& socket);
    void CreateOffloadUbMemTransport(
        const string& opTag, BaseMemTransport::CommonLocRes& locRes, BaseMemTransport::Attribution& attr,
        const LinkData& linkData, const Socket& socket);
    void CreateOneSidedUbMemTransport(
        BaseMemTransport::CommonLocRes& locRes, BaseMemTransport::Attribution& attr, const LinkData& linkData,
        const Socket& socket);
    void CreateUrmaDirectTransport(
        BaseMemTransport::CommonLocRes& locRes, BaseMemTransport::Attribution& attr, const LinkData& linkData,
        const Socket& socket);

    const CommunicatorImpl* comm;

    using MemTransportMap = std::unordered_map<LinkData, std::unique_ptr<BaseMemTransport>, hash<Hccl::LinkData>>;

    MemTransportMap opTagOpbasedMap;
    MemTransportMap oneSidedMap;
    std::unordered_map<std::string, MemTransportMap> opTagOffloadMap;
    MemTransportMap urmaDirectMap_;

    std::unordered_map<LinkData, u32> newOpbasedTransports;                                  // 0：新增transports
    std::unordered_map<LinkData, u32> newOneSidedTransports;                                 // 0：新增transports
    std::unordered_map<std::string, std::unordered_map<LinkData, u32>> newOffloadTransports; // 0：新增transports

    BaseMemTransport* RecoverOpbasedMemTransport(const LinkData& linkData);
    BaseMemTransport* RecoverOffloadMemTransport(const std::string& opTag, const LinkData& linkData);
};
} // namespace Hccl
#endif
