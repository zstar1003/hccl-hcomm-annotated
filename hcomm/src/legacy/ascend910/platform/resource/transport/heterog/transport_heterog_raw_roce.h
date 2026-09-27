/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TRANSPORT_HETEROG_PD_ROCE_PUB_H
#define TRANSPORT_HETEROG_PD_ROCE_PUB_H

#include "transport_heterog_roce_pub.h"

namespace hccl {
class TransportHeterogRawRoce : public TransportHeterogRoce {
public:
    explicit TransportHeterogRawRoce(
        const std::string& transTag, HcclIpAddress& selfIp, HcclIpAddress& peerIp, u32 peerPort, u32 selfPort,
        const TransportResourceInfo& transportResourceInfo);
    ~TransportHeterogRawRoce() override;

    HcclResult Init() override;
    HcclResult Init(SocketInfoT& socketInfo, RdmaHandle rdmaHandle, MrHandle mrHandle) override;
    HcclResult ImrecvScatter(
        void* buf[], int count[], int bufCount, HcclDataType datatype, HcclMessageInfo& msg,
        HcclRequestInfo*& request) override;

protected:
    HcclResult CreateCqAndQp() override;
    HcclResult DestroyCqAndQp() override;
    HcclResult PreQpConnect() override;

private:
    HcclResult EnterStateProcess(ConnState nextState) override;
    HcclResult LoopStateProcess() override;
    HcclResult PrepareModifyInfo(struct QpAttr& qpAttr, struct TypicalQp& typicalQpInfo);
    HcclResult GetQpAttr(QpHandle& qpHandle, struct QpAttr* attr, bool& completed);
    HcclResult
    TypicalQpModify(QpHandle& qpHandle, struct TypicalQp* localQpInfo, struct TypicalQp* remoteQpInfo, bool& completed);

    struct ibv_send_wr dataReadWrScatter_ {}; // scatter数据读取的wr模板
    struct ibv_send_wr dataAckWrScatter_ {};  // scatter ACK发送的wr模板

    struct QpAttr localTagQpAttr_ {};
    struct QpAttr localDataQpAttr_ {};
    struct QpAttr remoteTagQpAttr_ {};
    struct QpAttr remoteDataQpAttr_ {};
    struct TypicalQp localTagModifyInfo_ {};
    struct TypicalQp localDataModifyInfo_ {};
    struct TypicalQp remoteTagModifyInfo_ {};
    struct TypicalQp remoteDataModifyInfo_ {};
};
} // namespace hccl
#endif
