/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aiv_ins_preprocessor.h"
#include "aiv_ins.h"
#include "env_config_v2.h"
#include "local_rma_buf_manager.h"
#include "hccl_log_keywords.h"

namespace Hccl {

void AivInsPreprocessor::SetProtocol(uint8_t protocol) { protocol_ = protocol; }

uint8_t AivInsPreprocessor::GetProtocol() const { return protocol_; }

void AivInsPreprocessor::Preprocess(std::shared_ptr<InsQueue>& insQueue) const
{
    HCCL_INFO("[AivInsPreprocessor::%s] insQueue Preprocess start.", __func__);

    // 对每主queue中每个ins进行预处理
    for (auto ins = insQueue->Iter(); ins.HasNext(); ++ins) {
        if (ins->GetType() != InstructionType::AIV_INS) {
            continue;
        }
        InsPreprocess(ins);
    }

    HCCL_INFO("[AivInsPreprocessor::%s] insQueue Preprocess end.", __func__);
}

void AivInsPreprocessor::InsPreprocess(InsIterator& insIter) const
{
    HCCL_INFO("[AivInsPreprocessor::%s] start.", __func__);

    const AivInstruction& aivIns = dynamic_cast<const AivInstruction&>(*insIter);

    auto links = aivIns.GetLinks();

    if (protocol_ == 0) { // ubmemory
        BatchBuildTransports(links);
    } else if (protocol_ == 1) { // urma
        BatchBuildUrmaTransports(links);
    } else {
        THROW<InvalidParamsException>(StringFormat("protocol[%u] not supported", protocol_));
    }

    HCCL_INFO("[AivInsPreprocessor::%s] end.", __func__);
}

void AivInsPreprocessor::BatchBuildTransports(const vector<LinkData>& links) const
{
    HCCL_INFO("[AivInsPreprocessor::%s] start.", __func__);

    // 创建MemTransport并进行异步建链、交换
    comm->GetUbMemoryTransportMgr()->BatchCreateTransport(links);

    comm->GetUbMemoryTransportMgr()->TransportsConnect();

    HCCL_INFO("[AivInsPreprocessor::%s] end.", __func__);
}

void AivInsPreprocessor::BatchBuildUrmaTransports(const vector<LinkData>& links) const
{
    HCCL_RUN_INFO("[AivInsPreprocessor::%s] start.", __func__);

    std::string opTag = comm->GetCurrentCollOperator()->opTag;

    auto& localRmaBufManager = comm->GetLocalRmaBufManager();
    const string aivUrmaBufferTag = GetAivUrmaBufferTag(comm->GetId());
    for (size_t i = 0; i < links.size(); ++i) {
        const auto& link = links[i];
        auto localBuffer = localRmaBufManager.Reg(
            aivUrmaBufferTag, BufferType::SCRATCH, comm->GetCclBuffer(), link.GetLocalPort(), link.GetLinkProtocol());
        CHECK_NULLPTR(
            localBuffer, StringFormat(
                             "[AivInsPreprocessor::%s] fake local buffer is nullptr, linkIdx[%zu], port[%u]", __func__,
                             i, link.GetLocalPort().GetId()));
    }

    // 创建RmaConnectiuon
    RmaConnManager& connManager = comm->GetRmaConnManager();
    for (auto& link : links) {
        auto conn = connManager.Create(opTag, link, HrtUbJfcMode::USER_CTL);
        CHECK_NULLPTR(conn, "[AivInsPreprocessor::BatchBuildUrmaTransports] conn is nullptr!");
    }
    HCCL_INFO("[AivInsPreprocessor::%s] end creating rma connection", __func__);

    // 创建UrmaDirectTransport并进行异步建链、交换
    auto transportMgr = comm->GetMemTransportManager();
    CHECK_NULLPTR(transportMgr, "[AivInsPreprocessor::BatchBuildUrmaTransports] transportMgr is nullptr!");
    transportMgr->BatchBuildUrmaDirectTransports(links);

    auto timeout = std::chrono::seconds(EnvConfig::GetInstance().GetSocketConfig().GetLinkTimeOut());

    HcclUs startTime = std::chrono::steady_clock::now();
    bool isReady = false;
    while (!isReady) {
        isReady = transportMgr->IsAllTransportReady();
        if (isReady) {
            break;
        }
        if ((std::chrono::steady_clock::now() - startTime) >= timeout) {
            transportMgr->DumpNotReadyTransportsUrma();
            RPT_INPUT_ERR(
                true, "EI0006", std::vector<std::string>({"reason"}),
                std::vector<std::string>({"Aiv urma wait transports ready timeout."}));
            HCCL_ERROR(
                "[%s][%s] wait socket establish timeout, Aiv urma wait transports ready timeout, commId[%s].",
                LOG_KEYWORDS_INIT_CHANNEL.c_str(), LOG_KEYWORDS_TIMEOUT.c_str(), comm->GetId().c_str());
            THROW<InternalException>("Aiv WaitTransportReady timeout, commId[%s].", comm->GetId().c_str());
            break;
        }
    }

    HCCL_INFO("[AivInsPreprocessor::%s] end.", __func__);
}

std::vector<HcclAiRMAWQ> AivInsPreprocessor::GetWqs() const
{
    HCCL_INFO("[AivInsPreprocessor::%s] start.", __func__);
    if (protocol_ != 1) {
        THROW<InvalidParamsException>(StringFormat("can not get wq info when protocol is [%u]", protocol_));
    }
    auto memTransportMgr = comm->GetMemTransportManager();
    CHECK_NULLPTR(memTransportMgr, "[AivInsPreprocessor::GetWqs] memTransportMgr is nullptr!");
    return memTransportMgr->GetUrmaWqs();
}

std::vector<HcclAiRMACQ> AivInsPreprocessor::GetCqs() const
{
    HCCL_INFO("[AivInsPreprocessor::%s] start.", __func__);
    if (protocol_ != 1) {
        THROW<InvalidParamsException>(StringFormat("can not get cq info when protocol is [%u]", protocol_));
    }
    auto memTransportMgr = comm->GetMemTransportManager();
    CHECK_NULLPTR(memTransportMgr, "[AivInsPreprocessor::GetCqs] memTransportMgr is nullptr!");
    return memTransportMgr->GetUrmaCqs();
}

AivInsPreprocessor::~AivInsPreprocessor() {}

} // namespace Hccl
