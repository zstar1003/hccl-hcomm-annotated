/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "shared_jetty_connection_adapter.h"
#include "dev_ub_connection.h"
#include "log.h"
#include "adapter_rts_common.h"

namespace hcomm {

// 适配层当前实现：转发到 Hccl::DevUbConnection 的新增共享 jetty 方法。
// 后续演进：DevUbConnection 自 legacy/ 迁入 base_comm 后，本文件可直接操作 connection，
//                     届时移除对 dev_ub_connection.h 的 #include，base_comm 完全不依赖 legacy 实现细节。
// 注意：endpointTag 当前仅 builtin 路径传入，实际类型为 Endpoint*。
//       若将来 plugin 路径复用此函数，需将 acquireRemoteCb/publishRemoteCb 的构建提升到调用方，
//       避免适配层直接强转 void* 为 Endpoint* 造成类型不匹配。
HcclResult SetSharedJettyFieldsToConn(
    void* rawConnection, const JettyContext::Ctx& ctx, void* endpointTag, std::function<void(void*)> releaseCb)
{
    if (rawConnection == nullptr || endpointTag == nullptr) {
        return HCCL_E_PARA;
    }
    auto* endpoint = static_cast<Endpoint*>(endpointTag);
    auto acquireRemoteCb = [endpoint](
                               const uint8_t* remoteQpKey, uint32_t keySize, bool& needImport,
                               Hccl::TargetJettyHandle& handle, void*& handlePtr, uint32_t& tpn) -> HcclResult {
        JettyContext* jc = dynamic_cast<JettyContext*>(endpoint->GetCommQueueContext());
        CHK_PTR_NULL(jc);
        uint64_t cachedHandle = 0;
        HcclResult ret = jc->AcquireSharedRemoteJetty(remoteQpKey, keySize, needImport, cachedHandle, handlePtr, tpn);
        handle = static_cast<Hccl::TargetJettyHandle>(cachedHandle);
        return ret;
    };
    auto publishRemoteCb = [endpoint](
                               const uint8_t* remoteQpKey, uint32_t keySize, Hccl::TargetJettyHandle handle,
                               void* handlePtr, uint32_t tpn) -> HcclResult {
        JettyContext* jc = dynamic_cast<JettyContext*>(endpoint->GetCommQueueContext());
        CHK_PTR_NULL(jc);
        return jc->PublishSharedRemoteJetty(remoteQpKey, keySize, static_cast<uint64_t>(handle), handlePtr, tpn);
    };
    auto* conn = static_cast<Hccl::DevUbConnection*>(rawConnection);
    return conn->SetSharedJettyFields(
        ctx.handle, ctx.handlePtr, ctx.jettyId, ctx.sqBuffVa, ctx.dbAddr, ctx.localQpKey, ctx.keySize, ctx.sqDepth,
        static_cast<Hccl::JfcHandle>(ctx.jfcHandle), ctx.cqInfo, ctx.localPsn, endpointTag, std::move(releaseCb),
        std::move(acquireRemoteCb), std::move(publishRemoteCb));
}

HcclResult ExtractJettyInfoFromConn(void* rawConnection, JettyContext::Ctx& ctx)
{
    if (rawConnection == nullptr) {
        return HCCL_E_PARA;
    }
    auto* conn = static_cast<Hccl::DevUbConnection*>(rawConnection);
    Hccl::DevUbConnection::JettyInfo info;
    CHK_RET(conn->GetJettyInfo(info));
    ctx.handle = info.handle;
    ctx.handlePtr = info.handlePtr;
    ctx.jettyId = info.jettyId;
    ctx.sqBuffVa = info.sqBuffVa;
    ctx.dbAddr = info.dbAddr;
    ctx.keySize = info.keySize;
    ctx.sqDepth = info.sqDepth;
    ctx.rdmaHandle = info.rdmaHandle;
    ctx.jfcHandle = info.jfcHandle;
    ctx.cqInfo = info.cqInfo;
    ctx.localPsn = info.localPsn;
    // 严格按实际 keySize 拷贝，避免拷贝超出实际 keySize 的无效尾部
    if (info.keySize > Hccl::HRT_UB_QP_KEY_MAX_LEN) {
        HCCL_ERROR("[%s] invalid keySize[%u], max[%u].", __func__, info.keySize, Hccl::HRT_UB_QP_KEY_MAX_LEN);
        return HCCL_E_PARA;
    }
    CHK_SAFETY_FUNC_RET(memcpy_s(ctx.localQpKey, Hccl::HRT_UB_QP_KEY_MAX_LEN, info.localQpKey, info.keySize));
    return HCCL_SUCCESS;
}

HcclResult DetachConnJetty(void* rawConnection)
{
    if (rawConnection == nullptr) {
        return HCCL_E_PARA;
    }
    auto* conn = static_cast<Hccl::DevUbConnection*>(rawConnection);
    conn->DetachJetty();
    return HCCL_SUCCESS;
}

} // namespace hcomm
