/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "rma_conn_lite.h"
#include "ipc_conn_lite.h"
#include "rdma_conn_lite.h"
#include "ub_conn_lite.h"
#include "binary_stream.h"
#include "exception_util.h"
#include "not_support_exception.h"
#include "log.h"
namespace Hccl {
RmaConnLite::RmaConnLite(const UbJettyLiteId& id, const UbJettyLiteAttr& attr, const Eid& rmtEid)
    : dieId_(id.dieId_),
      funcId_(id.funcId_),
      jettyId_(id.jettyId_),
      dbAddr_(attr.dbAddr_),
      sqVa_(attr.sqVa_),
      sqDepth_(attr.sqDepth_),
      jfcPollMode_(attr.jfcPollMode_),
      tpn_(attr.tpn_),
      rmtEid_(rmtEid)
{
    HCCL_INFO(
        "RmaConnLite id.dieId %u, id.funcId %u, id.jettyId %u, attr.dbaddr %llu, attr.sqVa %llu, attr.sqDepth %u",
        id.dieId_, id.funcId_, id.jettyId_, attr.dbAddr_, attr.sqVa_, attr.sqDepth_);
}

RmaConnLite::RmaConnLite(const u64 qpVa) : qpVa_(qpVa) {}

std::unique_ptr<RmaConnLite> RmaConnLite::Create(std::vector<char>& uniqueId)
{
    BinaryStream binaryStream(uniqueId);
    u32 type;
    binaryStream >> type;
    auto connType = static_cast<RmaConnLiteType::Value>(type);
    if (connType == RmaConnLiteType::RDMA) {
        return std::make_unique<RdmaConnLite>(0); // 待RdmaConnLite构造实现
    }
    if (connType == RmaConnLiteType::P2P) {
        return std::make_unique<IpcConnLite>(); // 待IpcConnLite构造实现
    }
    if (connType == RmaConnLiteType::UB) {
        return std::make_unique<UbConnLite>(uniqueId);
    }
    THROW<NotSupportException>(StringFormat("Unsupported rma connection type: %d", connType));
    return nullptr;
}

std::string RmaConnLite::Describe() { return StringFormat("RmaConnLite"); }

UbJettyLiteId RmaConnLite::GetUbJettyLiteId() const { return UbJettyLiteId(dieId_, funcId_, jettyId_); }

UbJettyLiteAttr RmaConnLite::GetUbJettyLiteAttr() const
{
    return UbJettyLiteAttr(dbAddr_, sqVa_, sqDepth_, tpn_, dwqeCacheLocked_, jfcPollMode_);
}

Eid RmaConnLite::GetRmtEid() const { return rmtEid_; }

Eid RmaConnLite::GetLocEid() const { return locEid_; }

u32 RmaConnLite::GetQpVa() const { return qpVa_; }

u64 RmaConnLite::GetJettyHandle() const { return jettyHandle_; }

u32 RmaConnLite::GetJettyId() const { return jettyId_; }

u32 RmaConnLite::GetTpn() const { return tpn_; }

} // namespace Hccl
