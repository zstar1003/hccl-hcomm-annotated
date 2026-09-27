/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DEV_RDMA_CONNECTION_V2_H
#define DEV_RDMA_CONNECTION_V2_H

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "hccp_common.h"
#include "enum_factory.h"
#include "hccl_common.h"
#include "mem_device_pub.h"
#include "orion_adapter_hccp.h"
#include "hcomm/hcomm_res_entity_defs.h"
#include "../host/exchange_rdma_conn_dto.h"

// Orion
#include "../../../../../legacy/ascend950/unified_platform/resource/socket/socket.h"
#include "hccp_nda.h"

namespace hcomm {
class DevRdmaConnectionV2 {
public:
    struct QpAttrDto {
        uint32_t qpn{UINT32_MAX};
        uint32_t psn{UINT32_MAX};
        uint32_t gid_idx{0};
        unsigned char gid[HCCP_GID_RAW_LEN];

        bool IsValid() const
        {
            if (qpn == UINT32_MAX || psn == UINT32_MAX) {
                return false;
            }
            return true;
        }
    };
    MAKE_ENUM(RdmaConnStatus, CLOSED, INIT, QP_CREATED, QP_MODIFIED, SOCKET_TIMEOUT)
    DevRdmaConnectionV2(Hccl::Socket* socket, RdmaHandle rdmaHandle, uint32_t cqAttrFlags);
    ~DevRdmaConnectionV2();

    HcclResult Init();
    HcclResult CreateQp();
    HcclResult GetExchangeDto(std::unique_ptr<Hccl::Serializable>& serial);
    HcclResult ParseRmtExchangeDto(const Hccl::Serializable& rmtDto);
    HcclResult ModifyQp();

    std::string Describe() const;
    Hccl::QpInfo& GetQpInfo() { return qpInfo_; }

    HcclResult BuildSqContext(SqContext* context);
    HcclResult BuildCqContext(CqContext* context);

    std::vector<char> GetUniqueId() const;

private:
    void GetNdaOps();
    HcclResult GetDirectFlag();
    HcclResult GetDmaMode();
    HcclResult DestroyQp();

    std::vector<char> GetSqUniqueId() const;
    std::vector<char> GetCqUniqueId() const;

    Hccl::Socket* socket_{nullptr};
    Hccl::RdmaHandle rdmaHandle_{nullptr};
    int32_t directFlag_{0};
    uint32_t dmaMode_{0};
    uint32_t cqAttrFlags_{0};
    NdaOps ndaOps_{};

    Hccl::QpHandle qpHandle_;
    NdaQpInfo ndaQpInfo_;
    Hccl::CqHandle cqHandle_;
    NdaCqInfo ndaCqInfo_;

    RdmaConnStatus rdmaConnStatus_{RdmaConnStatus::CLOSED};
    QpAttrDto rmtQpAttr_{};
    QpAttrDto locQpAttr_{};
    Hccl::QpInfo qpInfo_{};

    hccl::DeviceMem SqPiMem_;
    hccl::DeviceMem SqCiMem_;
    hccl::DeviceMem CqPiMem_;
    hccl::DeviceMem CqCiMem_;
};

} // namespace hcomm

#endif // DEV_RDMA_CONNECTION_V2_H
