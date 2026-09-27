/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_SRC_DLHNSFUNCTION_H
#define HCCL_SRC_DLHNSFUNCTION_H

#include <functional>
#include <atomic>
#include <mutex>
#include <hccl/hccl_types.h>
#include "hccl/base.h"
#include "private_types.h"
#include "hccl_common.h"

struct WrExpRsp {
    unsigned int wqe_index;
    unsigned long db_info;
};

struct IbvPostSendExtResp {
    unsigned int wqe_index;
    unsigned long db_info;
};

struct IbvPostSendExtAddt {
    uint8_t reduce_op;
    uint8_t reduce_type;
};

namespace hccl {
class DlHnsFunction {
public:
    virtual ~DlHnsFunction();
    static DlHnsFunction& GetInstance();
    HcclResult DlHnsFunctionInit();

    std::function<int(
        struct ibv_qp* qp, struct ibv_send_wr* wr, struct ibv_send_wr** badwr, struct IbvPostSendExtAddt* extAttr,
        struct IbvPostSendExtResp* extResp)>
        dlHnsIbvExtPostSend;
    std::function<int(struct ibv_qp* qp, struct ibv_send_wr* wr, struct ibv_send_wr** badWr, struct WrExpRsp* expRsp)>
        dlHnsIbvExpPostSend;

private:
    void* handle_;
    std::mutex handleMutex_;
    DlHnsFunction(const DlHnsFunction&);
    DlHnsFunction& operator=(const DlHnsFunction&);
    DlHnsFunction();
    HcclResult DlHnsFunctionRoceInit();
    HcclResult DlHnsFunctionSoInit();
};
} // namespace hccl

#endif
