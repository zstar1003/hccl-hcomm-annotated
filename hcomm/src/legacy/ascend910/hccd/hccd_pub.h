/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCD_PUB_H
#define HCCD_PUB_H

#include "hccl_comm_pub.h"
using HccdInfo = struct HccdInfoTag {
    std::shared_ptr<hccl::HccdComm> pCommhccd;
    hccl::HcclCommParams params;
    hccl::RankTable_t rankTable;
    bool cloudFlag;

    HccdInfoTag() : pCommhccd(nullptr), cloudFlag(false) {}
};

HcclResult HccdGenerateCommId(hccl::HcclCommParams& params);

#endif
