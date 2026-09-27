/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// 本文件仅用于 legacy ascend910 历史兼容，不再演进，不承接新特性；新增能力请落 base_comm/ 或
// coll_communicator_mgr/ 正式目录。

#ifndef LEGACY_OP_HCOM_INFO_H
#define LEGACY_OP_HCOM_INFO_H

#include <mutex>
#include <unordered_map>
#include <map>
#include <memory>
#include "comm.h"
#include "hccl_comm_pub.h"
#include "topoinfo_struct.h"
#include "topoinfo_detect.h"

struct HcclInfoTag {
    HcclCommPtr pComm;
    hccl::HcclCommParams params;
    hccl::RankTable_t rankTable;
    bool cloudFlag = false; // cloudFlag为0即实验室场景,cloudFlag为1则为云场景
    bool isUsed;
    std::mutex opGroupMapMutex;
    std::unordered_map<std::string, std::shared_ptr<hccl::hcclComm>> opGroup2CommMap;
    std::map<std::string, std::shared_ptr<hccl::TopoInfoDetect>> hcclCommTopoInfoDetectServer;
    std::map<std::string, std::shared_ptr<hccl::TopoInfoDetect>> hcclCommTopoInfoDetectAgent;
    HcclInfoTag() : isUsed(false) {}

    ~HcclInfoTag()
    {
        pComm = nullptr;
        opGroup2CommMap.clear();
        hcclCommTopoInfoDetectServer.clear();
        hcclCommTopoInfoDetectAgent.clear();
    }
};

using HcclOpInfoCtx = HcclInfoTag;

#endif // LEGACY_OP_HCOM_INFO_H
