/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COMM_TOPO_DESC_H
#define COMM_TOPO_DESC_H

#include <functional>
#include <securec.h>
#include <mutex>
#include <dlfcn.h>
#include <unordered_map>
#include "hccl/base.h"
#include "hccl_rank_graph.h"

namespace hccl {
class CommTopoDesc {
public:
    virtual ~CommTopoDesc();
    static CommTopoDesc& GetInstance();
    void SaveRankSize(std::string& str, uint32_t rankSize);
    void SaveL0TopoType(std::string& str, CommTopo topoType);
    HcclResult GetRankSize(std::string& str, uint32_t* rankSize);
    HcclResult GetL0TopoType(std::string& str, CommTopo* topoType);

protected:
private:
    CommTopoDesc(const CommTopoDesc&) = delete;
    CommTopoDesc& operator=(const CommTopoDesc&) = delete;
    CommTopoDesc();

    std::mutex lock_;
    std::unordered_map<std::string, uint32_t> rankSizeMap_;
    std::unordered_map<std::string, CommTopo> l0TopoTypeMap_;
};
} // namespace hccl
#endif
