/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef REGISTER_OP_BIN_INFO_H
#define REGISTER_OP_BIN_INFO_H

#include <vector>
#include <tuple>
#include <string>
#include "graph/ascend_string.h"

namespace ops {
using OpInfo = std::vector<std::tuple<ge::AscendString, ge::AscendString, const uint8_t*, const uint8_t*>>;

class OpBinInfo {
public:
    OpBinInfo(const std::string& opType, const OpInfo& opInfo);
    ~OpBinInfo();
    uint32_t Generate(ge::AscendString* opLibPath, const std::string& targetPath);
    static bool Check(const std::string& path);

private:
    std::string opType_;
    std::string basePath_;
    const OpInfo& opInfo_;
};

} // namespace ops
#endif
