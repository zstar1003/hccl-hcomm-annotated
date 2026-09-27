/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FILE_UTILS_H
#define FILE_UTILS_H

#include <string>

#include "hccl_types.h"

namespace HcclSim {
std::string GetCurrentPath();
std::string GetRootPath(const std::string& anchorPath, int maxDepth = 3);
bool DirExists(const std::string& path);
bool FileExists(const std::string& path);
HcclResult EnsureDirectory(const std::string& path);
std::string GetParentPath(const std::string& path);
std::string JoinPath(const std::string& left, const std::string& right);
} // namespace HcclSim

#endif
