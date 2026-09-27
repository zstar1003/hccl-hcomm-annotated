/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "file_utils.h"

#include <cerrno>
#include <cstring>
#include <limits>
#include <sys/stat.h>
#include <unistd.h>

#include "sim_log.h"

namespace HcclSim {
std::string GetCurrentPath()
{
    char absPath[PATH_MAX];
    if (realpath(".", absPath) == nullptr) {
        HCCL_VM_ERROR("realpath failed for current path.");
        return "";
    }
    return std::string(absPath);
}

std::string GetRootPath(const std::string& anchorPath, int maxDepth)
{
    std::string currentSearchPath = GetCurrentPath();
    if (currentSearchPath.empty()) {
        return "";
    }

    for (int i = 0; i <= maxDepth; ++i) {
        const std::string checkTarget = (!anchorPath.empty() && anchorPath.front() == '/') ?
                                            currentSearchPath + anchorPath :
                                            JoinPath(currentSearchPath, anchorPath);
        if (DirExists(checkTarget)) {
            return currentSearchPath;
        }
        currentSearchPath = GetParentPath(currentSearchPath);
    }

    HCCL_VM_INFO("RootPath NOT found.");
    return "";
}

bool DirExists(const std::string& path)
{
    struct stat info;
    if (stat(path.c_str(), &info) != 0) {
        return false;
    }
    return S_ISDIR(info.st_mode);
}

bool FileExists(const std::string& path)
{
    struct stat info;
    if (stat(path.c_str(), &info) != 0) {
        return false;
    }
    return S_ISREG(info.st_mode);
}

HcclResult EnsureDirectory(const std::string& path)
{
    if (path.empty()) {
        return HcclResult::HCCL_E_PARA;
    }

    if (DirExists(path)) {
        return HcclResult::HCCL_SUCCESS;
    }

    std::string current;
    std::size_t start = 0;
    if (path[0] == '/') {
        current = "/";
        start = 1;
    }

    while (start <= path.size()) {
        const std::size_t pos = path.find('/', start);
        const std::string part = path.substr(start, pos == std::string::npos ? std::string::npos : pos - start);
        if (!part.empty()) {
            current = JoinPath(current, part);
            if (!DirExists(current)) {
                if (mkdir(current.c_str(), 0755) != 0 && errno != EEXIST) {
                    HCCL_VM_ERROR("mkdir failed for {}: {}", current, strerror(errno));
                    return HcclResult::HCCL_E_INTERNAL;
                }
            }
        }

        if (pos == std::string::npos) {
            break;
        }
        start = pos + 1;
    }

    return HcclResult::HCCL_SUCCESS;
}

std::string GetParentPath(const std::string& path)
{
    const std::size_t pos = path.find_last_of('/');
    if (pos == std::string::npos) {
        return ".";
    }
    if (pos == 0) {
        return "/";
    }
    return path.substr(0, pos);
}

std::string JoinPath(const std::string& left, const std::string& right)
{
    if (left.empty()) {
        return right;
    }
    if (right.empty()) {
        return left;
    }
    if (left.back() == '/') {
        return left + right;
    }
    return left + "/" + right;
}
} // namespace HcclSim
