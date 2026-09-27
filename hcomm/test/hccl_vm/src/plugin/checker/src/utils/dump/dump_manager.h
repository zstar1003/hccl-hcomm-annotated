/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_VM_DUMP_MANAGER_H
#define HCCL_VM_DUMP_MANAGER_H

#include <functional>
#include <iosfwd>
#include <mutex>
#include <nlohmann_json/json.hpp>
#include <string>

#include "hccl_types.h"

namespace HcclSim {
class DumpManager {
public:
    static DumpManager& GetInstance()
    {
        static DumpManager instance;
        return instance;
    }

    // 禁用拷贝构造和赋值
    DumpManager(const DumpManager&) = delete;
    DumpManager& operator=(const DumpManager&) = delete;

    HcclResult Initialize(const std::string& dataId = "");
    void Reset();

    bool IsEnabled() const;
    bool IsMemorySnapshotEnabled() const;
    std::string GetDumpRootDir() const;

    HcclResult Write(const std::string& relativePath, const nlohmann::json& document) const;
    HcclResult WriteJson(const std::string& relativePath, const nlohmann::json& document) const;
    HcclResult
    WriteMsgpackStream(const std::string& relativePath, const std::function<HcclResult(std::ostream&)>& writer) const;

private:
    DumpManager() = default;

    HcclResult PrepareDumpDirs() const;
    HcclResult WriteMsgpackFile(
        const std::string& dumpRootDir, const std::string& relativePath, const nlohmann::json& document) const;
    HcclResult WriteJsonFile(
        const std::string& dumpRootDir, const std::string& relativePath, const nlohmann::json& document) const;
    HcclResult WriteMsgpackStreamFile(
        const std::string& dumpRootDir, const std::string& relativePath,
        const std::function<HcclResult(std::ostream&)>& writer) const;

    mutable std::mutex m_mutex;
    std::string m_dataId;
    std::string m_pluginRootDir;
    std::string m_dataDir;
    std::string m_dumpRootDir;
};
} // namespace HcclSim

#endif // HCCL_VM_DUMP_MANAGER_H
