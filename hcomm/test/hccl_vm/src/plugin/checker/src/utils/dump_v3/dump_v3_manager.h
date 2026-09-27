/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_VM_DUMP_V3_MANAGER_H
#define HCCL_VM_DUMP_V3_MANAGER_H

#include <mutex>
#include <nlohmann_json/json.hpp>
#include <string>

#include "hccl_types.h"

namespace HcclSim {
class DumpV3Manager {
public:
    static DumpV3Manager& GetInstance()
    {
        static DumpV3Manager instance;
        return instance;
    }

    DumpV3Manager(const DumpV3Manager&) = delete;
    DumpV3Manager& operator=(const DumpV3Manager&) = delete;

    HcclResult Initialize(const std::string& dataId = "");
    void Reset();

    bool IsEnabled() const;
    std::string GetRootDir() const;

    HcclResult WriteMsgpack(const std::string& relativePath, const nlohmann::json& document) const;
    HcclResult WriteJson(const std::string& relativePath, const nlohmann::json& document) const;

private:
    DumpV3Manager() = default;

    HcclResult PrepareDirs() const;
    std::string GetOutputPath(const std::string& relativePath) const;

    mutable std::mutex m_mutex;
    std::string m_dataId;
    std::string m_pluginRootDir;
    std::string m_dataDir;
    std::string m_rootDir;
};
} // namespace HcclSim

#endif // HCCL_VM_DUMP_V3_MANAGER_H
