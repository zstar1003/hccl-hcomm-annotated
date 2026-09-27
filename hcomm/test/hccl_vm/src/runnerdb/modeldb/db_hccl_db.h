/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DB_HCCL_DB_H
#define DB_HCCL_DB_H

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <variant>
#include <vector>

#include "sim_common_defs.h"
#include "sim_op_db_types.h"

namespace HcclSim {
namespace DB {
    struct BlobData {
        const void* data;
        size_t size;
    };

    using Value = std::variant<int64_t, std::string, BlobData>;

    using Field = std::variant<std::nullptr_t, std::string, std::vector<uint8_t>>;
    using ExRow = std::vector<Field>;
    using ExRows = std::vector<ExRow>;

    class HcclDB {
    public:
        virtual ~HcclDB() = default;

        virtual HcclVmResult Connect(const sim::DBConfig& config) = 0;

        virtual HcclVmResult Close() = 0;

        virtual bool IsConnected() const = 0;

        virtual HcclVmResult Execute(const std::string& sql) = 0;

        virtual HcclVmResult Execute(const std::string& sql, const std::vector<Value>& params) = 0;

        virtual HcclVmResult Query(const std::string& sql, std::vector<std::vector<std::string>>& rows) = 0;

        virtual HcclVmResult
        Query(const std::string& sql, const std::vector<Value>& params, std::vector<std::vector<std::string>>& rows)
            = 0;

        virtual HcclVmResult
        QueryBlob(const std::string& sql, const std::vector<Value>& params, std::vector<uint8_t>& outBlob)
            = 0;

        virtual HcclVmResult QueryBlobColumns(
            const std::string& sql, const std::vector<Value>& params, std::vector<std::vector<uint8_t>>& outBlobs)
            = 0;

        virtual HcclVmResult QueryEx(const std::string& sql, const std::vector<Value>& params, ExRows& rows) = 0;

        virtual int64_t LastInsertRowId() const = 0;

        virtual int Changes() const = 0;

        virtual std::string GetLastError() const = 0;

        virtual sim::DbType GetDbType() const = 0;

        virtual void* GetNativeHandle() = 0;

        virtual HcclVmResult Backup(const std::string& destPath) = 0;

        virtual HcclVmResult RunInTransaction(std::function<HcclVmResult()> fn) = 0;
    };
} // namespace DB
} // namespace HcclSim

#endif
