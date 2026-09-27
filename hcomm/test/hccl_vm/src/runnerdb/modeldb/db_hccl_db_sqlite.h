/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DB_HCCL_DB_SQLITE_H
#define DB_HCCL_DB_SQLITE_H

#include <mutex>
#include <sqlite3.h>
#include <string>
#include <vector>

#include "db_hccl_db.h"

namespace HcclSim {
namespace DB {
    class HcclDBSqlite : public HcclDB {
    public:
        HcclDBSqlite();
        ~HcclDBSqlite() override;

        HcclVmResult Connect(const sim::DBConfig& config) override;

        HcclVmResult Close() override;

        bool IsConnected() const override;

        HcclVmResult Execute(const std::string& sql) override;

        HcclVmResult Execute(const std::string& sql, const std::vector<Value>& params) override;

        HcclVmResult Query(const std::string& sql, std::vector<std::vector<std::string>>& rows) override;

        HcclVmResult Query(
            const std::string& sql, const std::vector<Value>& params,
            std::vector<std::vector<std::string>>& rows) override;

        HcclVmResult
        QueryBlob(const std::string& sql, const std::vector<Value>& params, std::vector<uint8_t>& outBlob) override;

        HcclVmResult QueryBlobColumns(
            const std::string& sql, const std::vector<Value>& params,
            std::vector<std::vector<uint8_t>>& outBlobs) override;

        HcclVmResult QueryEx(const std::string& sql, const std::vector<Value>& params, ExRows& rows) override;

        int64_t LastInsertRowId() const override;

        int Changes() const override;

        std::string GetLastError() const override;

        sim::DbType GetDbType() const override;

        void* GetNativeHandle() override;

        HcclVmResult Backup(const std::string& destPath) override;

        HcclVmResult RunInTransaction(std::function<HcclVmResult()> fn) override;

    private:
        sqlite3* m_db = nullptr;
        std::string m_dbPath;
        std::string m_lastError;
        mutable std::recursive_mutex m_mutex;
    };
} // namespace DB
} // namespace HcclSim

#endif
