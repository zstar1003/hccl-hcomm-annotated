/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "db_hccl_db_sqlite.h"
#include <cstdint>
#include "sim_common_macro.h"
#include "sim_log.h"

namespace HcclSim {
namespace DB {
    HcclDBSqlite::HcclDBSqlite() = default;

    HcclDBSqlite::~HcclDBSqlite() { Close(); }

    HcclVmResult HcclDBSqlite::Connect(const sim::DBConfig& config)
    {
        std::lock_guard lock(m_mutex);

        if (m_db != nullptr) {
            sqlite3_close(m_db);
            m_db = nullptr;
        }

        int ret = sqlite3_open(config.dbPath.c_str(), &m_db);
        if (ret != SQLITE_OK) {
            m_lastError = sqlite3_errmsg(m_db);
            HCCL_VM_ERROR("Failed: {}, path: {}", m_lastError, config.dbPath);
            sqlite3_close(m_db);
            m_db = nullptr;
            return HCCL_SIM_E_OPEN_FILE_FAILURE;
        }

        m_dbPath = config.dbPath;

        sqlite3_exec(m_db, "PRAGMA journal_mode=WAL;", nullptr, nullptr, nullptr);
        sqlite3_exec(m_db, "PRAGMA busy_timeout=10000;", nullptr, nullptr, nullptr);
        sqlite3_exec(m_db, "PRAGMA synchronous=NORMAL;", nullptr, nullptr, nullptr);
        sqlite3_exec(m_db, "PRAGMA foreign_keys=ON;", nullptr, nullptr, nullptr);
        sqlite3_exec(m_db, "PRAGMA cache_size = -10000;", nullptr, nullptr, nullptr);
        sqlite3_exec(m_db, "PRAGMA temp_store = MEMORY;", nullptr, nullptr, nullptr);
        sqlite3_exec(m_db, "PRAGMA journal_size_limit = 67108864;", nullptr, nullptr, nullptr);
        sqlite3_exec(m_db, "PRAGMA wal_autocheckpoint = 200;", nullptr, nullptr, nullptr);
        sqlite3_exec(m_db, "PRAGMA mmap_size = 268435456;", nullptr, nullptr, nullptr);

        HCCL_VM_INFO("Connected: {}", config.dbPath);
        return HCCL_SIM_SUCCESS;
    }

    HcclVmResult HcclDBSqlite::Close()
    {
        std::lock_guard lock(m_mutex);

        if (m_db != nullptr) {
            sqlite3_close(m_db);
            m_db = nullptr;
        }

        return HCCL_SIM_SUCCESS;
    }

    bool HcclDBSqlite::IsConnected() const { return m_db != nullptr; }

    HcclVmResult HcclDBSqlite::Execute(const std::string& sql, const std::vector<Value>& params)
    {
        std::lock_guard lock(m_mutex);

        HCCLVM_CHK_PTR(m_db);

        sqlite3_stmt* stmt = nullptr;
        int ret = sqlite3_prepare_v2(m_db, sql.c_str(), -1, &stmt, nullptr);
        if (ret != SQLITE_OK) {
            m_lastError = sqlite3_errmsg(m_db);
            HCCL_VM_ERROR("Prepare failed: {}, sql: {}", m_lastError, sql);
            return HCCL_SIM_E_INTERNAL;
        }

        for (size_t i = 0; i < params.size(); ++i) {
            const Value& val = params[i];
            if (std::holds_alternative<int64_t>(val)) {
                sqlite3_bind_int64(stmt, i + 1, std::get<int64_t>(val));
            } else if (std::holds_alternative<std::string>(val)) {
                const std::string& str = std::get<std::string>(val);
                sqlite3_bind_text(stmt, i + 1, str.c_str(), str.size(), SQLITE_STATIC);
            } else if (std::holds_alternative<BlobData>(val)) {
                BlobData blob = std::get<BlobData>(val);
                sqlite3_bind_blob(stmt, i + 1, blob.data, blob.size, SQLITE_STATIC);
            }
        }

        ret = sqlite3_step(stmt);
        sqlite3_finalize(stmt);

        if (ret != SQLITE_DONE) {
            m_lastError = sqlite3_errmsg(m_db);
            HCCL_VM_ERROR("Step failed: {}, sql:{}", m_lastError, sql);
            return HCCL_SIM_E_INTERNAL;
        }

        return HCCL_SIM_SUCCESS;
    }

    HcclVmResult HcclDBSqlite::Query(
        const std::string& sql, const std::vector<Value>& params, std::vector<std::vector<std::string>>& rows)
    {
        std::lock_guard lock(m_mutex);

        HCCLVM_CHK_PTR(m_db);

        sqlite3_stmt* stmt = nullptr;
        int ret = sqlite3_prepare_v2(m_db, sql.c_str(), -1, &stmt, nullptr);
        if (ret != SQLITE_OK) {
            m_lastError = sqlite3_errmsg(m_db);
            HCCL_VM_ERROR("Prepare failed: {}", m_lastError);
            return HCCL_SIM_E_INTERNAL;
        }

        for (size_t i = 0; i < params.size(); ++i) {
            const Value& val = params[i];
            if (std::holds_alternative<int64_t>(val)) {
                sqlite3_bind_int64(stmt, i + 1, std::get<int64_t>(val));
            } else if (std::holds_alternative<std::string>(val)) {
                const std::string& str = std::get<std::string>(val);
                sqlite3_bind_text(stmt, i + 1, str.c_str(), str.size(), SQLITE_STATIC);
            } else if (std::holds_alternative<BlobData>(val)) {
                BlobData blob = std::get<BlobData>(val);
                sqlite3_bind_blob(stmt, i + 1, blob.data, blob.size, SQLITE_STATIC);
            }
        }

        int colCount = sqlite3_column_count(stmt);

        while ((ret = sqlite3_step(stmt)) == SQLITE_ROW) {
            std::vector<std::string> row;
            for (int i = 0; i < colCount; ++i) {
                const char* text = reinterpret_cast<const char*>(sqlite3_column_text(stmt, i));
                row.push_back(text ? text : "");
            }
            rows.push_back(std::move(row));
        }

        sqlite3_finalize(stmt);

        if (ret != SQLITE_DONE) {
            m_lastError = sqlite3_errmsg(m_db);
            HCCL_VM_ERROR("Step failed: {}", m_lastError);
            return HCCL_SIM_E_INTERNAL;
        }

        return HCCL_SIM_SUCCESS;
    }

    HcclVmResult
    HcclDBSqlite::QueryBlob(const std::string& sql, const std::vector<Value>& params, std::vector<uint8_t>& outBlob)
    {
        std::lock_guard lock(m_mutex);

        HCCLVM_CHK_PTR(m_db);

        sqlite3_stmt* stmt = nullptr;
        int ret = sqlite3_prepare_v2(m_db, sql.c_str(), -1, &stmt, nullptr);
        if (ret != SQLITE_OK) {
            m_lastError = sqlite3_errmsg(m_db);
            HCCL_VM_ERROR("Prepare failed: {}", m_lastError);
            return HCCL_SIM_E_INTERNAL;
        }

        for (size_t i = 0; i < params.size(); ++i) {
            const Value& val = params[i];
            if (std::holds_alternative<int64_t>(val)) {
                sqlite3_bind_int64(stmt, i + 1, std::get<int64_t>(val));
            } else if (std::holds_alternative<std::string>(val)) {
                const std::string& str = std::get<std::string>(val);
                sqlite3_bind_text(stmt, i + 1, str.c_str(), str.size(), SQLITE_STATIC);
            } else if (std::holds_alternative<BlobData>(val)) {
                BlobData blob = std::get<BlobData>(val);
                sqlite3_bind_blob(stmt, i + 1, blob.data, blob.size, SQLITE_STATIC);
            }
        }

        ret = sqlite3_step(stmt);
        if (ret == SQLITE_ROW) {
            const void* blob = sqlite3_column_blob(stmt, 0);
            int blobSize = sqlite3_column_bytes(stmt, 0);
            if (blob && blobSize > 0) {
                outBlob.assign(static_cast<const uint8_t*>(blob), static_cast<const uint8_t*>(blob) + blobSize);
            }
        }
        sqlite3_finalize(stmt);

        if (ret != SQLITE_ROW && ret != SQLITE_DONE) {
            m_lastError = sqlite3_errmsg(m_db);
            HCCL_VM_ERROR("Step failed: {}", m_lastError);
            return HCCL_SIM_E_INTERNAL;
        }

        return HCCL_SIM_SUCCESS;
    }

    HcclVmResult HcclDBSqlite::QueryBlobColumns(
        const std::string& sql, const std::vector<Value>& params, std::vector<std::vector<uint8_t>>& outBlobs)
    {
        std::lock_guard lock(m_mutex);

        HCCLVM_CHK_PTR(m_db);

        sqlite3_stmt* stmt = nullptr;
        int ret = sqlite3_prepare_v2(m_db, sql.c_str(), -1, &stmt, nullptr);
        if (ret != SQLITE_OK) {
            m_lastError = sqlite3_errmsg(m_db);
            HCCL_VM_ERROR("Prepare failed: {}", m_lastError);
            return HCCL_SIM_E_INTERNAL;
        }

        for (size_t i = 0; i < params.size(); ++i) {
            const Value& val = params[i];
            if (std::holds_alternative<int64_t>(val)) {
                sqlite3_bind_int64(stmt, i + 1, std::get<int64_t>(val));
            } else if (std::holds_alternative<std::string>(val)) {
                const std::string& str = std::get<std::string>(val);
                sqlite3_bind_text(stmt, i + 1, str.c_str(), str.size(), SQLITE_STATIC);
            } else if (std::holds_alternative<BlobData>(val)) {
                BlobData blob = std::get<BlobData>(val);
                sqlite3_bind_blob(stmt, i + 1, blob.data, blob.size, SQLITE_STATIC);
            }
        }

        ret = sqlite3_step(stmt);
        if (ret == SQLITE_ROW) {
            int colCount = sqlite3_column_count(stmt);
            outBlobs.resize(colCount);
            for (int i = 0; i < colCount; ++i) {
                const void* blob = sqlite3_column_blob(stmt, i);
                int blobSize = sqlite3_column_bytes(stmt, i);
                if (blob && blobSize > 0) {
                    outBlobs[i].assign(static_cast<const uint8_t*>(blob), static_cast<const uint8_t*>(blob) + blobSize);
                }
            }
        }
        sqlite3_finalize(stmt);

        if (ret != SQLITE_ROW && ret != SQLITE_DONE) {
            m_lastError = sqlite3_errmsg(m_db);
            HCCL_VM_ERROR("Step failed: {}", m_lastError);
            return HCCL_SIM_E_INTERNAL;
        }

        return HCCL_SIM_SUCCESS;
    }

    HcclVmResult HcclDBSqlite::QueryEx(const std::string& sql, const std::vector<Value>& params, ExRows& rows)
    {
        std::lock_guard lock(m_mutex);

        HCCLVM_CHK_PTR(m_db);

        sqlite3_stmt* stmt = nullptr;
        int ret = sqlite3_prepare_v2(m_db, sql.c_str(), -1, &stmt, nullptr);
        if (ret != SQLITE_OK) {
            m_lastError = sqlite3_errmsg(m_db);
            HCCL_VM_ERROR("Prepare failed: {}", m_lastError);
            return HCCL_SIM_E_INTERNAL;
        }

        for (size_t i = 0; i < params.size(); ++i) {
            const Value& val = params[i];
            if (std::holds_alternative<int64_t>(val)) {
                sqlite3_bind_int64(stmt, i + 1, std::get<int64_t>(val));
            } else if (std::holds_alternative<std::string>(val)) {
                const std::string& str = std::get<std::string>(val);
                sqlite3_bind_text(stmt, i + 1, str.c_str(), str.size(), SQLITE_STATIC);
            } else if (std::holds_alternative<BlobData>(val)) {
                BlobData blob = std::get<BlobData>(val);
                sqlite3_bind_blob(stmt, i + 1, blob.data, blob.size, SQLITE_STATIC);
            }
        }

        int colCount = sqlite3_column_count(stmt);

        while ((ret = sqlite3_step(stmt)) == SQLITE_ROW) {
            ExRow row;
            row.reserve(colCount);
            for (int i = 0; i < colCount; ++i) {
                int colType = sqlite3_column_type(stmt, i);
                if (colType == SQLITE_NULL) {
                    row.emplace_back(nullptr);
                } else if (colType == SQLITE_BLOB) {
                    const void* data = sqlite3_column_blob(stmt, i);
                    int size = sqlite3_column_bytes(stmt, i);
                    if (data && size > 0) {
                        row.emplace_back(std::vector<uint8_t>(
                            static_cast<const uint8_t*>(data), static_cast<const uint8_t*>(data) + size));
                    } else {
                        row.emplace_back(std::vector<uint8_t>{});
                    }
                } else {
                    const char* text = reinterpret_cast<const char*>(sqlite3_column_text(stmt, i));
                    row.emplace_back(std::string(text ? text : ""));
                }
            }
            rows.push_back(std::move(row));
        }

        sqlite3_finalize(stmt);

        if (ret != SQLITE_DONE) {
            m_lastError = sqlite3_errmsg(m_db);
            HCCL_VM_ERROR("Step failed: {}", m_lastError);
            return HCCL_SIM_E_INTERNAL;
        }

        return HCCL_SIM_SUCCESS;
    }

    HcclVmResult HcclDBSqlite::Execute(const std::string& sql)
    {
        std::lock_guard lock(m_mutex);

        HCCLVM_CHK_PTR(m_db);

        char* errMsg = nullptr;
        int ret = sqlite3_exec(m_db, sql.c_str(), nullptr, nullptr, &errMsg);

        if (ret != SQLITE_OK) {
            m_lastError = errMsg ? errMsg : "Unknown error";
            HCCL_VM_ERROR("SQL error: {}, sql: {}", m_lastError, sql);
            if (errMsg) {
                sqlite3_free(errMsg);
            }
            return HCCL_SIM_E_INTERNAL;
        }

        return HCCL_SIM_SUCCESS;
    }

    HcclVmResult HcclDBSqlite::Query(const std::string& sql, std::vector<std::vector<std::string>>& rows)
    {
        std::lock_guard lock(m_mutex);

        HCCLVM_CHK_PTR(m_db);

        sqlite3_stmt* stmt = nullptr;
        int ret = sqlite3_prepare_v2(m_db, sql.c_str(), -1, &stmt, nullptr);
        if (ret != SQLITE_OK) {
            m_lastError = sqlite3_errmsg(m_db);
            HCCL_VM_ERROR("Prepare failed: {}", m_lastError);
            return HCCL_SIM_E_INTERNAL;
        }

        int colCount = sqlite3_column_count(stmt);

        while ((ret = sqlite3_step(stmt)) == SQLITE_ROW) {
            std::vector<std::string> row;
            for (int i = 0; i < colCount; ++i) {
                const char* text = reinterpret_cast<const char*>(sqlite3_column_text(stmt, i));
                row.push_back(text ? text : "");
            }
            rows.push_back(std::move(row));
        }

        sqlite3_finalize(stmt);

        if (ret != SQLITE_DONE) {
            m_lastError = sqlite3_errmsg(m_db);
            HCCL_VM_ERROR("Step failed: {}", m_lastError);
            return HCCL_SIM_E_INTERNAL;
        }

        return HCCL_SIM_SUCCESS;
    }

    int64_t HcclDBSqlite::LastInsertRowId() const
    {
        if (m_db == nullptr) {
            return 0;
        }
        return sqlite3_last_insert_rowid(m_db);
    }

    int HcclDBSqlite::Changes() const
    {
        if (m_db == nullptr) {
            return 0;
        }
        return sqlite3_changes(m_db);
    }

    std::string HcclDBSqlite::GetLastError() const { return m_lastError; }

    sim::DbType HcclDBSqlite::GetDbType() const { return sim::DbType::SQLITE3; }

    void* HcclDBSqlite::GetNativeHandle() { return m_db; }

    HcclVmResult HcclDBSqlite::Backup(const std::string& destPath)
    {
        std::lock_guard lock(m_mutex);

        if (m_db == nullptr) {
            m_lastError = "Database not connected";
            return HCCL_SIM_E_INTERNAL;
        }

        sqlite3* pDest = nullptr;
        sqlite3_backup* pBackup = nullptr;
        int rc;

        // 1. Open destination database
        rc = sqlite3_open(destPath.c_str(), &pDest);
        if (rc != SQLITE_OK) {
            m_lastError = sqlite3_errstr(rc);
            HCCL_VM_ERROR("Failed to open destination DB: {}", m_lastError);
            return HCCL_SIM_E_OPEN_FILE_FAILURE;
        }

        // 2. Initialize backup
        pBackup = sqlite3_backup_init(pDest, "main", m_db, "main");
        if (pBackup) {
            // 3. Perform backup step by step
            // Using -1 to copy all pages at once, but loop handles BUSY/LOCKED
            do {
                rc = sqlite3_backup_step(pBackup, -1);
            } while (rc == SQLITE_OK || rc == SQLITE_BUSY || rc == SQLITE_LOCKED);

            // 4. Finish backup
            rc = sqlite3_backup_finish(pBackup);
        } else {
            rc = sqlite3_errcode(pDest);
            m_lastError = sqlite3_errmsg(pDest);
            HCCL_VM_ERROR("Backup init failed: {}", m_lastError);
        }

        // 5. Close destination
        sqlite3_close(pDest);

        if (rc == SQLITE_OK) {
            HCCL_VM_INFO("Backup completed successfully to: {}", destPath);
            return HCCL_SIM_SUCCESS;
        } else {
            m_lastError = sqlite3_errstr(rc);
            HCCL_VM_ERROR("Backup failed: {}", m_lastError);
            return HCCL_SIM_E_INTERNAL;
        }
    }

    HcclVmResult HcclDBSqlite::RunInTransaction(std::function<HcclVmResult()> fn)
    {
        std::lock_guard lock(m_mutex);

        HCCLVM_CHK_PTR(m_db);

        char* errMsg = nullptr;
        int rc = sqlite3_exec(m_db, "BEGIN IMMEDIATE", nullptr, nullptr, &errMsg);
        if (rc != SQLITE_OK) {
            m_lastError = errMsg ? errMsg : "BEGIN IMMEDIATE failed";
            HCCL_VM_ERROR("BEGIN failed: {}", m_lastError);
            if (errMsg) {
                sqlite3_free(errMsg);
            }
            return HCCL_SIM_E_INTERNAL;
        }

        HcclVmResult result = fn();

        if (result == HCCL_SIM_SUCCESS) {
            rc = sqlite3_exec(m_db, "COMMIT", nullptr, nullptr, &errMsg);
            if (rc != SQLITE_OK) {
                m_lastError = errMsg ? errMsg : "COMMIT failed";
                HCCL_VM_ERROR("COMMIT failed: {}, rolling back", m_lastError);
                if (errMsg) {
                    sqlite3_free(errMsg);
                }
                sqlite3_exec(m_db, "ROLLBACK", nullptr, nullptr, nullptr);
                return HCCL_SIM_E_INTERNAL;
            }
        } else {
            sqlite3_exec(m_db, "ROLLBACK", nullptr, nullptr, nullptr);
        }

        return result;
    }
} // namespace DB
} // namespace HcclSim
