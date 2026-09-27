/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SIM_SQLITE_TABLE_H
#define SIM_SQLITE_TABLE_H

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <sqlite3.h>
#include <string>
#include <thread>
#include <typeindex>
#include <unordered_map>
#include <vector>

#include "sim_log.h"
#include "sim_common_api.h"

namespace sim {
inline std::shared_mutex& GetConnectionMutex()
{
    static std::shared_mutex s_connMtx;
    return s_connMtx;
}

inline int BusyHandler(void* /*data*/, int retryCount)
{
    static constexpr int kDelaysMs[] = {1, 2, 5, 10, 20, 50, 100, 200, 500};
    int idx = retryCount < 9 ? retryCount : 8;
    std::this_thread::sleep_for(std::chrono::milliseconds(kDelaysMs[idx]));
    return retryCount < 120;
}

class TableBase {
public:
    virtual ~TableBase() = default;
    virtual std::string GetTableName() const = 0;
};

template <typename T>
class SqliteTable : public TableBase {
public:
    using KeyType = uint64_t;
    using ValueType = T;

    SqliteTable(sqlite3* db, const std::string& tableName) : m_db(db), m_tableName(tableName)
    {
        CreateTableIfNotExists();
    }

    ~SqliteTable() override = default;

    SqliteTable(const SqliteTable&) = delete;
    SqliteTable& operator=(const SqliteTable&) = delete;

    KeyType Add(T& rec)
    {
        static constexpr int kMaxRetries = 8;
        static constexpr int kBackoffMs[] = {2, 5, 10, 20, 50, 100, 200, 500};

        for (int attempt = 0; attempt <= kMaxRetries; ++attempt) {
            std::unique_lock<std::shared_mutex> connLock(GetConnectionMutex());

            std::string sql = "INSERT INTO " + m_tableName + " (data) VALUES (?)";

            sqlite3_stmt* rawStmt;
            int rc = sqlite3_prepare_v2(m_db, sql.c_str(), -1, &rawStmt, nullptr);
            if (rc != SQLITE_OK) {
                HCCL_VM_ERROR(
                    "[SqliteTable::Add] SQL prepare failed [{}]: {} (rc={})", m_tableName.c_str(), sqlite3_errmsg(m_db),
                    rc);
                return 0;
            }

            std::unique_ptr<sqlite3_stmt, decltype(&sqlite3_finalize)> stmt(rawStmt, sqlite3_finalize);
            sqlite3_bind_blob(stmt.get(), 1, &rec, sizeof(T), SQLITE_STATIC);

            int stepRc = sqlite3_step(stmt.get());
            if (stepRc == SQLITE_DONE) {
                rec.id = static_cast<KeyType>(sqlite3_last_insert_rowid(m_db));
                return rec.id;
            }

            int errc = sqlite3_errcode(m_db);
            int extErrc = sqlite3_extended_errcode(m_db);

            bool isRetryable
                = (errc == SQLITE_BUSY || errc == SQLITE_LOCKED || extErrc == SQLITE_BUSY_SNAPSHOT
                   || extErrc == SQLITE_LOCKED_SHAREDCACHE);

            connLock.unlock();

            if (!isRetryable || attempt == kMaxRetries) {
                HCCL_VM_ERROR(
                    "[SqliteTable::Add] SQL insert failed [{}]: rc={}, errc={}, ext={}, msg={}", m_tableName.c_str(),
                    stepRc, errc, extErrc, sqlite3_errmsg(m_db));
                return 0;
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(kBackoffMs[attempt]));
        }
        return 0;
    }

    bool Update(KeyType id, std::function<void(T&)> updater)
    {
        static constexpr int kMaxRetries = 5;
        static constexpr int kBackoffMs[] = {2, 5, 10, 50, 200};

        for (int attempt = 0; attempt <= kMaxRetries; ++attempt) {
            std::unique_lock<std::shared_mutex> connLock(GetConnectionMutex());

            sqlite3_exec(m_db, "BEGIN IMMEDIATE", nullptr, nullptr, nullptr);

            T rec;
            bool found = FindInternal(id, rec);
            if (!found) {
                sqlite3_exec(m_db, "ROLLBACK", nullptr, nullptr, nullptr);
                return false;
            }

            updater(rec);

            std::string sql = "UPDATE " + m_tableName + " SET data = ? WHERE id = ?";

            sqlite3_stmt* rawStmt;
            if (sqlite3_prepare_v2(m_db, sql.c_str(), -1, &rawStmt, nullptr) != SQLITE_OK) {
                sqlite3_exec(m_db, "ROLLBACK", nullptr, nullptr, nullptr);
                HCCL_VM_ERROR(
                    "[SqliteTable::Update] SQL prepare failed [{}]: {}", m_tableName.c_str(), sqlite3_errmsg(m_db));
                return false;
            }

            std::unique_ptr<sqlite3_stmt, decltype(&sqlite3_finalize)> stmt(rawStmt, sqlite3_finalize);
            sqlite3_bind_blob(stmt.get(), 1, &rec, sizeof(T), SQLITE_STATIC);
            sqlite3_bind_int64(stmt.get(), 2, static_cast<sqlite3_int64>(id));

            int result = sqlite3_step(stmt.get());
            bool changed = (result == SQLITE_DONE) && (sqlite3_changes(m_db) > 0);

            stmt.reset();

            if (!changed && result != SQLITE_DONE) {
                int errc = sqlite3_errcode(m_db);
                sqlite3_exec(m_db, "ROLLBACK", nullptr, nullptr, nullptr);
                bool isRetryable = (errc == SQLITE_BUSY || errc == SQLITE_LOCKED);

                connLock.unlock();

                if (!isRetryable || attempt == kMaxRetries) {
                    HCCL_VM_ERROR(
                        "[SqliteTable::Update] SQL step failed [{}]: rc={}, errc={}, msg={}", m_tableName.c_str(),
                        result, errc, sqlite3_errmsg(m_db));
                    return false;
                }

                std::this_thread::sleep_for(std::chrono::milliseconds(kBackoffMs[attempt]));
                continue;
            }

            sqlite3_exec(m_db, changed ? "COMMIT" : "ROLLBACK", nullptr, nullptr, nullptr);
            return changed;
        }
        return false;
    }

    bool Delete(KeyType id)
    {
        std::unique_lock<std::shared_mutex> connLock(GetConnectionMutex());

        for (int attempt = 0; attempt < 5; ++attempt) {
            std::string sql = "DELETE FROM " + m_tableName + " WHERE id = ?";

            sqlite3_stmt* rawStmt;
            if (sqlite3_prepare_v2(m_db, sql.c_str(), -1, &rawStmt, nullptr) != SQLITE_OK) {
                HCCL_VM_ERROR(
                    "[SqliteTable::Delete] SQL prepare failed [{}]: {}", m_tableName.c_str(), sqlite3_errmsg(m_db));
                return false;
            }

            std::unique_ptr<sqlite3_stmt, decltype(&sqlite3_finalize)> stmt(rawStmt, sqlite3_finalize);
            sqlite3_bind_int64(stmt.get(), 1, static_cast<sqlite3_int64>(id));
            int result = sqlite3_step(stmt.get());
            if (result == SQLITE_DONE) {
                return sqlite3_changes(m_db) > 0;
            }

            int errc = sqlite3_errcode(m_db);
            if (errc != SQLITE_BUSY && errc != SQLITE_LOCKED) {
                HCCL_VM_ERROR(
                    "[SqliteTable::Delete] SQL step failed [{}]: rc={}, errc={}", m_tableName.c_str(), result, errc);
                return false;
            }

            connLock.unlock();
            static constexpr int kBackoffMs[] = {2, 5, 10, 50, 200};
            std::this_thread::sleep_for(std::chrono::milliseconds(kBackoffMs[attempt]));
            connLock.lock();
        }
        return false;
    }

    bool DeleteAll()
    {
        std::unique_lock<std::shared_mutex> connLock(GetConnectionMutex());

        std::string sql = "DELETE FROM " + m_tableName;
        if (sqlite3_exec(m_db, sql.c_str(), nullptr, nullptr, nullptr) != SQLITE_OK) {
            HCCL_VM_ERROR(
                "[SqliteTable::DeleteAll] SQL delete all failed [{}]: {}", m_tableName.c_str(), sqlite3_errmsg(m_db));
            return false;
        }

        std::string seqSql = "DELETE FROM sqlite_sequence WHERE name = ?";
        sqlite3_stmt* rawStmt;
        if (sqlite3_prepare_v2(m_db, seqSql.c_str(), -1, &rawStmt, nullptr) == SQLITE_OK) {
            std::unique_ptr<sqlite3_stmt, decltype(&sqlite3_finalize)> stmt(rawStmt, sqlite3_finalize);
            sqlite3_bind_text(stmt.get(), 1, m_tableName.c_str(), -1, SQLITE_TRANSIENT);
            sqlite3_step(stmt.get());
        }

        return true;
    }

    std::optional<ValueType> Find(KeyType id) const
    {
        std::shared_lock<std::shared_mutex> connLock(GetConnectionMutex());

        T rec;
        if (FindInternal(id, rec)) {
            return rec;
        }
        return std::nullopt;
    }

    std::vector<ValueType> QueryList(std::function<bool(const T&)> pred) const
    {
        std::shared_lock<std::shared_mutex> connLock(GetConnectionMutex());

        std::vector<ValueType> result;
        std::string sql = "SELECT id, data FROM " + m_tableName;
        static constexpr int kMaxRetries = 30;

        for (int attempt = 0; attempt <= kMaxRetries; ++attempt) {
            result.clear();

            sqlite3_stmt* rawStmt;
            if (sqlite3_prepare_v2(m_db, sql.c_str(), -1, &rawStmt, nullptr) != SQLITE_OK) {
                HCCL_VM_ERROR(
                    "[SqliteTable::QueryList] SQL prepare failed [{}]: {}", m_tableName.c_str(), sqlite3_errmsg(m_db));
                return result;
            }

            std::unique_ptr<sqlite3_stmt, decltype(&sqlite3_finalize)> stmt(rawStmt, sqlite3_finalize);
            bool busy = false;
            while (true) {
                int rc = sqlite3_step(stmt.get());
                if (rc == SQLITE_ROW) {
                    T rec;
                    const void* blob = sqlite3_column_blob(stmt.get(), 1);
                    int blobSize = sqlite3_column_bytes(stmt.get(), 1);
                    if (blobSize == static_cast<int>(sizeof(T))) {
                        std::memcpy(&rec, blob, sizeof(T));
                        rec.id = static_cast<KeyType>(sqlite3_column_int64(stmt.get(), 0));
                        if (pred(rec)) {
                            result.push_back(rec);
                        }
                    }
                } else if (rc == SQLITE_BUSY || rc == SQLITE_LOCKED) {
                    busy = true;
                    break;
                } else {
                    break;
                }
            }

            if (!busy) {
                return result;
            }

            connLock.unlock();
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            connLock.lock();
        }

        HCCL_VM_ERROR("[SqliteTable::QueryList] SQLITE_BUSY after {} retries [{}]", kMaxRetries, m_tableName.c_str());
        return result;
    }

    std::pair<ValueType, bool> Query(std::function<bool(const T&)> pred) const
    {
        std::shared_lock<std::shared_mutex> connLock(GetConnectionMutex());
        std::string sql = "SELECT id, data FROM " + m_tableName;
        static constexpr int kMaxRetries = 30;
        for (int attempt = 0; attempt <= kMaxRetries; ++attempt) {
            sqlite3_stmt* rawStmt;
            if (sqlite3_prepare_v2(m_db, sql.c_str(), -1, &rawStmt, nullptr) != SQLITE_OK) {
                HCCL_VM_ERROR(
                    "[SqliteTable::Query] SQL prepare failed [{}]: {}", m_tableName.c_str(), sqlite3_errmsg(m_db));
                return {ValueType{}, false};
            }

            std::unique_ptr<sqlite3_stmt, decltype(&sqlite3_finalize)> stmt(rawStmt, sqlite3_finalize);

            bool busy = false;
            while (true) {
                int rc = sqlite3_step(stmt.get());
                if (rc == SQLITE_ROW) {
                    T rec;
                    const void* blob = sqlite3_column_blob(stmt.get(), 1);
                    int blobSize = sqlite3_column_bytes(stmt.get(), 1);
                    if (blobSize == static_cast<int>(sizeof(T))) {
                        std::memcpy(&rec, blob, sizeof(T));
                        rec.id = static_cast<KeyType>(sqlite3_column_int64(stmt.get(), 0));
                        if (pred(rec)) {
                            return {rec, true};
                        }
                    }
                } else if (rc == SQLITE_BUSY || rc == SQLITE_LOCKED) {
                    busy = true;
                    break;
                } else {
                    break;
                }
            }

            if (!busy) {
                return {ValueType{}, false};
            }

            connLock.unlock();
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            connLock.lock();
        }

        HCCL_VM_ERROR("[SqliteTable::Query] SQLITE_BUSY after {} retries [{}]", kMaxRetries, m_tableName.c_str());
        return {ValueType{}, false};
    }

    std::string GetTableName() const override { return m_tableName; }

private:
    void CreateTableIfNotExists()
    {
        std::unique_lock<std::shared_mutex> connLock(GetConnectionMutex());
        std::string sql = "CREATE TABLE IF NOT EXISTS " + m_tableName
                          + " (id INTEGER PRIMARY KEY AUTOINCREMENT, data BLOB NOT NULL)";

        if (sqlite3_exec(m_db, sql.c_str(), nullptr, nullptr, nullptr) != SQLITE_OK) {
            HCCL_VM_ERROR(
                "[SqliteTable::CreateTableIfNotExists] SQL create table failed [{}]: {}", m_tableName.c_str(),
                sqlite3_errmsg(m_db));
        }
    }

    bool FindInternal(KeyType id, T& rec) const
    {
        std::string sql = "SELECT id, data FROM " + m_tableName + " WHERE id = ?";

        sqlite3_stmt* rawStmt;
        if (sqlite3_prepare_v2(m_db, sql.c_str(), -1, &rawStmt, nullptr) != SQLITE_OK) {
            HCCL_VM_ERROR(
                "[SqliteTable::FindInternal] SQL prepare failed [{}]: {}", m_tableName.c_str(), sqlite3_errmsg(m_db));
            return false;
        }

        std::unique_ptr<sqlite3_stmt, decltype(&sqlite3_finalize)> stmt(rawStmt, sqlite3_finalize);
        sqlite3_bind_int64(stmt.get(), 1, static_cast<sqlite3_int64>(id));

        if (sqlite3_step(stmt.get()) == SQLITE_ROW) {
            const void* blob = sqlite3_column_blob(stmt.get(), 1);
            int blobSize = sqlite3_column_bytes(stmt.get(), 1);

            if (blobSize == static_cast<int>(sizeof(T))) {
                std::memcpy(&rec, blob, sizeof(T));
                rec.id = static_cast<KeyType>(sqlite3_column_int64(stmt.get(), 0));
                return true;
            }
        }

        return false;
    }

    sqlite3* m_db;
    std::string m_tableName;
};

class SqliteDatabase {
public:
    static SqliteDatabase& Instance()
    {
        static SqliteDatabase s_instance;
        return s_instance;
    }

    static void SetDbPath(const std::string& path) { s_dbPath = path; }

    SqliteDatabase(const SqliteDatabase&) = delete;
    SqliteDatabase& operator=(const SqliteDatabase&) = delete;

    template <typename T>
    SqliteTable<T>& GetTable(const std::string& tableName)
    {
        {
            std::lock_guard<std::mutex> lock(m_tablesMutex);
            auto it = m_tables.find(tableName);
            if (it != m_tables.end()) {
                return *static_cast<SqliteTable<T>*>(m_tables[tableName].get());
            }
        }

        auto newTable = std::make_unique<SqliteTable<T>>(m_db, tableName);
        auto* rawPtr = newTable.get();

        std::lock_guard<std::mutex> lock(m_tablesMutex);
        auto it = m_tables.find(tableName);
        if (it == m_tables.end()) {
            m_tables[tableName] = std::move(newTable);
            m_tableTypes[tableName] = typeid(T).name();
            return *rawPtr;
        }
        return *static_cast<SqliteTable<T>*>(m_tables[tableName].get());
    }

    void Close()
    {
        std::unique_lock<std::shared_mutex> connLock(GetConnectionMutex());
        {
            std::lock_guard<std::mutex> lock(m_tablesMutex);
            m_tables.clear();

            if (m_db) {
                sqlite3_close(m_db);
                m_db = nullptr;
            }
        }
    }

    sqlite3* GetDb() { return m_db; }

    void ClearAllTables()
    {
        std::unique_lock<std::shared_mutex> connLock(GetConnectionMutex());
        std::lock_guard<std::mutex> lock(m_tablesMutex);

        for (auto& pair : m_tables) {
            std::string sql = "DELETE FROM " + pair.first;
            sqlite3_exec(m_db, sql.c_str(), nullptr, nullptr, nullptr);
        }
    }

    static void RemoveDbFile()
    {
        if (!s_dbPath.empty() && s_dbPath != ":memory:") {
            std::remove(s_dbPath.c_str());
            std::string walPath = s_dbPath + "-wal";
            std::remove(walPath.c_str());
            std::string shmPath = s_dbPath + "-shm";
            std::remove(shmPath.c_str());
        }
    }

public:
    static std::string s_dbPath;

    SqliteDatabase() : m_db(nullptr)
    {
        std::string actualPath = s_dbPath.empty() ? InstallPath::ResolveToInstallRoot("data/hccl_sim.db") : s_dbPath;
        if (sqlite3_open_v2(
                actualPath.c_str(), &m_db, SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_URI, nullptr)
            != SQLITE_OK) {
            HCCL_VM_ERROR("[SqliteDatabase] CRITICAL: Failed to open SQLite database: {}", sqlite3_errmsg(m_db));
        } else {
            sqlite3_exec(m_db, "PRAGMA journal_mode     = WAL", nullptr, nullptr, nullptr);
            sqlite3_exec(m_db, "PRAGMA synchronous      = NORMAL", nullptr, nullptr, nullptr);
            sqlite3_exec(m_db, "PRAGMA cache_size       = -10000", nullptr, nullptr, nullptr);
            sqlite3_exec(m_db, "PRAGMA temp_store       = MEMORY", nullptr, nullptr, nullptr);
            sqlite3_exec(m_db, "PRAGMA locking_mode     = NORMAL", nullptr, nullptr, nullptr);
            sqlite3_exec(m_db, "PRAGMA wal_autocheckpoint = 1000", nullptr, nullptr, nullptr);
            sqlite3_exec(m_db, "PRAGMA busy_timeout     = 60000", nullptr, nullptr, nullptr);
            sqlite3_busy_handler(m_db, BusyHandler, nullptr);
        }
    }

    ~SqliteDatabase() { Close(); }

    sqlite3* m_db;
    mutable std::mutex m_tablesMutex;
    std::unordered_map<std::string, std::unique_ptr<TableBase>> m_tables;
    std::unordered_map<std::string, std::string> m_tableTypes;
};
} // namespace sim

#endif
