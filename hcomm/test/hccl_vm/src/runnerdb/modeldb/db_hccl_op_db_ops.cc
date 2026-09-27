/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "db_hccl_op_db_ops.h"

#include <cstdint>
#include <filesystem>
#include <unistd.h>

#include "sim_common_macro.h"
#include "db_hccl_db_factory.h"
#include "db_hccl_db_sqlite.h"
#include "sim_log.h"
#include "sim_common_api.h"
#include "sim_yaml_config.h"

namespace HcclSim {
namespace DB {
    OpDbOps& OpDbOps::Instance()
    {
        static OpDbOps instance;
        return instance;
    }

    OpDbOps::OpDbOps()
    {
        sim::DBConfig config;
        config.type = sim::DbType::SQLITE3;
        config.dbPath = InstallPath::ResolveToInstallRoot("data/hccl_vm_data.db");

        auto db = HcclSim::DB::HcclDBFactory::Instance().CreateDB(config.type);
        if (!db || db->Connect(config) != HcclSim::HCCL_SIM_SUCCESS) {
            HCCL_VM_ERROR("Connect database failed");
            return;
        }

        SetDB(std::move(db));

        if (InitSchema() != HcclSim::HCCL_SIM_SUCCESS) {
            HCCL_VM_ERROR("Init schema failed");
        } else {
            HCCL_VM_INFO("OpDbOps tables initialized successfully");
        }
    }

    OpDbOps::~OpDbOps()
    {
        if (m_db != nullptr) {
            m_db->Close();
        }
    }

    int OpDbOps::SetDbConfig(sim::DBConfig& config)
    {
        m_db->Close();
        if (m_db->Connect(config) != HcclSim::HCCL_SIM_SUCCESS) {
            HCCL_VM_ERROR("Connect database:{} failed", config.dbPath);
            return -1;
        }
        return 0;
    }

    void OpDbOps::SetDB(std::unique_ptr<HcclDB> db) { m_db = std::move(db); }

    HcclDB* OpDbOps::GetDB() const { return m_db.get(); }

    HcclVmResult OpDbOps::InitSchema()
    {
        HCCLVM_CHK_PTR(m_db);

        std::string opTaskTable = "opTask_P_" + std::to_string(getpid());
        auto ret = m_db->RunInTransaction([this, &opTaskTable]() -> HcclVmResult {
            HCCLVM_CHK_RET(
                m_db->Execute("CREATE TABLE IF NOT EXISTS opDetails ("
                              "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                              "pid INTEGER, rankId INTEGER, opIter INTEGER, syncIter INTEGER, "
                              "streamId INTEGER, root INTEGER, opExpansionMode INTEGER, devType INTEGER,"
                              "rankSize INTEGER, srcRank INTEGER, dstRank INTEGER, opDetail BLOB, opExtInfo BLOB);"));

            HCCLVM_CHK_RET(
                m_db->Execute("CREATE INDEX IF NOT EXISTS idx_opdetail_pid_synciter ON opDetails(pid, syncIter);"));

            HCCLVM_CHK_RET(m_db->Execute("CREATE TABLE IF NOT EXISTS opMemInfo ("
                                         "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                                         "opDetailId INTEGER, inputAddr INTEGER, inputSize INTEGER, "
                                         "outputAddr INTEGER, outputSize INTEGER, cclAddr INTEGER, cclSize INTEGER, "
                                         "FOREIGN KEY (opDetailId) REFERENCES opDetails(id) ON DELETE CASCADE);"));

            HCCLVM_CHK_RET(
                m_db->Execute("CREATE INDEX IF NOT EXISTS idx_opmeminfo_op_detail_id ON opMemInfo(opDetailId);"));

            HCCLVM_CHK_RET(m_db->Execute("CREATE TABLE IF NOT EXISTS syncRecords ("
                                         "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                                         "pid INTEGER, rankId INTEGER, rankSize INTEGER, syncIter INTEGER, streamId "
                                         "INTEGER, status INTEGER DEFAULT 0);"));

            HCCLVM_CHK_RET(m_db->Execute("CREATE INDEX IF NOT EXISTS idx_sync_status ON syncRecords(status);"));
            HCCLVM_CHK_RET(
                m_db->Execute("CREATE INDEX IF NOT EXISTS idx_sync_pid_status ON syncRecords(pid, status);"));

            HCCLVM_CHK_RET(m_db->Execute("CREATE TABLE IF NOT EXISTS ccuChannels ("
                                         "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                                         "channelId INTEGER, srcDieId INTEGER, dstDieId INTEGER, "
                                         "srcRankId INTEGER, dstRankId INTEGER, leid BLOB, reid BLOB, "
                                         "protocol INTEGER, jettyNum INTEGER, jettyId BLOB);"));

            HCCLVM_CHK_RET(
                m_db->Execute("CREATE INDEX IF NOT EXISTS idx_ccuchannel_channelid ON ccuChannels(channelId);"));

            HCCLVM_CHK_RET(
                m_db->Execute("CREATE TABLE IF NOT EXISTS JettyMaps ("
                              "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                              "opDetailId INTEGER, srcDieId INTEGER, dstDieId INTEGER, "
                              "srcRankId INTEGER, dstRankId INTEGER, leid BLOB, reid BLOB, protocol INTEGER);"));

            HCCLVM_CHK_RET(
                m_db->Execute("CREATE INDEX IF NOT EXISTS idx_jettymap_op_detail_id ON JettyMaps(opDetailId);"));
            HCCLVM_CHK_RET(m_db->Execute(
                "CREATE INDEX IF NOT EXISTS idx_jettymap_srcdie_dstdie ON JettyMaps(srcDieId, dstDieId);"));

            HCCLVM_CHK_RET(m_db->Execute(
                "CREATE TABLE IF NOT EXISTS ccuInstrRes ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                "deviceId INTEGER, rankId INTEGER, dieId INTEGER, instrCount INTEGER, instrSpace BLOB);"));

            HCCLVM_CHK_RET(
                m_db->Execute("CREATE TABLE IF NOT EXISTS ccuInstr ("
                              "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                              "ccuInstrResId INTEGER, rankId INTEGER, startId INTEGER, instrInfoSize INTEGER, "
                              "FOREIGN KEY (ccuInstrResId) REFERENCES ccuInstrRes(id) ON DELETE CASCADE);"));

            HCCLVM_CHK_RET(m_db->Execute(
                "CREATE TABLE IF NOT EXISTS " + opTaskTable
                + " ("
                  "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                  "opDetailId INTEGER, taskSeq INTEGER, opTaskMeta BLOB);"));

            return HCCL_SIM_SUCCESS;
        });

        if (ret == HCCL_SIM_SUCCESS) {
            HCCL_VM_INFO("All tables created");
        }
        return ret;
    }

    int OpDbOps::ExecInsert(const std::string& sql, const std::vector<Value>& params, uint32_t& outId)
    {
        HCCLVM_CHK_PTR(m_db);

        auto ret = m_db->Execute(sql, params);
        if (ret != HCCL_SIM_SUCCESS) {
            return -1;
        }

        outId = static_cast<uint32_t>(m_db->LastInsertRowId());
        return 0;
    }

    int OpDbOps::ExecInsertNoId(const std::string& sql, const std::vector<Value>& params)
    {
        HCCLVM_CHK_PTR(m_db);
        return m_db->Execute(sql, params) == HCCL_SIM_SUCCESS ? 0 : -1;
    }

    int OpDbOps::ExecQuery(
        const std::string& sql, const std::vector<Value>& params, std::vector<std::vector<std::string>>& rows)
    {
        HCCLVM_CHK_PTR(m_db);
        return m_db->Query(sql, params, rows) == HCCL_SIM_SUCCESS ? 0 : -1;
    }

    int OpDbOps::ExecUpdate(const std::string& sql, const std::vector<Value>& params)
    {
        HCCLVM_CHK_PTR(m_db);
        return m_db->Execute(sql, params) == HCCL_SIM_SUCCESS ? 0 : -1;
    }

    int
    OpDbOps::QuerySingleBlob(const std::string& sql, const std::vector<Value>& params, std::vector<uint8_t>& outBlob)
    {
        HCCLVM_CHK_PTR(m_db);
        return m_db->QueryBlob(sql, params, outBlob) == HCCL_SIM_SUCCESS ? 0 : -1;
    }

    int OpDbOps::QueryBlobColumns(
        const std::string& sql, const std::vector<Value>& params, std::vector<std::vector<uint8_t>>& outBlobs)
    {
        HCCLVM_CHK_PTR(m_db);
        return m_db->QueryBlobColumns(sql, params, outBlobs) == HCCL_SIM_SUCCESS ? 0 : -1;
    }

    int OpDbOps::ExecQueryEx(const std::string& sql, const std::vector<Value>& params, ExRows& rows)
    {
        HCCLVM_CHK_PTR(m_db);
        return m_db->QueryEx(sql, params, rows) == HCCL_SIM_SUCCESS ? 0 : -1;
    }

    HcclVmResult OpDbOps::Backup(const std::string& destPath)
    {
        HCCLVM_CHK_PTR(m_db);
        return m_db->Backup(destPath);
    }

    int OpDbOps::RunInTransaction(std::function<int()> fn)
    {
        HCCLVM_CHK_PTR(m_db);
        auto ret = m_db->RunInTransaction([&fn]() -> HcclVmResult {
            return fn() == 0 ? HCCL_SIM_SUCCESS : HCCL_SIM_E_INTERNAL;
        });
        return ret == HCCL_SIM_SUCCESS ? 0 : -1;
    }
} // namespace DB
} // namespace HcclSim
