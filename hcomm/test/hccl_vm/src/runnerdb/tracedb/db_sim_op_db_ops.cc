/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "db_sim_op_db_ops.h"

#include <cstdint>
#include <cstring>
#include <variant>

#include "sim_common_macro.h"
#include "db_hccl_op_db_ops.h"
#include "sim_log.h"
#include "sim_models.h"

using HcclSim::DB::BlobData;
using HcclSim::DB::ExRows;
using HcclSim::DB::Field;
using HcclSim::DB::OpDbOps;
using HcclSim::DB::Value;

namespace sim {
uint32_t g_opIterCounter = 0;
uint32_t g_syncIterCounter = 0;
uint32_t g_currOpDetailId = 0;
uint32_t g_currOpMemId = 0;

int InitOpDataDb()
{
    OpDbOps::Instance();
    return 0;
}

int SetDbConfig(DBConfig& config) { return OpDbOps::Instance().SetDbConfig(config); }

// ============================================================
// Insert APIs
// ============================================================

int InsertOpDetail(OpDetailTab& rec)
{
    rec.opIter = g_opIterCounter++;
    rec.syncIter = g_syncIterCounter;

    const std::string sql
        = "INSERT INTO opDetails (pid, rankId, opIter, syncIter, streamId, root, opExpansionMode, devType, rankSize, "
          "srcRank, dstRank, opDetail, opExtInfo) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)";
    std::vector<Value> params
        = {(int64_t)rec.pid,
           (int64_t)rec.rankId,
           (int64_t)rec.opIter,
           (int64_t)rec.syncIter,
           (int64_t)rec.streamId,
           (int64_t)rec.root,
           (int64_t)rec.opExpansionMode,
           (int64_t)rec.devType,
           (int64_t)rec.rankSize,
           (int64_t)rec.srcRank,
           (int64_t)rec.dstRank,
           BlobData{rec.opDetail.data(), rec.opDetail.size()},
           BlobData{rec.opExtInfo.data(), rec.opExtInfo.size()}};

    int ret = OpDbOps::Instance().ExecInsert(sql, params, rec.id);
    if (ret == 0) {
        g_currOpDetailId = rec.id;
    }
    return ret;
}

int InsertOpMem(OpMemInfoTab& rec)
{
    rec.opDetailId = g_currOpDetailId;

    const std::string sql = "INSERT INTO opMemInfo (opDetailId, inputAddr, inputSize, outputAddr, outputSize, cclAddr, "
                            "cclSize) VALUES (?, ?, ?, ?, ?, ?, ?)";
    std::vector<Value> params
        = {(int64_t)rec.opDetailId, (int64_t)rec.inputAddr, (int64_t)rec.inputSize, (int64_t)rec.outputAddr,
           (int64_t)rec.outputSize, (int64_t)rec.cclAddr,   (int64_t)rec.cclSize};
    return OpDbOps::Instance().ExecInsert(sql, params, g_currOpMemId);
}

int InsertOpDetailAndMem(OpDetailTab& detail, OpMemInfoTab& mem)
{
    return OpDbOps::Instance().RunInTransaction([&]() -> int {
        int ret = InsertOpDetail(detail);
        if (ret != 0) {
            return ret;
        }
        mem.opDetailId = detail.id;
        return InsertOpMem(mem);
    });
}

int InsertCcuChannel(CcuChannelTab& rec)
{
    const std::string sql = "INSERT INTO ccuChannels (channelId, srcDieId, dstDieId, srcRankId, dstRankId, leid, reid, "
                            "protocol, jettyNum, jettyId) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)";
    std::vector<uint8_t> jettyId((uint8_t*)rec.jettyId, (uint8_t*)rec.jettyId + rec.jettyNum * 4);
    std::vector<Value> params = {(int64_t)rec.channelId, (int64_t)rec.srcDieId,
                                 (int64_t)rec.dstDieId,  (int64_t)rec.srcRankId,
                                 (int64_t)rec.dstRankId, BlobData{rec.leid, 16},
                                 BlobData{rec.reid, 16}, (int64_t)rec.protocol,
                                 (int64_t)rec.jettyNum,  BlobData{jettyId.data(), jettyId.size()}};
    uint32_t dummy;
    return OpDbOps::Instance().ExecInsert(sql, params, dummy);
}

int InsertJettyMap(JettyMapTab& rec)
{
    rec.opDetailId = g_currOpDetailId;
    const std::string sql = "INSERT INTO JettyMaps (opDetailId, srcDieId, dstDieId, srcRankId, dstRankId, leid, reid, "
                            "protocol) VALUES (?, ?, ?, ?, ?, ?, ?, ?)";
    std::vector<Value> params
        = {(int64_t)rec.opDetailId, (int64_t)rec.srcDieId,  (int64_t)rec.dstDieId,  (int64_t)rec.srcRankId,
           (int64_t)rec.dstRankId,  BlobData{rec.leid, 16}, BlobData{rec.reid, 16}, (int64_t)rec.protocol};
    uint32_t dummy;
    return OpDbOps::Instance().ExecInsert(sql, params, dummy);
}

int InsertOpTask(OpTaskTab& rec, bool isDevice)
{
    rec.opDetailId = g_currOpDetailId;
    // host进程使用pid，AICPU模式的device进程使用ppid(即host进程的pid)
    std::string tabPid = isDevice ? std::to_string(getppid()) : std::to_string(getpid());
    std::string tabName = "opTask_P_" + tabPid;

    const std::string sql = "INSERT INTO " + tabName + " (opDetailId, taskSeq, opTaskMeta) VALUES (?, ?, ?)";
    std::vector<Value> params
        = {(int64_t)rec.opDetailId, (int64_t)rec.taskSeq, BlobData{rec.optaskMeta.data(), rec.optaskMeta.size()}};
    uint32_t dummy;
    return OpDbOps::Instance().ExecInsert(sql, params, dummy);
}

int InsertSyncRecord(SyncRecordTab& rec)
{
    rec.syncIter = g_syncIterCounter++;
    const std::string sql
        = "INSERT INTO syncRecords (pid, rankId, rankSize, syncIter, streamId, status) VALUES (?, ?, ?, ?, ?, ?)";
    std::vector<Value> params = {(int64_t)rec.pid,      (int64_t)rec.rankId,   (int64_t)rec.rankSize,
                                 (int64_t)rec.syncIter, (int64_t)rec.streamId, (int64_t)rec.status};
    return OpDbOps::Instance().ExecInsert(sql, params, rec.id);
}

int InsertCcuInstrRes(CcuInstrResTab& rec)
{
    const std::string sql
        = "INSERT INTO ccuInstrRes (deviceId, rankId, dieId, instrCount, instrSpace) VALUES (?, ?, ?, ?, ?)";
    std::vector<Value> params
        = {(int64_t)rec.deviceId, (int64_t)rec.rankId, (int64_t)rec.dieId, (int64_t)rec.instrCount,
           BlobData{rec.instrSpace, sizeof(rec.instrSpace)}};
    return OpDbOps::Instance().ExecInsert(sql, params, rec.id);
}

int InsertCcuInstr(CcuInstrTab& rec)
{
    const std::string sql = "INSERT INTO ccuInstr (ccuInstrResId, rankId, startId, instrInfoSize) VALUES (?, ?, ?, ?)";
    std::vector<Value> params
        = {(int64_t)rec.ccuInstrResId, (int64_t)rec.rankId, (int64_t)rec.startId, (int64_t)rec.instrInfoSize};
    return OpDbOps::Instance().ExecInsert(sql, params, rec.id);
}

int UpdateAndInsertByCcuId(
    uint64_t& ccuId, uint32_t deviceId, uint32_t rankId, uint32_t dieId, uint32_t instrCount, uint32_t instrOffset,
    uint32_t instrInfoSize, const void* instrInfo)
{
    constexpr size_t kInstrSpaceSize = HcclSim::CCU_INSTRUCTION_NUM * HcclSim::CCU_INSTRUCTION_SIZE;

    if (instrOffset + instrInfoSize > kInstrSpaceSize) {
        HCCL_VM_ERROR("Offset {} + size {} exceeds space {}", instrOffset, instrInfoSize, kInstrSpaceSize);
        return -1;
    }

    const std::string querySql = "SELECT id, instrSpace, instrCount FROM ccuInstrRes WHERE deviceId = ? AND dieId = ?";
    std::vector<Value> queryParams = {(int64_t)deviceId, (int64_t)dieId};
    ExRows rows;
    int ret = OpDbOps::Instance().ExecQueryEx(querySql, queryParams, rows);
    if (ret != 0) {
        HCCL_VM_ERROR("query ccuInstrRes failed for deviceId: {} dieId: {}", deviceId, dieId);
        return -1;
    }

    if (!rows.empty() && rows[0].size() >= 3) {
        uint32_t existId = std::stoul(std::get<std::string>(rows[0][0]));
        std::vector<uint8_t> space(kInstrSpaceSize, 0);
        if (std::holds_alternative<std::vector<uint8_t>>(rows[0][1])) {
            const auto& existBlob = std::get<std::vector<uint8_t>>(rows[0][1]);
            std::memcpy(space.data(), existBlob.data(), std::min(existBlob.size(), kInstrSpaceSize));
        }
        std::memcpy(space.data() + instrOffset, instrInfo, instrInfoSize);

        uint32_t oldCount = std::stoul(std::get<std::string>(rows[0][2]));
        uint32_t newCount = oldCount + instrCount;

        const std::string updateSql = "UPDATE ccuInstrRes SET instrSpace = ?, instrCount = ? WHERE id = ?";
        std::vector<Value> updateParams = {BlobData{space.data(), space.size()}, (int64_t)newCount, (int64_t)existId};
        ret = OpDbOps::Instance().ExecUpdate(updateSql, updateParams);
        if (ret == 0) {
            ccuId = existId;
        }
        return ret;
    } else {
        std::vector<uint8_t> space(kInstrSpaceSize, 0);
        std::memcpy(space.data() + instrOffset, instrInfo, instrInfoSize);

        const std::string insertSql
            = "INSERT INTO ccuInstrRes (deviceId, rankId, dieId, instrCount, instrSpace) VALUES (?, ?, ?, ?, ?)";
        std::vector<Value> insertParams
            = {(int64_t)deviceId, (int64_t)rankId, (int64_t)dieId, (int64_t)instrCount,
               BlobData{space.data(), space.size()}};
        uint32_t newId = 0;
        ret = OpDbOps::Instance().ExecInsert(insertSql, insertParams, newId);
        if (ret == 0) {
            ccuId = newId;
        }
        return ret;
    }
}

// ============================================================
// Query / Update APIs
// ============================================================

int QuerySyncRecordByStatus(uint8_t status, std::vector<SyncRecordTab>& out)
{
    // 1. 补全缺失的列 (rankId, rankSize)，确保结构体数据完整
    const std::string sql
        = "SELECT id, pid, rankId, rankSize, syncIter, streamId, status FROM syncRecords WHERE status = ?";
    std::vector<Value> params = {(int64_t)status};
    std::vector<std::vector<std::string>> rows;

    if (OpDbOps::Instance().ExecQuery(sql, params, rows) != 0) {
        return -1;
    }

    out.clear();

    // 2. 使用枚举替代魔鬼数字，提高可读性和维护性
    enum : int { COL_ID, COL_PID, COL_RANK_ID, COL_RANK_SIZE, COL_SYNC_ITER, COL_STREAM_ID, COL_STATUS };
    const size_t COL_COUNT = 7;

    for (const auto& r : rows) {
        if (r.size() < COL_COUNT) {
            continue;
        }

        SyncRecordTab rec;
        rec.id = std::stoul(r[COL_ID]);
        rec.pid = std::stoul(r[COL_PID]);
        rec.rankId = std::stoul(r[COL_RANK_ID]);
        rec.rankSize = std::stoul(r[COL_RANK_SIZE]);
        rec.syncIter = std::stoul(r[COL_SYNC_ITER]);
        rec.streamId = std::stoull(r[COL_STREAM_ID]);
        rec.status = static_cast<uint8_t>(std::stoul(r[COL_STATUS]));

        out.push_back(std::move(rec));
    }
    return 0;
}

int UpdateSyncRecordStatus(std::vector<SyncRecordTab>& syncRecord)
{
    return OpDbOps::Instance().RunInTransaction([&]() -> int {
        const std::string sql = "UPDATE syncRecords SET status = ? WHERE id = ?";
        for (const auto& rec : syncRecord) {
            std::vector<Value> params = {(int64_t)rec.status, (int64_t)rec.id};
            int ret = OpDbOps::Instance().ExecUpdate(sql, params);
            if (ret != 0) {
                return ret;
            }
        }
        return 0;
    });
}

int UpdateOpMemCclBuffer(uint64_t cclAddr, uint64_t cclSize)
{
    const std::string sql = "UPDATE opMemInfo SET cclAddr = ?, cclSize = ? WHERE id = ?";
    std::vector<Value> params = {(int64_t)cclAddr, (int64_t)cclSize, (int64_t)g_currOpMemId};
    int ret = OpDbOps::Instance().ExecUpdate(sql, params);
    if (ret != 0) {
        return ret;
    }
    return 0;
}

int QueryCurrentOpMemInfoByRank(uint32_t rankId, OpMemInfoTab& out)
{
    const std::string sql = "SELECT m.id, m.opDetailId, m.inputAddr, m.inputSize, "
                            "m.outputAddr, m.outputSize, m.cclAddr, m.cclSize "
                            "FROM opMemInfo m "
                            "JOIN opDetails d ON m.opDetailId = d.id "
                            "WHERE d.syncIter = (SELECT syncIter FROM opDetails WHERE id = ?) "
                            "AND d.rankId = ? "
                            "ORDER BY d.id DESC LIMIT 1";
    std::vector<Value> params = {(int64_t)g_currOpDetailId, (int64_t)rankId};
    std::vector<std::vector<std::string>> rows;
    if (OpDbOps::Instance().ExecQuery(sql, params, rows) != 0 || rows.empty()) {
        HCCL_VM_ERROR("query failed, currentOpDetailId={}, rankId={}", g_currOpDetailId, rankId);
        return -1;
    }

    const auto& row = rows[0];
    if (row.size() < 8) {
        HCCL_VM_ERROR("invalid row size={}, currentOpDetailId={}, rankId={}", row.size(), g_currOpDetailId, rankId);
        return -1;
    }

    out.id = std::stoul(row[0]);
    out.opDetailId = std::stoul(row[1]);
    out.inputAddr = std::stoull(row[2]);
    out.inputSize = std::stoull(row[3]);
    out.outputAddr = std::stoull(row[4]);
    out.outputSize = std::stoull(row[5]);
    out.cclAddr = std::stoull(row[6]);
    out.cclSize = std::stoull(row[7]);
    return 0;
}

int UpdateOpExpansionMode(uint8_t mode)
{
    const std::string sql = "UPDATE opDetails SET opExpansionMode = ? WHERE id = ?";
    std::vector<Value> params = {(int64_t)mode, (int64_t)g_currOpDetailId};
    int ret = OpDbOps::Instance().ExecUpdate(sql, params);
    if (ret != 0) {
        HCCL_VM_ERROR("Failed to update mode={}, id={}", mode, g_currOpDetailId);
    }
    return ret;
}

int QueryLatestOpExpansionMode()
{
    const std::string sql = "SELECT opExpansionMode FROM opDetails WHERE id = ?";
    std::vector<Value> params = {(int64_t)g_currOpDetailId};
    std::vector<std::vector<std::string>> rows;
    if (OpDbOps::Instance().ExecQuery(sql, params, rows) != 0 || rows.empty() || rows[0].empty()) {
        HCCL_VM_ERROR("Failed to query, id={}", g_currOpDetailId);
        return static_cast<int>(SimOpExpansionMode::SIM_OP_EXPANSION_MODE_RESERVED);
    }
    return static_cast<int>(std::stoul(rows[0][0]));
}

int QueryCompositeOpDetailBySyncIter(uint32_t syncIter, std::map<uint32_t, std::vector<CompositeOpDetail>>& detail)
{
    // 1. 定义 opDetails 表的列索引枚举 (共 14 列)
    enum : int {
        OD_ID,
        OD_PID,
        OD_RANK_ID,
        OD_OP_ITER,
        OD_SYNC_ITER,
        OD_STREAM_ID,
        OD_ROOT,
        OD_OP_EXPANSION,
        OD_DEV_TYPE,
        OD_RANK_SIZE,
        OD_SRC_RANK,
        OD_DST_RANK,
        OD_OP_DETAIL,
        OD_OP_EXT_INFO
    };
    const size_t OD_COL_COUNT = 14;

    // 2. 定义 opMemInfo 表的列索引枚举 (共 8 列)
    enum : int {
        MEM_ID,
        MEM_OP_DETAIL_ID,
        MEM_INPUT_ADDR,
        MEM_INPUT_SIZE,
        MEM_OUTPUT_ADDR,
        MEM_OUTPUT_SIZE,
        MEM_CCL_ADDR,
        MEM_CCL_SIZE
    };
    const size_t MEM_COL_COUNT = 8;

    // 3. 定义 opTask 表的列索引枚举 (共 4 列)
    enum : int { TASK_ID, TASK_OP_DETAIL_ID, TASK_SEQ, TASK_META };
    const size_t TASK_COL_COUNT = 4;

    using HcclSim::DB::Field;

    detail.clear();

    const std::string opSql = "SELECT id, pid, rankId, opIter, syncIter, streamId, "
                              "root, opExpansionMode, devType, rankSize, srcRank, dstRank, "
                              "opDetail, opExtInfo FROM opDetails WHERE syncIter = ?";
    std::vector<Value> opParams = {(int64_t)syncIter};
    HcclSim::DB::ExRows opRows;
    if (OpDbOps::Instance().ExecQueryEx(opSql, opParams, opRows) != 0) {
        HCCL_VM_ERROR("Query opDetails failed for syncIter: {}", syncIter);
        return -1;
    }

    const std::string memSql = "SELECT id, opDetailId, inputAddr, inputSize, "
                               "outputAddr, outputSize, cclAddr, cclSize "
                               "FROM opMemInfo WHERE opDetailId = ?";

    // 提取 Lambda 到循环外，避免重复定义
    auto toStr = [](const Field& f) -> std::string {
        if (std::holds_alternative<std::string>(f)) {
            return std::get<std::string>(f);
        }
        return {};
    };

    for (const auto& r : opRows) {
        // 使用枚举常量替代魔鬼数字 14
        if (r.size() < OD_COL_COUNT) {
            continue;
        }

        CompositeOpDetail comp = {};
        comp.rankId = static_cast<uint32_t>(std::stoul(toStr(r[OD_RANK_ID])));
        comp.detail.id = std::stoul(toStr(r[OD_ID]));
        comp.detail.pid = std::stoul(toStr(r[OD_PID]));
        comp.detail.rankId = std::stoul(toStr(r[OD_RANK_ID]));
        comp.detail.opIter = std::stoul(toStr(r[OD_OP_ITER]));
        comp.detail.syncIter = std::stoul(toStr(r[OD_SYNC_ITER]));
        comp.detail.streamId = std::stoull(toStr(r[OD_STREAM_ID]));
        comp.detail.root = std::stoul(toStr(r[OD_ROOT]));
        comp.detail.opExpansionMode = std::stoul(toStr(r[OD_OP_EXPANSION]));
        comp.detail.devType = std::stoul(toStr(r[OD_DEV_TYPE]));
        comp.detail.rankSize = std::stoul(toStr(r[OD_RANK_SIZE]));
        comp.detail.srcRank = std::stoul(toStr(r[OD_SRC_RANK]));
        comp.detail.dstRank = std::stoul(toStr(r[OD_DST_RANK]));

        if (std::holds_alternative<std::vector<uint8_t>>(r[OD_OP_DETAIL])) {
            comp.detail.opDetail = std::get<std::vector<uint8_t>>(r[OD_OP_DETAIL]);
        }
        if (std::holds_alternative<std::vector<uint8_t>>(r[OD_OP_EXT_INFO])) {
            comp.detail.opExtInfo = std::get<std::vector<uint8_t>>(r[OD_OP_EXT_INFO]);
        }

        const uint32_t opDetailId = comp.detail.id;
        const uint32_t pid = comp.detail.pid;

        // 查询 MemInfo
        std::vector<Value> memParams = {(int64_t)opDetailId};
        std::vector<std::vector<std::string>> memRows;
        if (OpDbOps::Instance().ExecQuery(memSql, memParams, memRows) == 0 && !memRows.empty()) {
            const auto& mr = memRows[0];
            // 使用枚举常量替代魔鬼数字
            if (mr.size() >= MEM_COL_COUNT) {
                comp.memInfo.id = std::stoul(mr[MEM_ID]);
                comp.memInfo.opDetailId = std::stoul(mr[MEM_OP_DETAIL_ID]);
                comp.memInfo.inputAddr = std::stoull(mr[MEM_INPUT_ADDR]);
                comp.memInfo.inputSize = std::stoull(mr[MEM_INPUT_SIZE]);
                comp.memInfo.outputAddr = std::stoull(mr[MEM_OUTPUT_ADDR]);
                comp.memInfo.outputSize = std::stoull(mr[MEM_OUTPUT_SIZE]);
                comp.memInfo.cclAddr = std::stoull(mr[MEM_CCL_ADDR]);
                comp.memInfo.cclSize = std::stoull(mr[MEM_CCL_SIZE]);
            }
        }

        // 查询 Tasks
        const std::string taskTabName = "opTask_P_" + std::to_string(pid);
        const std::string taskSql = "SELECT id, opDetailId, taskSeq, opTaskMeta FROM " + taskTabName
                                    + " WHERE opDetailId = ? ORDER BY taskSeq";
        std::vector<Value> taskParams = {(int64_t)opDetailId};
        HcclSim::DB::ExRows taskRows;
        if (OpDbOps::Instance().ExecQueryEx(taskSql, taskParams, taskRows) == 0) {
            for (const auto& tr : taskRows) {
                // 为确保数据完整性（特别是 opTaskMeta），应检查完整的列数
                if (tr.size() < TASK_COL_COUNT) {
                    continue;
                }
                OpTaskTab taskRec;
                taskRec.id = std::stoul(toStr(tr[TASK_ID]));
                taskRec.opDetailId = std::stoul(toStr(tr[TASK_OP_DETAIL_ID]));
                taskRec.taskSeq = std::stoul(toStr(tr[TASK_SEQ]));

                // 安全访问 Meta 数据
                if (std::holds_alternative<std::vector<uint8_t>>(tr[TASK_META])) {
                    taskRec.optaskMeta = std::get<std::vector<uint8_t>>(tr[TASK_META]);
                }
                comp.tasks.push_back(std::move(taskRec));
            }
        }

        detail[static_cast<int32_t>(comp.rankId)].push_back(std::move(comp));
    }

    return 0;
}

// ============================================================
// Query Single APIs
// ============================================================

int QueryNewestOpDeatailIdByPid(uint64_t pid, uint32_t& OpDetailId)
{
    const std::string sql = "SELECT id FROM opDetails WHERE pid = ? ORDER BY id DESC LIMIT 1";
    std::vector<Value> params = {(int64_t)pid};
    std::vector<std::vector<std::string>> rows;

    if (OpDbOps::Instance().ExecQuery(sql, params, rows) != 0) {
        return -1;
    }

    if (rows.empty()) {
        return -1;
    }

    OpDetailId = std::stoull(rows[0][0]);
    return 0;
}

// ============================================================
// Query All APIs
// ============================================================

int QueryCcuChannelAll(std::vector<CcuChannelTab>& out)
{
    enum : int {
        ID,
        CHANNEL_ID,
        SRC_DIE_ID,
        DST_DIE_ID,
        SRC_RANK_ID,
        DST_RANK_ID,
        LEID,
        REID,
        PROTOCOL,
        JETTY_NUM,
        JETTY_ID
    };
    const size_t CHANNEL_COL_COUNT = 11;
    const std::string sql = "SELECT id, channelId, srcDieId, dstDieId, srcRankId, dstRankId, leid, reid, protocol, "
                            "jettyNum, jettyId FROM ccuChannels";
    using HcclSim::DB::Field;
    HcclSim::DB::ExRows rows;
    if (OpDbOps::Instance().ExecQueryEx(sql, {}, rows) != 0) {
        return -1;
    }

    auto toStr = [](const Field& f) -> std::string {
        if (std::holds_alternative<std::string>(f)) {
            return std::get<std::string>(f);
        }
        return {};
    };

    out.clear();
    for (const auto& r : rows) {
        if (r.size() < CHANNEL_COL_COUNT) {
            continue;
        }
        CcuChannelTab rec = {};
        rec.id = std::stoul(toStr(r[ID]));
        rec.channelId = std::stoul(toStr(r[CHANNEL_ID]));
        rec.srcDieId = std::stoul(toStr(r[SRC_DIE_ID]));
        rec.dstDieId = std::stoul(toStr(r[DST_DIE_ID]));
        rec.srcRankId = std::stoul(toStr(r[SRC_RANK_ID]));
        rec.dstRankId = std::stoul(toStr(r[DST_RANK_ID]));
        rec.protocol = static_cast<uint16_t>(std::stoul(toStr(r[PROTOCOL])));
        rec.jettyNum = static_cast<uint16_t>(std::stoul(toStr(r[JETTY_NUM])));

        if (std::holds_alternative<std::vector<uint8_t>>(r[LEID])) {
            const auto& blob = std::get<std::vector<uint8_t>>(r[LEID]);
            std::memcpy(rec.leid, blob.data(), std::min(blob.size(), sizeof(rec.leid)));
        }
        if (std::holds_alternative<std::vector<uint8_t>>(r[REID])) {
            const auto& blob = std::get<std::vector<uint8_t>>(r[REID]);
            std::memcpy(rec.reid, blob.data(), std::min(blob.size(), sizeof(rec.reid)));
        }
        if (std::holds_alternative<std::vector<uint8_t>>(r[JETTY_ID])) {
            const auto& blob = std::get<std::vector<uint8_t>>(r[JETTY_ID]);
            std::memcpy(rec.jettyId, blob.data(), std::min(blob.size(), sizeof(rec.jettyId)));
        }

        out.push_back(std::move(rec));
    }
    return 0;
}

int QueryJettyMapAll(std::vector<JettyMapTab>& out)
{
    enum : int { ID, OP_DETAIL_ID, SRC_DIE_ID, DST_DIE_ID, SRC_RANK_ID, DST_RANK_ID, LEID, REID, PROTOCOL };
    const size_t JETTY_COL_COUNT = 9;
    const std::string sql
        = "SELECT id, opDetailId, srcDieId, dstDieId, srcRankId, dstRankId, leid, reid, protocol FROM JettyMaps";
    using HcclSim::DB::Field;
    HcclSim::DB::ExRows rows;
    if (OpDbOps::Instance().ExecQueryEx(sql, {}, rows) != 0) {
        return -1;
    }

    auto toStr = [](const Field& f) -> std::string {
        if (std::holds_alternative<std::string>(f)) {
            return std::get<std::string>(f);
        }
        return {};
    };

    out.clear();
    for (const auto& r : rows) {
        if (r.size() < JETTY_COL_COUNT) {
            continue;
        }
        JettyMapTab rec = {};
        rec.id = std::stoul(toStr(r[ID]));
        rec.opDetailId = std::stoul(toStr(r[OP_DETAIL_ID]));
        rec.srcDieId = std::stoul(toStr(r[SRC_DIE_ID]));
        rec.dstDieId = std::stoul(toStr(r[DST_DIE_ID]));
        rec.srcRankId = std::stoul(toStr(r[SRC_RANK_ID]));
        rec.dstRankId = std::stoul(toStr(r[DST_RANK_ID]));
        rec.protocol = static_cast<uint16_t>(std::stoul(toStr(r[PROTOCOL])));

        if (std::holds_alternative<std::vector<uint8_t>>(r[LEID])) {
            const auto& blob = std::get<std::vector<uint8_t>>(r[LEID]);
            std::memcpy(rec.leid, blob.data(), std::min(blob.size(), sizeof(rec.leid)));
        }
        if (std::holds_alternative<std::vector<uint8_t>>(r[REID])) {
            const auto& blob = std::get<std::vector<uint8_t>>(r[REID]);
            std::memcpy(rec.reid, blob.data(), std::min(blob.size(), sizeof(rec.reid)));
        }

        out.push_back(std::move(rec));
    }
    return 0;
}

int QuerySyncRecordAll(std::vector<SyncRecordTab>& out)
{
    enum : int { ID, PID, RANK_ID, RANK_SIZE, SYNC_ITER, STREAM_ID, STATUS };
    const size_t RECORD_COL_COUNT = 7;
    const std::string sql = "SELECT id, pid, rankId, rankSize, syncIter, streamId, status FROM syncRecords";
    std::vector<std::vector<std::string>> rows;
    if (OpDbOps::Instance().ExecQuery(sql, {}, rows) != 0) {
        return -1;
    }

    out.clear();
    for (const auto& r : rows) {
        if (r.size() < RECORD_COL_COUNT) {
            continue;
        }
        SyncRecordTab rec = {};
        rec.id = std::stoul(r[ID]);
        rec.pid = std::stoul(r[PID]);
        rec.rankId = std::stoul(r[RANK_ID]);
        rec.rankSize = std::stoul(r[RANK_SIZE]);
        rec.syncIter = std::stoul(r[SYNC_ITER]);
        rec.streamId = std::stoull(r[STREAM_ID]);
        rec.status = static_cast<uint8_t>(std::stoul(r[STATUS]));
        out.push_back(std::move(rec));
    }
    return 0;
}

int QueryCcuInstrResAll(std::vector<CcuInstrResTab>& out)
{
    enum : int { ID, DEVICE_ID, RANK_ID, DIE_ID, INSTR_COUNT, INSTR_SPACE };
    const size_t CCU_COL_COUNT = 6;
    const std::string sql = "SELECT id, deviceId, rankId, dieId, instrCount, instrSpace FROM ccuInstrRes";
    ExRows rows;
    if (OpDbOps::Instance().ExecQueryEx(sql, {}, rows) != 0) {
        return -1;
    }

    out.clear();
    for (const auto& r : rows) {
        if (r.size() < CCU_COL_COUNT) {
            continue;
        }
        CcuInstrResTab rec = {};
        rec.id = std::stoul(std::get<std::string>(r[ID]));
        rec.deviceId = std::stoul(std::get<std::string>(r[DEVICE_ID]));
        rec.rankId = std::stoul(std::get<std::string>(r[RANK_ID]));
        rec.dieId = std::stoul(std::get<std::string>(r[DIE_ID]));
        rec.instrCount = std::stoul(std::get<std::string>(r[INSTR_COUNT]));
        const auto& blob = std::get<std::vector<uint8_t>>(r[INSTR_SPACE]);
        std::memcpy(rec.instrSpace, blob.data(), std::min(blob.size(), sizeof(rec.instrSpace)));
        out.push_back(std::move(rec));
    }
    return 0;
}
} // namespace sim
