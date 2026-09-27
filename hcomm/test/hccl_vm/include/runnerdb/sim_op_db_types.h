/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SIM_OP_DB_TYPES_H
#define SIM_OP_DB_TYPES_H

#include <string>
#include <vector>

namespace sim {
enum class DbType {
    SQLITE3,
};
struct DBConfig {
    DbType type = DbType::SQLITE3;
    std::string dbPath = "data/hccl_vm_data.db";
    std::string host;
    uint16_t port = 0;
    std::string user;
    std::string password;
    std::string database;
};

struct OpDetailTab {
    uint32_t id;       // PK (自增)
    uint32_t pid;      // proxy 进程 ID
    uint32_t rankId;   // 设备 ID
    uint32_t opIter;   // 算子迭代次数 (从 0 开始)
    uint32_t syncIter; // 所属 sync 周期 (从 0 开始)

    uint64_t streamId;              // streamId
    uint32_t root;                  // root
    uint32_t opExpansionMode;       // 算子执行模式 0: ccu, 1: aicpu, 2: aiv
    uint32_t devType;               // 设备类型
    uint32_t rankSize;              // ransize
    uint32_t srcRank;               // srcRank
    uint32_t dstRank;               // dstRank
    std::vector<uint8_t> opDetail;  // BLOB
    std::vector<uint8_t> opExtInfo; // BLOB
};

struct OpMemInfoTab {
    uint32_t id;         // PK (自增)
    uint32_t opDetailId; // FK，关联 OpDetailTab
    uint64_t inputAddr;
    uint64_t inputSize;
    uint64_t outputAddr;
    uint64_t outputSize;
    uint64_t cclAddr;
    uint64_t cclSize;
};

struct SyncRecordTab {
    uint32_t id;  // PK (自增)
    uint32_t pid; // proxy 进程 ID
    uint32_t rankId;
    uint32_t rankSize;
    uint32_t syncIter; // sync 迭代次数 (从 0 开始)
    uint64_t streamId; // streamId
    uint8_t status;    // 0=pending, 2=completed
};

struct OpTaskTab {
    uint32_t id;                     // PK (自增)
    uint32_t opDetailId;             // FK，关联 OpDetailTables
    uint32_t taskSeq;                // 任务序号
    std::vector<uint8_t> optaskMeta; // BLOB，动态大小
};

struct CcuChannelTab {
    uint32_t id;        // PK (自增)
    uint32_t channelId; // ccuChannelId
    uint32_t srcDieId;
    uint32_t dstDieId;
    uint32_t srcRankId;
    uint32_t dstRankId;
    uint8_t leid[16];
    uint8_t reid[16];
    uint16_t protocol;
    uint16_t jettyNum;
    uint32_t jettyId[64]; // BLOB，动态大小 最大64个
};

struct JettyMapTab {
    uint32_t id;         // PK (自增)
    uint32_t opDetailId; // FK，关联 OpDetailTab
    uint32_t srcDieId;
    uint32_t dstDieId;
    uint32_t srcRankId;
    uint32_t dstRankId;
    uint8_t leid[16];
    uint8_t reid[16];
    uint16_t protocol;
};

struct CcuInstrResTab {
    uint32_t id; // PK (自增)
    uint32_t deviceId;
    uint32_t rankId;
    uint32_t dieId;
    uint32_t instrCount;
    uint8_t instrSpace[32 * 1024][32]; // 指令空间1M: 32K个指令
};

struct CcuInstrTab {
    uint32_t id; // PK (自增)
    uint32_t ccuInstrResId;
    uint32_t rankId;
    uint32_t startId;
    uint32_t instrInfoSize;
};

// 定义包含单卡完整信息的复合结构体
struct CompositeOpDetail {
    uint32_t rankId; // 关键键值，用于 Map 排序
    OpDetailTab detail;
    OpMemInfoTab memInfo;
    std::vector<OpTaskTab> tasks;
};
} // namespace sim
#endif
