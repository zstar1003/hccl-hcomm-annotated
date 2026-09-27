/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SIM_OP_DB_OPS_H
#define SIM_OP_DB_OPS_H

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "sim_op_db_types.h"

namespace sim {
int InitOpDataDb();
int SetDbConfig(DBConfig& config);

int InsertOpDetail(OpDetailTab& rec);
int InsertOpMem(OpMemInfoTab& rec);
int InsertOpDetailAndMem(OpDetailTab& detail, OpMemInfoTab& mem);
int InsertCcuChannel(CcuChannelTab& rec);
int InsertJettyMap(JettyMapTab& rec);
int InsertOpTask(OpTaskTab& rec, bool isDevice = false);
int InsertSyncRecord(SyncRecordTab& rec);
int InsertCcuInstrRes(CcuInstrResTab& rec);
int InsertCcuInstr(CcuInstrTab& rec);

int UpdateAndInsertByCcuId(
    uint64_t& ccuId, uint32_t deviceId, uint32_t rankId, uint32_t dieId, uint32_t instrCount, uint32_t instrOffset,
    uint32_t instrInfoSize, const void* instrInfo);
int UpdateSyncRecordStatus(std::vector<SyncRecordTab>& syncRecord);
int UpdateOpMemCclBuffer(uint64_t cclAddr, uint64_t cclSize);
int UpdateOpExpansionMode(uint8_t mode);

int QueryLatestOpExpansionMode();
int QueryCcuChannelAll(std::vector<CcuChannelTab>& out);
int QueryJettyMapAll(std::vector<JettyMapTab>& out);
int QuerySyncRecordAll(std::vector<SyncRecordTab>& out);
int QueryCcuInstrResAll(std::vector<CcuInstrResTab>& out);
int QueryNewestOpDeatailIdByPid(uint64_t pid, uint32_t& OpDetailId);
int QueryCurrentOpMemInfoByRank(uint32_t rankId, OpMemInfoTab& out);
int QuerySyncRecordByStatus(uint8_t status, std::vector<SyncRecordTab>& out);
int QueryCompositeOpDetailBySyncIter(uint32_t syncIter, std::map<uint32_t, std::vector<CompositeOpDetail>>& detail);
} // namespace sim

#endif
