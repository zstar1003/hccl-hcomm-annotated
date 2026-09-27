/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DB_HCCL_OP_DB_OPS_H
#define DB_HCCL_OP_DB_OPS_H

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "sim_common_defs.h"
#include "db_hccl_db.h"
#include "sim_op_db_types.h"

namespace HcclSim {
namespace DB {
    class OpDbOps {
    public:
        static OpDbOps& Instance();

        int SetDbConfig(sim::DBConfig& config);

        void SetDB(std::unique_ptr<HcclDB> db);

        HcclDB* GetDB() const;

        HcclVmResult InitSchema();

        int ExecInsert(const std::string& sql, const std::vector<Value>& params, uint32_t& outId);

        int ExecInsertNoId(const std::string& sql, const std::vector<Value>& params);

        int ExecQuery(
            const std::string& sql, const std::vector<Value>& params, std::vector<std::vector<std::string>>& rows);

        int ExecUpdate(const std::string& sql, const std::vector<Value>& params);

        int QuerySingleBlob(const std::string& sql, const std::vector<Value>& params, std::vector<uint8_t>& outBlob);

        int QueryBlobColumns(
            const std::string& sql, const std::vector<Value>& params, std::vector<std::vector<uint8_t>>& outBlobs);

        int ExecQueryEx(const std::string& sql, const std::vector<Value>& params, ExRows& rows);

        HcclVmResult Backup(const std::string& destPath);

        int RunInTransaction(std::function<int()> fn);

    private:
        OpDbOps();
        ~OpDbOps();
        OpDbOps(const OpDbOps&) = delete;
        OpDbOps& operator=(const OpDbOps&) = delete;

        std::unique_ptr<HcclDB> m_db;
    };
} // namespace DB
} // namespace HcclSim

#endif
