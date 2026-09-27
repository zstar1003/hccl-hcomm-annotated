/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "db_hccl_db_factory.h"

#include "db_hccl_db_sqlite.h"
#include "sim_log.h"

namespace HcclSim {
namespace DB {
    HcclDBFactory::HcclDBFactory() { RegisterBuiltInTypes(); }

    void HcclDBFactory::RegisterBuiltInTypes()
    {
        RegisterDB(sim::DbType::SQLITE3, []() -> std::unique_ptr<HcclDB> {
            return std::make_unique<HcclDBSqlite>();
        });
    }
} // namespace DB
} // namespace HcclSim
