/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SIM_RUNNER_IMPL_INL
#define SIM_RUNNER_IMPL_INL

#include "db_sim_sqlite_db.h"
#include <optional>
#include "db_sim_runner_db.h"

namespace RunnerDB {

template <typename T>
uint64_t Add(T& rec)
{
    return SimRunnerSqliteDB::Instance().Add<T>(rec);
}

template <typename T>
std::optional<T> GetById(uint64_t id)
{
    return SimRunnerSqliteDB::Instance().Find<T>(id);
}

template <typename T>
std::vector<T> GetByPred(std::function<bool(const T&)> pred)
{
    return SimRunnerSqliteDB::Instance().QueryList<T>(pred);
}

template <typename T>
std::pair<T, bool> GetOneByPred(std::function<bool(const T&)> pred)
{
    return SimRunnerSqliteDB::Instance().Query<T>(pred);
}

template <typename T>
bool Update(uint64_t id, std::function<void(T&)> updater)
{
    return SimRunnerSqliteDB::Instance().Update<T>(id, updater);
}

template <typename T>
bool Delete(uint64_t id)
{
    return SimRunnerSqliteDB::Instance().Delete<T>(id);
}

template <typename T>
bool DeleteAll()
{
    return SimRunnerSqliteDB::Instance().DeleteAll<T>();
}
} // namespace RunnerDB

#endif
