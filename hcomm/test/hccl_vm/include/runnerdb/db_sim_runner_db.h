/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SIM_RUNNER_DB_H
#define SIM_RUNNER_DB_H

#include <functional>
#include <optional>
#include <vector>

#include "sim_models.h"
#include "sim_runner_impl.inl"
namespace RunnerDB {
template <typename T>
uint64_t Add(T& rec);

template <typename T>
std::optional<T> GetById(uint64_t id);

template <typename T>
std::vector<T> GetByPred(std::function<bool(const T&)> pred);

template <typename T>
std::pair<T, bool> GetOneByPred(std::function<bool(const T&)> pred);

template <typename T>
bool Update(uint64_t id, std::function<void(T&)> updater);

template <typename T>
bool Delete(uint64_t id);

template <typename T>
bool DeleteAll();

std::vector<std::string> GetAllTableName();
} // namespace RunnerDB
#endif
