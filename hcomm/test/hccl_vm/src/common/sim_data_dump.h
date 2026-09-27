/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SIM_DATA_DUMP_H
#define SIM_DATA_DUMP_H

#include <nlohmann_json/json.hpp>
#include <string>

#include "sim_common_defs.h"

using namespace HcclSim;

HcclVmResult DumpData(nlohmann::json& j);

HcclVmResult DumpModel(std::string dataId);
HcclVmResult DumpMemLayout(std::string dataId);
HcclVmResult DumpTask(std::string dataId);

#endif
