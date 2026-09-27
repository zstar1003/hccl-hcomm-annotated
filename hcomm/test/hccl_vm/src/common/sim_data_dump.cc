/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "sim_data_dump.h"

#include <chrono>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <json.hpp>
#include <sstream>
#include <string>
#include <unistd.h>

#include "store_binary_data_operator.h"
#include "sim_binary_data_type_pub.h"
#include "cmd_base_utils.h"
#include "store_dump_shm_data.h"
#include "hccl/hccl_types.h"
#include "sim_log.h"
#include "sim_models.h"
#include "db_sim_runner_db.h"

static const std::string TASK_COLLECTION_FILE = "/%s_task.jsonl.gz";
static const std::string MEM_LAYOUT_FILE = "/%s_mem_layout.jsonl.gz";
static const std::string MODEL_FILE = "/%s_model.jsonl.gz";

HcclVmResult DumpData(nlohmann::json& j)
{
    j["status"] = "finish";
    j["data_id"] = "1234";
    return HcclVmResult::HCCL_SIM_SUCCESS;
}
