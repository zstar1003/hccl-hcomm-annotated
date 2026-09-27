/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RES_TE_H
#define RES_TE_H

#include <map>
#include <string>
#include "res_pub.h"

namespace Hccl {

inline const std::map<u8, std::string>& GetTaskConciseNameMap()
{
    static const std::map<u8, std::string> taskConciseNameMap = {
        {static_cast<u8>(TaskParamTypeVal::TASK_SDMA), "M"},
        {static_cast<u8>(TaskParamTypeVal::TASK_RDMA), "RS"},
        {static_cast<u8>(TaskParamTypeVal::TASK_SEND_PAYLOAD), "SP"},
        {static_cast<u8>(TaskParamTypeVal::TASK_REDUCE_INLINE), "IR"},
        {static_cast<u8>(TaskParamTypeVal::TASK_UB_REDUCE_INLINE), "IR"},
        {static_cast<u8>(TaskParamTypeVal::TASK_REDUCE_TBE), "R"},
        {static_cast<u8>(TaskParamTypeVal::TASK_UB), "WorR"},
        {static_cast<u8>(TaskParamTypeVal::TASK_NOTIFY_RECORD), "NR"},
        {static_cast<u8>(TaskParamTypeVal::TASK_NOTIFY_WAIT), "NW"},
        {static_cast<u8>(TaskParamTypeVal::TASK_SEND_NOTIFY), "SN"},
        {static_cast<u8>(TaskParamTypeVal::TASK_WRITE_WITH_NOTIFY), "WN"},
        {static_cast<u8>(TaskParamTypeVal::TASK_UB_INLINE_WRITE), "IW"},
        {static_cast<u8>(TaskParamTypeVal::TASK_WRITE_REDUCE_WITH_NOTIFY), "WRN"},
        {static_cast<u8>(TaskParamTypeVal::TASK_CCU), "CCU"},
        {static_cast<u8>(TaskParamTypeVal::TASK_AICPU_KERNEL), "AIK"},
    };
    return taskConciseNameMap;
}

} // namespace Hccl

#endif
