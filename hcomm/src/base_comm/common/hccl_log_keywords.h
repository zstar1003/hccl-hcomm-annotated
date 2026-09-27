/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_LOG_KEYWORDS_H
#define HCCL_LOG_KEYWORDS_H

#include <string>

/* 对关键报错日志提供多级检索关键字 */
/* 一级检索关键字 */
const std::string LOG_KEYWORDS_TASK_EXEC = "TaskExecStage";   // 算子执行阶段异常
const std::string LOG_KEYWORDS_INIT_GROUP = "InitGroupStage"; // 通信域初始化阶段异常
const std::string LOG_KEYWORDS_INIT_CHANNEL = "InitChannelStage";
const std::string LOG_KEYWORDS_LINK_INFO = "LinkInfo";

/* 二级检索关键字 */
const std::string LOG_KEYWORDS_TIMEOUT = "Timeout";                      // 算子执行阶段超时
const std::string LOG_KEYWORDS_RUN_FAILED = "RunFailed";                 // 算子执行阶段失败，如SDMA ERROR
const std::string LOG_KEYWORDS_HEARTBEAT_EVENT = "HeartbeatAbnormal";    // 算子执行阶段心跳异常事件
const std::string LOG_KEYWORDS_ENV_CONFIG = "EnvConfig";                 // 环境变量配置异常
const std::string LOG_KEYWORDS_RANKTABLE_CONFIG = "RanktableConfig";     // ranktable读取失败
const std::string LOG_KEYWORDS_RANKTABLE_CHECK = "RanktableCheck";       // ranktable校验失败
const std::string LOG_KEYWORDS_RANKTABLE_DETECT = "RanktableDetect";     // ranktable协商失败
const std::string LOG_KEYWORDS_PARAMETER_CONFLICT = "ParameterConflict"; // 参数不一致
const std::string LOG_KEYWORDS_VERSION_CONFLICT = "VersionConflict";     // HCCL版本不一致
const std::string LOG_KEYWORDS_INVALID_ARGUMENT = "InvalidArgument";     // 外部入参非法
const std::string LOG_KEYWORDS_RESOURCE = "Resource";                    // 资源初始化失败
const std::string LOG_KEYWORDS_NOT_SUPPORTED = "Not Supported";

/* 三级检索关键字 */
const std::string LOG_KEYWORDS_HOST = "HOST";
const std::string LOG_KEYWORDS_HOST_TS = "HOST_TS";
const std::string LOG_KEYWORDS_AIV = "AIV";
const std::string LOG_KEYWORDS_AICPU = "AICPU";
const std::string LOG_KEYWORDS_CCU = "CCU";
const std::string LOG_KEYWORDS_CQE_ERROR = "CQE ERROR";

/* 通信域及本卡信息关键字 */
const std::string LOG_KEYWORDS_COMMUNICATOR = "Communicator Key Info";
const std::string LOG_KEYWORDS_LOCALRANK = "LocalRank Key Info";

#endif // HCCL_LOG_KEYWORDS_H
