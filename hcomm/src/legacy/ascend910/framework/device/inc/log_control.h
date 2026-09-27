/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef LOG_CONTROL_H
#define LOG_CONTROL_H

#include "log.h"

// 用于将特定场景下的ERROR日志，转换为RUN_WARNING日志
class LogControl {
public:
    // 构造时初始化initState，并将当前日志级别设置为targetState
    LogControl(bool initState, bool targetState) : initState_(initState) { SetErrToWarnSwitch(targetState); }

    // 析构时恢复为initState
    ~LogControl() { SetErrToWarnSwitch(initState_); }

private:
    bool initState_ = false;
};

#endif // LOG_CONTROL_H
