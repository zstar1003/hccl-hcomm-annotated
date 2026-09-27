/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BKF_CH_CLI_LETCP_H
#define BKF_CH_CLI_LETCP_H
#include "bkf_comm.h"
#include "bkf_ch_cli_adef.h"

#ifdef __cplusplus
extern "C" {
#endif
#pragma pack(4)
/**
 * @brief 构造linux tcp传输层客户端虚表
 *
 * @param[in] *name 注入虚表名称
 * @param[in] *temp 注入虚表结构体指针，堆变量或栈变量均可
 * @return 构造结果
 *   @retval VOS_NULL 失败
 *   @retval 其他 成功
 */
uint32_t BkfChCliLetcpBuildVTbl(char *name, BkfChCliTypeVTbl *cliVTbl, uint32_t nameLen);

#pragma pack()
#ifdef __cplusplus
}
#endif

#endif
