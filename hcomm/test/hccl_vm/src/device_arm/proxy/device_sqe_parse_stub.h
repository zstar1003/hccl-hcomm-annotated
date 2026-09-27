/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DEVICE_SQE_PARSE_STUB_H
#define DEVICE_SQE_PARSE_STUB_H

#include "ascend_hal.h"
#include "hccl/hccl_types.h"
#include "hccl_device_pub.h"
#include "runtime/rt.h"

void ParseA5SqeFromSqBuffer(uint32_t devId, struct halSqCqConfigInfo* info);

void ParseDavidSDMASqe(uint32_t streamId, void* sqeBuf);

void ParseDavidNotifySqe(uint32_t streamId, void* sqeBuf, bool isPost);

void ParseDavidUDMASqe(uint32_t streamId, void* sqeBuf);

void ParseDavidUBReadWriteSqe(uint64_t wqeAddr, uint16_t streamId, uint32_t jettyId, bool isRead);

void ParseDavidUBWriteWithNotifySqe(uint64_t wqeAddr, uint16_t streamId, uint32_t jettyId);

uint64_t GetFull64BitAddr(uint32_t lowAddr, uint32_t highAddr);

HcclReduceOp ParseReduceTypeDavid(uint8_t result);

HcclDataType ParseDataTypeDavid(uint8_t result);

HcclReduceOp ParseUbReduceTypeDavid(uint32_t type);

HcclDataType ParseUbDataTypeDavid(uint32_t type);

void PrintTaskMetaData(const HcclTaskMetaData& taskMeta);

uint32_t GetRmtRankIdByEid(uint32_t eid);

#endif // DEVICE_SQE_PARSE_STUB_H
