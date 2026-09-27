/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_DEVICE_PUB_H
#define HCCL_DEVICE_PUB_H

#include "sim_common_defs.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

constexpr uint32_t HCCL_SQE_SIZE = 64;
constexpr uint32_t HCCL_WQE_SIZE = 64;
constexpr uint32_t HCCL_SQE_MAX_CNT = 2048;

HcclSim::HcclVmResult SetCurRankId(uint32_t rankId);

HcclSim::HcclVmResult GetCurRankId(uint32_t* rankId);

void SetCurDeviceKey(uint32_t deviceKey);

uint32_t GetCurDeviceKey();

HcclSim::HcclVmResult GetSqBufferAddr(uint8_t** sqBuff);

HcclSim::HcclVmResult GetPiValByJettyId(uint32_t jettyId, uint32_t* piValue);

HcclSim::HcclVmResult UpdatePiValByJettyId(uint32_t jettyId, uint32_t piValue);

uint64_t TransLocalAddrToVirtual(uint64_t devAddr);

uint64_t TransRemoteAddrToVirtualByRank(uint64_t devAddr, uint32_t rankId);

uint32_t GetRankIdByDevAddr(uint64_t devAddr);

uint32_t GetRankIdByIpAddr(std::string ipAddr);

uint64_t GetDevMapperAddrByDevAddrImpl(uint64_t devAddr, const char* file, int line);

#define GetDevMapperAddrByDevAddr(devAddr) GetDevMapperAddrByDevAddrImpl(devAddr, __FILE__, __LINE__)

void UpdataKfcStatus(uint64_t d2hAddr);

void RegisterSignalHandler();

uint32_t GetSqTail(uint32_t sqId);

void UpdateSqTail(uint32_t sqId, uint32_t newTail);

bool GetWqebufferByJettyId(uint64_t jettyId, uint64_t& wqeBuffer);

void InitPipeFds(int h2dReadFd, int d2hWriteFd);

int DeviceSendMsg(uint8_t cmd, const void* data, uint32_t dataLen);

int DeviceRecvMsg(uint8_t& outCmd, void* outData, uint32_t maxLen, uint32_t& outDataLen);

#ifdef __cplusplus
}
#endif

#endif
