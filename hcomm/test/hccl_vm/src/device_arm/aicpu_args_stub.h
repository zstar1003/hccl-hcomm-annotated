/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AICPU_ARGS_STUB_H
#define AICPU_ARGS_STUB_H

#include <cstdint>

constexpr uint32_t HCOMID_MAX_SIZE = 128;

constexpr uint32_t P2P_MAX_ARG_SIZE_STUB = 8192;

// HcclP2pKernelParam
struct HcclP2pKernelParamStub {
    uint64_t sendRecvThread;
    uint8_t opParams[P2P_MAX_ARG_SIZE_STUB];
};

struct ThreadNotifyRecordParam {
    uint64_t thread;
    uint64_t dstThread;
    uint32_t dstNotifyIdx;
};

struct ThreadNotifyWaitParam {
    uint64_t thread;
    uint32_t notifyIdx;
};

struct HDCommunicateParams {
    uint64_t hostAddr{0};
    uint64_t deviceAddr{0};
    uint64_t readCacheAddr{0};
    uint32_t devMemSize{0};
    uint32_t buffLen{0};
    uint32_t flag{0};
};

struct DevAicpuCommConfig {
    bool taskExceptionEnable{true};
    uint32_t notifyWaitTimeout{1836};
    // 如要新增配置类字段，在此处添加
};
struct CommAicpuParam {
    char hcomId[HCOMID_MAX_SIZE];
    int32_t deviceLogicId;
    uint32_t devicePhyId;
    uint32_t deviceType;
    uint32_t userRankSize;
    uint32_t userRank;
    HDCommunicateParams kfcControlTransferH2DParams;
    HDCommunicateParams kfcStatusTransferD2HParams;
    DevAicpuCommConfig commConfig; // 收编通信域配置类变量
};

#endif
