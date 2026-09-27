/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_KERNEL_PARAM_LITE_H
#define HCCLV2_KERNEL_PARAM_LITE_H

#include "coll_operator.h"
#include "dev_type.h"
#include "hdc_param.h"

constexpr u32 MAX_OP_TAG_LEN = 191; // 最大的tag 长度, 和对外接口保持一致
constexpr u32 MAX_NAME_LEN = 64;

struct HcclAicpuLocBufLite {
    uint64_t addr{0};
    uint64_t size{0};
    uint32_t tokenId;
    uint32_t tokenValue;
};

struct SendRecvItemTokenInfo {
    uint32_t tokenId;
    uint32_t tokenValue;
};

struct HcclAicpuCommunicatorLite {
    uint32_t idIndex;
    uint32_t myRank;
    uint32_t rankSize;
    Hccl::DevType devType;
    uint32_t devPhyId;
    HcclAicpuLocBufLite opBaseScratch;
    uint64_t opCounterAddr;
    char commId[COMM_NAME_MAX_LENGTH]{0};
    u32 opIndex_{0};
};

struct HcclAicpuOpLite {
    Hccl::CollAlgOperator algOperator;
    uint32_t sendRecvRemoteRank;
    HcclAicpuLocBufLite input;
    HcclAicpuLocBufLite output;
    HcclAicpuLocBufLite scratch; // used for offload op
    void* batchPutGetLocalAddr{nullptr};
    void* batchPutGetRemoteAddr{nullptr};
    uint32_t batchPutGetDescNum{0};
    uint32_t userStreamId{0};
};

struct HcclDeviceEnvConfigLite {
    u32 hcclExecTimeout{1080};
    bool taskExceptionEnable{true};
};

struct BatchSendRecvDataDes {
    u32 itemNum;
    HcclSendRecvItem batchSendRecvItem[];
};

struct AllToAllvDataDes {
    u8 sendType;
    u8 recvType;
    u64 sendRecvInfos[];
};

struct AllToAllvcDataDes {
    u8 sendType;
    u8 recvType;
    u64 sendCountMatrix[];
};

struct HcclKernelParamLite {
    uint64_t binaryResAddr{0};
    uint64_t binaryResSize{0};
    HcclAicpuCommunicatorLite comm;
    HcclAicpuOpLite op;
    char algName[MAX_NAME_LEN]{0};
    bool oneSidedComm{false};
    char opTag[MAX_OP_TAG_LEN]{0};
    char tagKey[MAX_OP_TAG_LEN]{0};
    Hccl::HDCommunicateParams kfcControlTransferH2DParams;
    Hccl::HDCommunicateParams kfcControlTransferD2HParams;
    HcclDeviceEnvConfigLite envConfig;
};

constexpr u32 KERNEL_PARAM_NAME_SIZE = 32;
struct HcclKernelLaunchParam {
    HcclKernelParamLite kernel;
    char soName[32] = "libccl_kernel.so";
    char kernelName[KERNEL_PARAM_NAME_SIZE] = "HcclKernelEntrance";
    char opName[32] = "LoadWithOpBasedMode";
};
#endif
