/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_LAUNCH_H
#define HCCL_LAUNCH_H

#include <stdint.h>
#include <hccl/hccl_types.h>
#include <acl/acl_rt.h>
#include "hcomm_res_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

const uint32_t P2P_MAX_ARG_SIZE = 8192U;
typedef struct {
    ThreadHandle sendRecvThread;
    uint8_t opParams[P2P_MAX_ARG_SIZE];
} HcclP2pKernelParam;

typedef struct {
    void* buffer;
    uint8_t reserved[8];
    HcclCMDType cmdType;
    HcclDataType dataType;
    uint64_t count;
    uint32_t remoteRank;
    void* unfoldStream;
} HcclOpP2pDesc;

const uint32_t HCCL_OP_DESC_OP_NAME_MAX_LEN = 256;

typedef struct {
    CommAbiHeader header;
    uint32_t opDescType;
    char opName[HCCL_OP_DESC_OP_NAME_MAX_LEN];
    union {
        uint8_t raws[76];
        HcclOpP2pDesc p2p;
    };
} HcclOpDesc;

const uint32_t HCCL_OPDESC_MAGIC_WORD = 0x0f0f0f0f;
const uint32_t HCCL_OPDESC_VERSION = 1;

/**
 * @brief 初始化HcclOpDesc结构体
 *
 * @param[inout] HcclOpDesc 返回的算子描述参数
 * @param[in] descNum 描述数量
 * @return HcclResult 执行结果状态码
 */
static inline HcclResult HcclOpDescInit(HcclOpDesc* opDesc)
{
    if (opDesc != nullptr) {
        // 先用0xFF填充整个结构体
        (void)memset_s(opDesc, sizeof(HcclOpDesc), 0xFF, sizeof(HcclOpDesc));

        // 初始化ABI头信息
        opDesc->header.version = HCCL_OPDESC_VERSION;
        opDesc->header.magicWord = HCCL_OPDESC_MAGIC_WORD;
        opDesc->header.size = sizeof(HcclOpDesc);
        opDesc->header.reserved = 0;
    } else {
        return HCCL_E_PTR;
    }
    return HCCL_SUCCESS;
}

const uint32_t HCCL_KERNEL_SO_NAME_MAX_LEN = 256;
const uint32_t HCCL_KERNEL_FUNC_NAME_MAX_LEN = 256;

typedef struct {
    char kernelSoName[HCCL_KERNEL_SO_NAME_MAX_LEN];
    char kernelFuncName[HCCL_KERNEL_FUNC_NAME_MAX_LEN];
    void* args;
    uint32_t argSize;
} HcclKernelFuncInfo;

const uint32_t HCCL_KERNEL_LAUNCH_CFG_MAGIC_WORD = 0x0f0f0f0f;
const uint32_t HCCL_KERNEL_LAUNCH_CFG_VERSION = 1;

typedef struct {
    CommAbiHeader header;
    uint64_t timeOut;
    uint8_t reserved[104];
} HcclKernelLaunchCfg;

/**
 * @brief 初始化HcclKernelLaunchCfg结构体
 *
 * @param[inout] HcclKernelLaunchCfg 返回的KernelLaunch配置参数
 * @return HcclResult 执行结果状态码
 */
static inline HcclResult HcclKernelLaunchCfgInit(HcclKernelLaunchCfg* kernelLaunchCfg)
{
    if (kernelLaunchCfg != nullptr) {
        // 先用0xFF填充整个结构体
        (void)memset_s(kernelLaunchCfg, sizeof(HcclKernelLaunchCfg), 0xFF, sizeof(HcclKernelLaunchCfg));

        // 初始化ABI头信息
        kernelLaunchCfg->header.version = HCCL_KERNEL_LAUNCH_CFG_VERSION;
        kernelLaunchCfg->header.magicWord = HCCL_KERNEL_LAUNCH_CFG_MAGIC_WORD;
        kernelLaunchCfg->header.size = sizeof(HcclKernelLaunchCfg);
        kernelLaunchCfg->header.reserved = 0;
    } else {
        return HCCL_E_PTR;
    }
    return HCCL_SUCCESS;
}

/**
 * @brief Launch AICPU kernel for P2P task
 *
 * @param comm Communication domain handle
 * @param opInfo Operator descriptor parameters
 * @param funcInfo Kernel function info (dynamic library name, kernel function name, args and argSize)
 * @param aicpuThreadHandle AICPU communication main stream thread handle
 * @param userStream User host main stream
 * @param HcclKernelLaunchCfg kernel launch cfg
 * @return HcclResult
 *   HCCL_SUCCESS: success
 *   HCCL_E_NOT_SUPPORT: HcclGroupStart not called
 *   HCCL_E_INTERNAL: internal error (async task failure, communication task failure)
 */

extern HcclResult HcclAicpuKernelLaunch(
    HcclComm comm, const HcclOpDesc* opInfo, const HcclKernelFuncInfo* funcInfo, ThreadHandle aicpuThreadHandle,
    aclrtStream userStream, const HcclKernelLaunchCfg* kernelLaunchCfg);

#ifdef __cplusplus
}
#endif

#endif
