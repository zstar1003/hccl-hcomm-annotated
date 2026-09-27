/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_RES_H
#define HCOMM_RES_H

#include "hcomm_res_defs.h"
#include "hcomm_channel.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

extern HcommResult HcommEndpointCreate(const EndpointDesc* endpoint, EndpointHandle* endpointHandle);

extern HcommResult HcommEndpointDestroy(EndpointHandle endpointHandle);

/**
 * @brief 查询指定设备上的Endpoint描述符数量
 * @param[in] deviceLogicId NPU设备逻辑ID
 * @param[out] descNum 返回的Endpoint描述符数量
 * @return HcommResult 执行结果状态码
 * @note 典型用法是先调用本接口获取数量，再按该数量申请数组并调用HcommEndpointGetDescs。
 */
extern HcommResult HcommEndpointGetDescNum(int32_t deviceLogicId, uint32_t* descNum);

/**
 * @brief 获取指定设备上的Endpoint描述符
 * @param[in] deviceLogicId NPU设备逻辑ID
 * @param[in,out] descNum 入参为endpointDescs数组容量，出参为实际填充的描述符数量
 * @param[out] endpointDescs 调用方申请的Endpoint描述符数组
 * @return HcommResult 执行结果状态码；数组容量小于实际Endpoint数量时返回参数错误
 * @note 推荐按HcommEndpointGetDescNum返回的数量申请数组，并将数组实际容量通过descNum传入。
 */
extern HcommResult HcommEndpointGetDescs(int32_t deviceLogicId, uint32_t* descNum, EndpointDesc* endpointDescs);

extern HcommResult
HcommMemReg(EndpointHandle endpointHandle, const char* memTag, const CommMem* mem, HcommMemHandle* memHandle);

extern HcommResult HcommMemUnreg(EndpointHandle endpointHandle, HcommMemHandle memHandle);

extern HcommResult
HcommMemExport(EndpointHandle endpointHandle, HcommMemHandle memHandle, void** memDesc, uint32_t* memDescLen);

extern HcommResult
HcommMemImport(EndpointHandle endpointHandle, const void* memDesc, uint32_t descLen, CommMem* outMem);

extern HcommResult HcommMemUnimport(EndpointHandle endpointHandle, const void* memDesc, uint32_t descLen);

extern HcommResult
HcommThreadAlloc(CommEngine engine, uint32_t threadNum, const uint32_t* notifyNumPerThread, ThreadHandle* threads);

extern HcommResult HcommThreadFree(const ThreadHandle* threads, uint32_t threadNum);

extern HcommResult HcommEndpointGetListenPort(EndpointHandle endpointHandle, uint32_t* port);

extern HcommResult HcommThreadResGetInfo(ThreadHandle thread, ThreadResType resType, uint32_t infoLen, void** info);

extern HcommResult
HcommEndpointCheckFeature(HcommEndpointFeatureType featureType, const EndpointDesc* endpointDesc, bool* value);

extern HcommResult HcommMemAlloc(void** ptr, size_t size);

extern HcommResult HcommMemFree(void* ptr);

#ifdef __cplusplus
}
#endif // __cplusplus
#endif // HCOMM_RES_H
