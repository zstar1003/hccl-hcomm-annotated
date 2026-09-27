/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_ONE_SIDED_SERVICES_H
#define HCCL_ONE_SIDED_SERVICES_H

#include <hccl/hccl_types.h>
#include <hccl/base.h>
#include "hcomm_res_defs.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

const u32 HCCL_MEM_DESC_LENGTH = 511;

typedef struct {
    char desc[HCCL_MEM_DESC_LENGTH + 1]; // 具体内容对调用者不可见
} HcclMemDesc;

typedef struct {
    HcclMemDesc* array;
    u32 arrayLength;
} HcclMemDescs;

typedef struct {
    void* localAddr;  // 本端VA
    void* remoteAddr; // 远端VA
    u64 count;
    HcclDataType dataType;
} HcclOneSideOpDesc;

typedef enum {
    HCCL_TOPO_FULLMESH = 0, // fullmesh连接
    HCCL_TOPO_NUM,
} HcclTopoType;

typedef struct {
    HcclTopoType topoType;
    u64 rsvd0;
    u64 rsvd1;
    u64 rsvd2;
} HcclPrepareConfig;

/**
 * @brief comm粒度注册内存
 *
 * @param comm [input]A pointer identifying the communication resource target
 * @param remoteRank [input]本内存希望共享给的对端rank
 * @param type [input]注册的内存类型，host/device
 * @param addr [input]内存的VA地址
 * @param size [input]内存size，以字节为单位
 * @param desc [output]内存信息描述符
 * @return HcclResult
 */
extern HcclResult HcclRegisterMem(HcclComm comm, u32 remoteRank, int type, void* addr, u64 size, HcclMemDesc* desc);

/**
 * @brief comm粒度注销内存
 *
 * @param comm [input]A pointer identifying the communication resource target
 * @param desc [input]内存信息描述符
 * @return HcclResult
 */
extern HcclResult HcclDeregisterMem(HcclComm comm, HcclMemDesc* desc);

/**
 * @brief 与{comm, remoteRank}指示的对端交互单边操作的memory描述符
 *
 * @param comm [input]A pointer identifying the communication resource target
 * @param remoteRank [input]对端rank_id
 * @param local [input]本端用于单边通信的内存块描述符信息
 * @param timeout
 * [input]超时时间，预留以后使用。当前使用通信库内部的超时时间，故该参数不使用，HCCL不做检查，调用者设置为0即可
 * @param remote [output]用于接收对端共享的内存块描述符信息
 * @param actualNum [output]实际的对端交换的MemDesc数量
 * @return HcclResult
 */
extern HcclResult HcclExchangeMemDesc(
    HcclComm comm, u32 remoteRank, HcclMemDescs* local, int timeout, HcclMemDescs* remote, u32* actualNum);

/**
 * @brief 使能一个memDesc，使本端具有访问对端memDesc对应的远端memory的能力
 *
 * @param comm [input]A pointer identifying the communication resource target
 * @param remoteMemDesc [input]远端用于单边通信的内存块描述符
 * @param remoteMem [output]描述符对应的对端内存
 * @return HcclResult
 */
extern HcclResult HcclEnableMemAccess(HcclComm comm, HcclMemDesc* remoteMemDesc, CommMem* remoteMem);

/**
 * @brief 去使能一个memDesc，使本端不再具有访问对端memDesc对应的远端memory的能力
 *
 * @param comm [input]A pointer identifying the communication resource target
 * @param remoteMemDesc [input]远端用于单边通信的内存块描述符
 * @return HcclResult
 */
extern HcclResult HcclDisableMemAccess(HcclComm comm, HcclMemDesc* remoteMemDesc);

/**
 * @brief 发起批量单边put操作，将本地数据写到{comm, remoteRank}指示的远端地址
 *
 * @param comm [input]A pointer identifying the communication resource target
 * @param remoteRank [input]对端rank_id
 * @param desc [input]单边操作描述信息，包括src, dst, dataType和count
 * @param descNum [input]desc数组的长度，表示内存描述符的个数
 * @param stream [input]执行算子的runtime stream
 * @return HcclResult
 */
extern HcclResult HcclBatchPut(HcclComm comm, u32 remoteRank, HcclOneSideOpDesc* desc, u32 descNum, rtStream_t stream);

/**
 * @brief 发起批量单边get操作，将远端数据读到{comm, remoteRank}指示的远端地址
 *
 * @param comm [input]A pointer identifying the communication resource target
 * @param remoteRank [input]对端rank_id
 * @param desc [input]单边操作描述信息，包括src, dst, dataType和count
 * @param descNum [input]desc数组的长度，表示内存描述符的个数
 * @param stream [input]执行算子的runtime stream
 * @return HcclResult
 */
extern HcclResult HcclBatchGet(HcclComm comm, u32 remoteRank, HcclOneSideOpDesc* desc, u32 descNum, rtStream_t stream);

/**
 * @brief 发起批量单边get操作，将远端数据读到{comm, remoteRank}指示的远端地址
 *
 * @param comm [input]A pointer identifying the communication resource target
 * @param memInfoArray [input]单边操作描述信息，包括memType, addr, size
 * @param commSize [input]通信域的size
 * @param arraySize [input]memInfoArray数组的长度
 * @return HcclResult
 */
extern HcclResult HcclRemapRegistedMemory(HcclComm* comm, CommMem* memInfoArray, u64 commSize, u64 arraySize);

/**
 * @brief 进程粒度注册内存
 *
 * @param mem [input]要注册的内存信息
 * @param memHandle [output]注册成功后返回的内存句柄
 */
extern HcclResult HcclRegisterGlobalMem(const CommMem* mem, void** memHandle);

/**
 * @brief 进程粒度注销内存
 *
 * @param memHandle [input]注册过的内存句柄
 */
extern HcclResult HcclDeregisterGlobalMem(void* memHandle);

/**
 * @brief 将注册过的内存绑定到通信域
 *
 * @param comm [input]通信域句柄
 * @param memHandle [output]要绑定的内存句柄
 */
extern HcclResult HcclCommBindMem(HcclComm comm, void* memHandle);

/**
 * @brief 将绑定过的内存从通信域解绑
 *
 * @param comm [input]通信域句柄
 * @param memHandle [output]要解绑的内存句柄
 */
extern HcclResult HcclCommUnbindMem(HcclComm comm, void* memHandle);

/**
 * @brief 使用固定的连接方式为通信域预先分配需要协商的资源，阻塞接口
 *
 * @param comm [input]通信域句柄
 * @param prepareConfig [input]预分配资源的配置参数
 * @param timeout 连接超时时间，单位秒
 */
extern HcclResult HcclCommPrepare(HcclComm comm, const HcclPrepareConfig* prepareConfig, const int timeout);

#ifdef __cplusplus
}
#endif // __cplusplus
#endif // HCCL_MEM_COMM_H
