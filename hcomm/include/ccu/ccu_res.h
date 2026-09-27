/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CCU_RES_H
#define CCU_RES_H

#include "ccu_res_defs.h"
#include "ccu_types.h"
#include "hcomm_res_defs.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

/**
 * @brief 在指定 CCU 实例上预约一段连续的 Variable(XN) 资源，并在申请期完成地址映射。
 * @param[in] insHandle CCU 实例句柄，资源从该实例的资源池中切分。
 * @param[in] dieId 资源归属的 ioDie ID，取值范围 [0, CCU_MAX_IODIE_NUM)。
 * @param[in] num 预约的连续资源个数，须大于 0。
 * @param[out] varHandle 预约句柄，不可为 nullptr；失败时置 0。
 * @return CcuResult。CCU_E_PARA：dieId 或 num 非法；CCU_E_UNAVAIL：无满足 num 的连续块；
 *         CCU_E_PTR：varHandle 为 nullptr；CCU_E_RUNTIME：地址映射失败。
 * @note 句柄生命周期与 insHandle 绑定，无显式释放接口，由 HcommCcuInsDestroy 统一解映射并还池。
 */
extern CcuResult
HcommCcuVariableAlloc(CcuInsHandle insHandle, uint8_t dieId, uint32_t num, CcuVariableHandle* varHandle);

/**
 * @brief 预约一段连续的 Event(CKE) 资源，参数与生命周期约束同 HcommCcuVariableAlloc。
 * @param[in] insHandle CCU 实例句柄。
 * @param[in] dieId 资源归属的 ioDie ID，取值范围 [0, CCU_MAX_IODIE_NUM)。
 * @param[in] num 预约的连续资源个数，须大于 0。
 * @param[out] eventHandle 预约句柄，不可为 nullptr；失败时置 0。
 * @return CcuResult，错误码语义同 HcommCcuVariableAlloc。
 * @note CKE 池与 kernel 自身的完成事件共用且单 die 容量有限，大 num 可能总量足够却无连续块。
 */
extern CcuResult HcommCcuEventAlloc(CcuInsHandle insHandle, uint8_t dieId, uint32_t num, CcuEventHandle* eventHandle);

/**
 * @brief 查询预约句柄名下第 index 个 Variable(XN) 的进程可访问虚拟地址。
 * @param[in] varHandle HcommCcuVariableAlloc 返回的预约句柄。
 * @param[in] index 预约段内序号，取值范围 [0, num)。
 * @param[out] va 该资源的进程可访问虚拟地址，不可为 nullptr。
 * @return CcuResult。CCU_E_NOT_FOUND：句柄不存在；CCU_E_PARA：句柄类型不符或 index 越界；
 *         CCU_E_PTR：va 为 nullptr。
 * @note 地址已在 Alloc 阶段映射并缓存，此处为纯查表；VA 供 CCU 外模块访问，有效期同句柄。
 */
extern CcuResult HcommCcuVariableGetAddr(CcuVariableHandle varHandle, uint32_t index, uint64_t* va);

/**
 * @brief 查询预约句柄名下第 index 个 Event(CKE) 的进程可访问虚拟地址。
 * @param[in] eventHandle HcommCcuEventAlloc 返回的预约句柄。
 * @param[in] index 预约段内序号，取值范围 [0, num)。
 * @param[out] va 该资源的进程可访问虚拟地址，不可为 nullptr。
 * @return CcuResult，错误码与约束同 HcommCcuVariableGetAddr。
 */
extern CcuResult HcommCcuEventGetAddr(CcuEventHandle eventHandle, uint32_t index, uint64_t* va);

/**
 * @brief 创建 CCU 实例资源描述符
 * @param[in]  dieId    IO Die 编号，取值范围 [0, CCU_MAX_IODIE_NUM)
 * @param[out] resDesc  输出指针，成功时写入 HcommCcuResDescHandle 句柄；调用方负责通过
 *                      HcommCcuInsResDescDestroy 释放
 * @return CCU_SUCCESS  创建成功
 *         CCU_E_PARA   dieId 超出合法范围
 *         CCU_E_PTR    resDesc 为 nullptr
 *         CCU_E_INTERNAL 内部资源描述符创建或登记失败
 */
extern CcuResult HcommCcuInsResDescCreate(uint32_t dieId, HcommCcuResDescHandle* resDesc);

/**
 * @brief 销毁 CCU 实例资源描述符
 * @param[in] resDesc 资源描述符句柄，由 HcommCcuInsResDescCreate 创建
 * @return CCU_SUCCESS  销毁成功
 *         CCU_E_NOT_FOUND resDesc 未注册
 */
extern CcuResult HcommCcuInsResDescDestroy(HcommCcuResDescHandle resDesc);

/**
 * @brief 查询 CCU 资源描述符中已设置的 Die ID
 * @param[in]  resDesc  资源描述符句柄
 * @param[out] dieId    输出指针，接收已设置的 Die ID
 * @return CCU_SUCCESS  查询成功
 *         CCU_E_PTR    dieId 为 nullptr
 *         CCU_E_NOT_FOUND resDesc 未注册
 */
extern CcuResult HcommCcuInsResDescQueryDieId(HcommCcuResDescHandle resDesc, uint32_t* dieId);

/**
 * @brief 按资源类型设置 CCU 资源描述符中的资源数量
 * @param[in,out] resDesc  由 HcommCcuInsResDescCreate 创建的资源描述符句柄，成功时更新指定资源类型的资源数量
 * @param[in] resType  资源类型，取 HcommCcuResType 枚举值
 * @param[in] resNum   期望的资源数量；设置为 0 表示不申请该类型资源
 * @return CCU_SUCCESS  设置成功
 *         CCU_E_PARA   resType 不是合法的 HcommCcuResType 枚举值
 *         CCU_E_NOT_FOUND resDesc 未注册
 */
extern CcuResult HcommCcuInsResDescSetNum(HcommCcuResDescHandle resDesc, HcommCcuResType resType, uint32_t resNum);

/**
 * @brief 按资源类型查询 CCU 资源描述符中已设置的资源数量
 * @param[in]  resDesc  资源描述符句柄
 * @param[in]  resType  资源类型，取 HcommCcuResType 枚举值
 * @param[out] resNum   输出指针，接收已设置的资源数量
 * @return CCU_SUCCESS  查询成功
 *         CCU_E_PARA   resType 超出合法范围
 *         CCU_E_PTR    resNum 为 nullptr
 *         CCU_E_NOT_FOUND resDesc 未注册
 */
extern CcuResult HcommCcuInsResDescQueryNum(HcommCcuResDescHandle resDesc, HcommCcuResType resType, uint32_t* resNum);

/**
 * @brief 基于资源描述符创建 CCU 实例。
 * @param[in] resDescs 资源描述符句柄数组。
 * @param[in] resDescNum 资源描述符数量。
 * @param[out] ccuInsHandle 创建成功后返回的 CCU 实例句柄，不可为 nullptr。
 * @return CCU_SUCCESS  创建成功
 *         CCU_E_PTR    resDescs 或 ccuInsHandle 为 nullptr，或内部资源包分配失败
 *         CCU_E_PARA   resDescNum 超出合法范围，或资源描述符中的 ioDie ID 重复
 *         CCU_E_INTERNAL 内部 CCU 实例创建或登记失败
 */
extern CcuResult
HcommCcuInsCreate(const HcommCcuResDescHandle* resDescs, uint32_t resDescNum, CcuInsHandle* ccuInsHandle);

/**
 * @brief 使用当前 Device 上所有已使能 ioDie 的全部资源创建 CCU 实例。
 * @param[in] dieIds 保留参数，当前版本不读取，调用方应传入 nullptr。
 * @param[in] dieNum 保留参数，当前版本不读取，调用方应传入 0。
 * @param[out] ccuInsHandle 创建成功后返回的 CCU 实例句柄，不可为 nullptr。
 * @return CCU_SUCCESS  创建成功
 *         CCU_E_PTR    ccuInsHandle 为 nullptr，或内部资源包分配失败
 *         CCU_E_INTERNAL 内部 CCU 实例创建或登记失败
 */
extern CcuResult HcommCcuInsCreateDefault(const uint32_t* dieIds, uint32_t dieNum, CcuInsHandle* ccuInsHandle);

/**
 * @brief 销毁 CCU 实例。
 * @param[in] ccuInsHandle CCU 实例句柄。
 * @return CcuResult。
 */
extern CcuResult HcommCcuInsDestroy(CcuInsHandle ccuInsHandle);

/**
 * @brief 查询 CCU 实例在资源描述符所属 ioDie 上占用的资源。
 * @param[in] ccuInsHandle CCU 实例句柄。
 * @param[in,out] resDesc 由 HcommCcuInsResDescCreate 创建的资源描述符句柄；接口读取创建 resDesc 时设置的
 *                        dieId 作为待查询 ioDie，查询成功后写入该 ioDie 上的占用资源数量。
 * @return CcuResult。
 */
extern CcuResult HcommCcuInsQueryResDesc(CcuInsHandle ccuInsHandle, HcommCcuResDescHandle resDesc);

/**
 * @brief 查询指定 Die 上的剩余 CCU 资源（查询最大连续资源）
 * @param[in,out] resDesc 由 HcommCcuInsResDescCreate 创建的资源描述符句柄；接口读取创建 resDesc 时设置的
 *                        dieId 作为待查询 ioDie，查询成功后写入各资源类型的最大连续剩余数量
 * @return CCU_SUCCESS     查询成功
 *         CCU_E_NOT_FOUND resDesc 未注册
 *         CCU_E_PARA      dieId 超出合法范围
 *         CCU_E_UNAVAIL   指定的 Die 未使能
 *         CCU_E_INTERNAL  底层资源查询失败
 */
extern CcuResult HcommCcuQueryRemainResDesc(HcommCcuResDescHandle resDesc);

/**
 * @brief 查询 CCU Kernel 的资源诉求。
 * @param[in] kernelFunc CCU Kernel 函数指针，不可为 nullptr。
 * @param[in] kernelArgs CCU Kernel 参数数组；argNum 为 0 时可为 nullptr。
 * @param[in] argNum CCU Kernel 参数数量。
 * @param[in,out] resDesc 由 HcommCcuInsResDescCreate 创建的资源描述符句柄；接口读取创建 resDesc 时设置的
 *                        dieId 作为资源统计的目标 ioDie，查询成功后写入资源诉求。
 * @return CcuResult。
 */
extern CcuResult HcommCcuKernelQueryResReq(
    const void* kernelFunc, const void** kernelArgs, uint32_t argNum, HcommCcuResDescHandle resDesc);

/**
 * @brief 查询/计算一段本端内存区域的 CCU 访问 token，供 host 端组装 LocalAddr/RemoteAddr 使用。
 * @param[in]  srcVa     内存区域起始虚地址（VA），必须为已注册的合法地址，不可为 0。
 * @param[in]  size      内存区域长度，单位：字节，必须 > 0 且不超过该 VA 对应映射的实际大小。
 * @param[out] tokenInfo 输出指针，指向单个 uint64_t（非数组），成功时写入计算得到的 token 值；不可为 nullptr。
 * @return CcuResult。CCU_SUCCESS 表示成功；CCU_E_PARA 表示 srcVa/size 为 0 或非法；
 *         CCU_E_PTR 表示 tokenInfo 为 nullptr；其余为底层驱动错误码。
 * @note 仅可在 host 端调用，不能在 kernel 函数体内调用。token 属安全信息，调用方不应打印；
 *       其生命周期与对应内存注册绑定。跨 rank 场景下本端 token 需经带外通道交换给对端组装 RemoteAddr。
 */
extern CcuResult HcommCcuGetMemToken(uint64_t srcVa, uint64_t size, uint64_t* tokenInfo);

/**
 * @brief 在 CCU 实例上申请一个级联计数器块（1022 wishCntXn + totalCntXn + expectedCntXn）。
 * @param[in]  ccuInsHandle HcommCcuInsCreate 返回的实例句柄，须为当前 device 上有效实例。
 * @param[in]  dieId     资源归属的 ioDie ID，取值范围 [0, CCU_MAX_IODIE_NUM)。
 * @param[out] handle    输出级联计数器句柄；失败时置 0。
 * @return CcuResult。CCU_SUCCESS 表示成功；CCU_E_PARA 表示 ccuInsHandle/dieId 非法；
 *         CCU_E_PTR 表示 handle 为 nullptr；CCU_E_NOT_FOUND 表示 ccuInsHandle 在当前 device 上不存在（含已被销毁）；
 *         CCU_E_UNAVAIL 表示本 die 无空闲块；CCU_E_INTERNAL 表示底层资源分配失败。
 * @note 句柄生命周期与 ccuInsHandle 绑定，随 HcommCcuInsDestroy 一并释放，无独立释放接口；
 *       仅 host 端调用。每 die 上限 4 块。
 */
extern CcuResult HcommCcuCascCntAlloc(CcuInsHandle ccuInsHandle, uint8_t dieId, HcommCcuCascCntHandle* handle);

/**
 * @brief 查询级联计数器Cascade Counter块对应的 wishCntXn 内存区域，供 host 端组装 LocalAddr/RemoteAddr 使用。
 * @param[in]  handle  HcommCcuCascCntAlloc 返回的级联计数器句柄。
 * @param[out] commMem  输出内存描述符，包含地址与长度；不可为 nullptr。
 * @return CcuResult。CCU_SUCCESS 表示成功；CCU_E_PARA 表示 handle 非法；
 *         CCU_E_PTR 表示 commMem 为 nullptr；CCU_E_INTERNAL 表示底层查询失败。
 * @note 仅可在 host 端调用。返回的内存区域用于 CCU 算子数据面读写。
 */
extern CcuResult HcommCcuCascCntGetMem(HcommCcuCascCntHandle handle, CommMem* commMem);

/**
 * @brief 查询对端 rank 内存区域的 CCU 访问 token，供 host 端组装 RemoteAddr 使用。
 * @param[in]  channelHandle  与对端 rank 建立的通道句柄。
 * @param[in]  srcVa          对端内存区域起始虚地址（VA），不可为 0。
 * @param[out] tokenInfo      输出指针，指向单个 uint64_t（非数组），成功时写入对端 token 值；不可为 nullptr。
 * @return CcuResult。CCU_SUCCESS 表示成功；CCU_E_PARA 表示 channelHandle/srcVa 非法；
 *         CCU_E_PTR 表示 tokenInfo 为 nullptr；CCU_E_INTERNAL 表示底层查询失败。
 * @note 与 HcommCcuGetMemToken 的差异：本接口查询的是对端 rank 的 token，经 channel 带外交换获取；
 *       HcommCcuGetMemToken 查询的是本端 token。token 属安全信息，调用方不应打印。
 */
extern CcuResult HcommCcuGetRmtMemToken(ChannelHandle channelHandle, uint64_t srcVa, uint64_t* tokenInfo);
#ifdef __cplusplus
}
#endif // __cplusplus

#endif // CCU_RES_H
