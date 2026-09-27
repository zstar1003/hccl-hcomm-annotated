/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CCU_DEVICE_PUB_H
#define CCU_DEVICE_PUB_H

#include <memory>
#include <vector>
#include <cstdint>

#include "ccu_common.h"
#include "ccu_device_res.h"
#include "ccu_res_repo.h"
#include "ccu_drv_handle.h"

#include "hccl_types.h"
#include "hccl_rank_graph.h"

// 支持新老通信域混跑，引入legacy数据结构
#include "unified_platform/pub_inc/ccu/ccu_dev_mgr.h"
#include "dev_type.h"

namespace hcomm {

// CcuAllocResHandleByResDescs 仅以 const CcuResDesc* 形式使用，前向声明即可，
// 避免此处包含 ccu_res_desc.h（其会引入 ccu_dev_mgr_imp.h，造成循环包含）。
class CcuResDesc;

using CcuResHandle = void*;

// 不复用legacy数据结构，对上层支持CommAddr，不使用Hccl::IpAddress
struct CcuChannelPara {
    CommAddr commAddr{};
    uint32_t channelNum{0};
    uint32_t jettyNum{0};
    uint32_t sqSize{0};

    CcuChannelPara() = default;
    CcuChannelPara(const CommAddr& address, const uint32_t channelNum, const uint32_t jettyNum, const uint32_t sqSize)
        : commAddr(address),
          channelNum(channelNum),
          jettyNum(jettyNum),
          sqSize(sqSize)
    {}
};

using CcuJettyType = Hccl::CcuJettyType;
/* 开源自定义算子CCU设备管理实现，当前支持新老通信域混跑，
 * 暂时改用legacy数据结构，避免反向依赖
 * MAKE_ENUM(CcuJettyType, CCUM_CACHED_JETTY, TA_CACHED_JETTY, INVALID_JETTY);
 */

using CcuJettyInfo = Hccl::CcuJettyInfo;
/* 开源自定义算子CCU设备管理实现，当前支持新老通信域混跑，
 * 暂时改用legacy数据结构，避免反向依赖
 * struct CcuJettyInfo {
 *     CcuJettyType jettyType{CcuJettyType::INVALID_JETTY};
 *     uint16_t jettyCtxId{0};
 *     uint16_t taJettyId{0};

 *     uint32_t sqDepth{0};
 *     uint32_t wqeBBStartId{0};

 *     uint64_t sqBufVa{0};
 *     uint32_t sqBufSize{0};
 * };
 */

using CcuChannelInfo = Hccl::CcuChannelInfo;
/* 开源自定义算子CCU设备管理实现，当前支持新老通信域混跑，
 * 暂时改用legacy数据结构，避免反向依赖
 * struct CcuChannelInfo {
 *     uint32_t channelId{0};
 *     uint8_t dieId{0};
 *     std::vector<CcuJettyInfo> jettyInfos;
 * };
 */

/**
 * @brief 启用CCU特性，初始化CCU平台层
 *
 * @param userDevId 用户设备ID
 * @param ccuDrvHandle CCU驱动句柄
 * @return HcclResult 返回HcclResult类型的结果
 * @note 资源不足时返回HCCL_E_UNAVIL，其余非HCCL_SUCCESS结果属于错误
 */
CcuResult CcuInitFeature(const int32_t userDevId, std::shared_ptr<CcuDrvHandle>& ccuDrvHandle);

/**
 * @brief 关闭CCU特性，解初始化CCU平台层
 *
 * @param userDevId 用户设备ID
 * @return HcclResult 返回HcclResult类型的结果
 * @note 资源不足时返回HCCL_E_UNAVIL，其余非HCCL_SUCCESS结果属于错误
 */
CcuResult CcuDeinitFeature(const int32_t userDevId);

/**
 * @brief 获取指定die是否启用
 *
 * @param userDevId 用户设备ID
 * @param dieId ccu channel 所属的 IO Die 编号
 * @param enableFlag 出参，表示该die是否启用
 * @return HcclResult 返回HcclResult类型的结果
 * @note dieId越界时返回HCCL_E_PARA
 */
CcuResult CcuGetDieEnableInfo(int32_t userDevId, uint8_t dieId, bool& enableFlag);

/**
 * @brief 查询指定 ioDie
 * 上各类资源总量，按block分的资源类型（LoopEngine、Ms、Cke）查询的是可以分配的总量（块大小*块总数）
 *
 * @param userDevId 用户设备ID
 * @param dieId ioDie ID
 * @param num 出参，返回该 die 上对应资源类型的总量
 * @return CcuResult
 * @note dieId 越界或 die 未启用时返回错误
 */
CcuResult CcuGetLoopEngineNum(int32_t userDevId, uint8_t dieId, uint32_t& num);
CcuResult CcuGetMsNum(int32_t userDevId, uint8_t dieId, uint32_t& num);
CcuResult CcuGetCkeNum(int32_t userDevId, uint8_t dieId, uint32_t& num);
CcuResult CcuGetXnNum(int32_t userDevId, uint8_t dieId, uint32_t& num);
CcuResult CcuGetGsaNum(int32_t userDevId, uint8_t dieId, uint32_t& num);
CcuResult CcuGetInstructionNum(int32_t userDevId, uint8_t dieId, uint32_t& num);
CcuResult CcuGetMissionNum(int32_t userDevId, uint8_t dieId, uint32_t& num);
CcuResult CcuGetCascCntNum(int32_t userDevId, uint8_t dieId, uint32_t& num);

/**
 * @brief 获取指定device的主板类型
 *
 * @param userDevId 用户设备ID
 * @param hcclMainboardId 出参，返回该device的主板类型
 * @return HcclResult 返回HcclResult类型的结果
 */
HcclResult CcuGetMainboardType(uint32_t userDevId, Hccl::HcclMainboardId& hcclMainboardId);

/**
 * @brief 基于资源描述符数组申请批量资源
 *
 * @param userDevId 用户设备ID
 * @param descs 资源描述符指针数组（每个描述符携带 dieId 及各资源数量）
 * @param descNum 资源描述符数量
 * @param resHandle 返回的CCU批量资源句柄
 * @return HcclResult 返回HcclResult类型的结果
 * @note 资源不足时返回HCCL_E_UNAVIL，其余非HCCL_SUCCESS结果属于错误
 */
CcuResult
CcuAllocResHandleByResDescs(int32_t userDevId, const CcuResDesc* descs[], uint32_t descNum, CcuResHandle& resHandle);

/**
 * @brief 按加速引擎模式申请批量资源
 *
 * @param userDevId 用户设备ID
 * @param ccuEngine CCU通信引擎类型
 * @param resHandle 返回的CCU批量资源句柄
 * @return HcclResult 返回HcclResult类型的结果
 * @note 资源不足时返回HCCL_E_UNAVIL，其余非HCCL_SUCCESS结果属于错误
 */
CcuResult CcuAllocResHandleByInsType(int32_t userDevId, CcuInstanceType ccuInsType, CcuResHandle& resHandle);

/**
 * @brief 根据资源句柄查看对应资源信息
 *
 * @param userDevId 用户设备ID
 * @param resHandle 查询的CCU批量资源句柄
 * @param resRepo 返回的CCU批量资源信息
 * @return HcclResult 返回HcclResult类型的结果
 * @note 资源句柄无法查找到时返回HCCL_E_NOT_FOUND，其余非HCCL_SUCCESS结果属于错误
 */
CcuResult CcuCheckResource(const int32_t userDevId, const CcuResHandle resHandle, CcuResRepository& resRepo);

/**
 * @brief 根据资源句柄释放对应资源信息
 *
 * @param userDevId 用户设备ID
 * @param resHandle 查询的CCU批量资源句柄
 * @note 资源句柄无法查找到时返回HCCL_E_NOT_FOUND，其余非HCCL_SUCCESS结果属于错误
 * @note 返回批量的channel资源总数可能超过申请数量，jettyNum为0时由平台层决定分配数量
 */
HcclResult CcuReleaseResHandle(const int32_t userDevId, const CcuResHandle handle);

/**
 * @brief 申请批量ccu channel资源
 *
 * @param userDevId 用户设备ID
 * @param ccuChannelPara ccu channel 申请参数
 * @param ccuChannelInfos 返回的channel资源信息
 * @return HcclResult 返回HcclResult类型的结果
 * @note 返回批量的channel资源总数可能超过申请数量，jettyNum为0时由平台层决定分配数量
 */
HcclResult CcuAllocChannels(
    const int32_t userDevId, const CcuChannelPara& ccuChannelPara, std::vector<CcuChannelInfo>& ccuChannelInfos);

/**
 * @brief 释放ccu channel资源
 *
 * @param userDevId 用户设备ID
 * @param dieId ccu channel 所属的 IO Die 编号
 * @param ccuChannelId ccu channel 编号
 * @return HcclResult 返回HcclResult类型的结果
 * @note 无
 */
HcclResult CcuReleaseChannel(const int32_t userDevId, const uint8_t dieId, const uint32_t ccuChannelId);

/**
 * @brief 查询CCU设备是否已完成初始化
 *
 * @param userDevId 用户设备ID
 * @return bool true表示已初始化，false表示未初始化或入参非法
 * @note 无
 */
bool CcuIsInited(const int32_t userDevId);

/**
 * @brief 触发CCU Task Kill
 *
 * @param userDevId 用户设备ID
 * @return HcclResult 返回HcclResult类型的结果
 * @note 该接口会处理全部die，未启用die将跳过
 */
HcclResult CcuSetTaskKill(const int32_t userDevId);

/**
 * @brief 配置CCU Task Kill完成状态
 *
 * @param userDevId 用户设备ID
 * @return HcclResult 返回HcclResult类型的结果
 * @note 该接口会处理全部die，未启用die将跳过
 */
HcclResult CcuSetTaskKillDone(const int32_t userDevId);

/**
 * @brief 清空CCU Task Kill状态
 *
 * @param userDevId 用户设备ID
 * @return HcclResult 返回HcclResult类型的结果
 * @note 该接口会处理全部die，未启用die将跳过
 */
HcclResult CcuCleanTaskKillState(const int32_t userDevId);

/**
 * @brief 清理指定ioDie CCU的全部CKE资源，重置为0
 *
 * @param userDevId 用户设备ID
 * @return HcclResult 返回HcclResult类型的结果
 * @note 未启用die无需清理将视为成功
 */
HcclResult CcuCleanDieCkes(const int32_t userDevId, const uint8_t dieId);

/**
 * @brief 申请批量cnt xn资源
 *
 * @param userDevId 用户设备ID
 * @param dieId ioDie ID
 * @param cntXnBlock count xn block结构体
 * @return CcuResult
 * @note 返回申请的count xn block
 */
CcuResult CcuAllocCntXnBlock(const int32_t userDevId, const uint8_t dieId, CntXnBlock& cntXnBlock);

/**
 * @brief 清理指定ioDie CCU的全部CKE资源，重置为0
 *
 * @param userDevId 用户设备ID
 * @param dieId ioDie ID
 * @param cntXnBlock count xn block结构体
 * @return CcuResult
 * @note 根据cntXnBlock释放对应的count xn资源
 */
CcuResult CcuReleaseCntXnBlock(const int32_t userDevId, const uint8_t dieId, const CntXnBlock& cntXnBlock);

/**
 * @brief 查询指定device指定地址的资源token信息
 *
 * @param userDevId 用户设备ID
 * @param srcVa 资源地址
 * @param size 资源size
 * @param tokenId token ID
 * @param tokenValue token value
 * @return CcuResult
 * @note 查询资源token信息，返回token信息
 */
CcuResult
CcuQueryTokenInfo(const int32_t userDevId, uint64_t srcVa, uint64_t size, uint64_t& tokenId, uint64_t& tokenValue);
}; // namespace hcomm
#endif // CCU_DEVICE_PUB_H
