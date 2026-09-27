/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_DEV_MGR_H
#define HCCL_CCU_DEV_MGR_H

#include "ip_address.h"

namespace Hccl {

constexpr uint8_t MAX_CCU_IODIE_NUM = 2; // 设计支持的最大IOdie数量

struct CcuChannelPara {
    IpAddress ipAddr{};
    uint32_t channelNum{0};
    uint32_t jettyNum{0};
    uint32_t sqSize{0};

    CcuChannelPara() = default;
    CcuChannelPara(const IpAddress& ip, const uint32_t channelNum, const uint32_t jettyNum, const uint32_t sqSize)
        : ipAddr(ip),
          channelNum(channelNum),
          jettyNum(jettyNum),
          sqSize(sqSize)
    {}
};

MAKE_ENUM(CcuJettyType, CCUM_CACHED_JETTY, TA_CACHED_JETTY, INVALID_JETTY);

struct CcuJettyInfo {
    CcuJettyType jettyType{CcuJettyType::INVALID_JETTY};
    uint16_t jettyCtxId{0};
    uint16_t taJettyId{0};

    uint32_t sqDepth{0};
    uint32_t wqeBBStartId{0};

    uint64_t sqBufVa{0};
    uint32_t sqBufSize{0};

    std::string ToString() const
    {
        return StringFormat(
            "jettyType=%d, jettyCtxId=%u, taJettyId=%u, sqDepth=%u, "
            "wqeBBStartId=%u, sqBufVa=%llu, sqBufSize=%u.",
            jettyType, jettyCtxId, taJettyId, sqDepth, wqeBBStartId, sqBufVa, sqBufSize);
    }
};

struct CcuChannelInfo {
    uint32_t channelId{0};
    uint8_t dieId{0};
    vector<CcuJettyInfo> jettyInfos;
};

/**
 * @brief 申请批量ccu channel资源
 *
 * @param deviceLogicId device逻辑ID
 * @param ccuChannelPara ccu channel 申请参数
 * @param ccuChannelInfos 返回的channel资源信息
 * @return HcclResult 返回HcclResult类型的结果
 * @note 返回批量的channel资源总数可能超过申请数量，jettyNum为0时由平台层决定分配数量
 */
HcclResult CcuAllocChannels(
    const int32_t deviceLogicId, const CcuChannelPara& ccuChannelPara, std::vector<CcuChannelInfo>& ccuChannelInfos);

/**
 * @brief 释放ccu channel资源
 *
 * @param deviceLogicId device逻辑ID
 * @param dieId ccu channel 所属的 IO Die 编号
 * @param ccuChannelId ccu channel 编号
 * @return HcclResult 返回HcclResult类型的结果
 * @note 无
 */
HcclResult CcuReleaseChannel(const int32_t deviceLogicId, const uint8_t dieId, const uint32_t ccuChannelId);

/**
 * @brief 获取设备Channel资源的规格数量
 *
 * @param deviceLogicId device逻辑ID
 * @param dieId ccu 设备所属 IO Die Id
 * @param channelNum ccu channel 资源的规格数量
 * @return HcclResult 返回HcclResult类型的结果
 * @note 无
 */
HcclResult CcuGetChannelSpecNum(const int32_t deviceLogicId, const uint8_t dieId, uint32_t& channelNum);

/**
 * @brief 查询CCU设备是否已完成初始化
 *
 * @param deviceLogicId device逻辑ID
 * @return bool true表示已初始化，false表示未初始化或入参非法
 * @note 无
 */
bool CcuIsInited(const int32_t deviceLogicId);

/**
 * @brief 触发CCU Task Kill
 *
 * @param deviceLogicId device逻辑ID
 * @return HcclResult 返回HcclResult类型的结果
 * @note 该接口会处理全部die，未启用die将跳过
 */
HcclResult CcuSetTaskKill(const int32_t deviceLogicId);

/**
 * @brief 配置CCU Task Kill完成状态
 *
 * @param deviceLogicId device逻辑ID
 * @return HcclResult 返回HcclResult类型的结果
 * @note 该接口会处理全部die，未启用die将跳过
 */
HcclResult CcuSetTaskKillDone(const int32_t deviceLogicId);

/**
 * @brief 清空CCU Task Kill状态
 *
 * @param deviceLogicId device逻辑ID
 * @return HcclResult 返回HcclResult类型的结果
 * @note 该接口会处理全部die，未启用die将跳过
 */
HcclResult CcuCleanTaskKillState(const int32_t deviceLogicId);

/**
 * @brief 清理指定ioDie CCU的全部CKE资源，重置为0
 *
 * @param deviceLogicId device逻辑ID
 * @return HcclResult 返回HcclResult类型的结果
 * @note 未启用die无需清理将视为成功
 */
HcclResult CcuCleanDieCkes(const int32_t deviceLogicId, const uint8_t dieId);

}; // namespace Hccl
#endif
