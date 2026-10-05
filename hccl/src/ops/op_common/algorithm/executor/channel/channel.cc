/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "channel.h"
#include <algorithm>
#include <functional>
#include <vector>
#include <set>
#include <hccl/hccl_types.h>
#include "alg_type.h"
#include "channel_request.h"
#include "topo.h"
#include "topo_host.h"
#include "alg_env_config.h"
#include "comm_engine_utils.h"
#include "op_common.h"
#include "template_utils.h"
#if !defined(HCCL_CANN_COMPAT_850)
#include "ccu_alg_template_base.h"
#endif

constexpr u32 PORT_IDX = 5;
namespace ops_hccl {
HcclResult CalcLevel0ChannelRequest(
    const OpParam& param, const TopoInfo* topoInfo, AlgHierarchyInfo& algHierarchyInfo, const AlgType& algType,
    std::vector<HcclChannelDesc>& channels)
{
    (void)param;
    (void)topoInfo;
    channels.clear();
    SubCommInfo& subCommInfo = algHierarchyInfo.infos[COMM_LEVEL0];
    std::set<u32> connectRanks; // 非通信域rank

    switch (algType.algoLevel0) {
        case AlgTypeLevel0::ALG_LEVEL0_NP_SINGLE_RING:
        case AlgTypeLevel0::ALG_LEVEL0_NP_DOUBLE_RING:
            CHK_RET(CalcRingChannelConnect(
                subCommInfo.localRank, subCommInfo.localRankSize, INVALID_VALUE_RANKID, connectRanks));
            break;
        case AlgTypeLevel0::ALG_LEVEL0_NP_MESH:
        default:
            CHK_RET(CalcMeshChannelConnect(
                subCommInfo.localRank, subCommInfo.localRankSize, INVALID_VALUE_RANKID, connectRanks));
            break;
    }

    CommProtocol protocol = CommProtocol::COMM_PROTOCOL_HCCS;
    for (u32 rank : connectRanks) {
        HcclChannelDesc channelDesc;
        CHK_RET(HcclChannelDescInit(&channelDesc, 1));
        CHK_RET(GetUserRankBySubCommRank(rank, COMM_LEVEL0, algHierarchyInfo, channelDesc.remoteRank));
        channelDesc.channelProtocol = protocol;
        channelDesc.notifyNum = NORMAL_NOTIFY_NUM;
        channels.push_back(channelDesc);
    }
    return HCCL_SUCCESS;
}

HcclResult ProcessMeshInfo(
    const HcclComm comm, const std::vector<std::vector<u32>>& subcommInfo, std::map<u32, u32>& rank2ChannelIdx,
    u32 myRank, std::vector<std::vector<HcclChannelDesc>>& channelsPerDie, u32 enableDieNum, u32 enableDieId,
    std::map<u32, std::vector<HcclChannelDesc>>& rankIdToChannelDesc)
{
#if !defined(AICPU_COMPILE) && (CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0))
    constexpr u32 DIE_NUM_1 = 1;
    constexpr u32 DIE_NUM_2 = 2;
    constexpr u32 DIE_0 = 0;
    constexpr u32 DIE_1 = 1;
    for (u32 rank : subcommInfo[COMM_LEVEL0]) {
        HCCL_INFO("rank = %lld", rank);
        if (rank == myRank) {
            continue;
        }
        if (enableDieNum == DIE_NUM_1) {
            CHK_RET(CcuAlgTemplateBase::SelectChannelToVec(
                comm, myRank, rank, rankIdToChannelDesc, enableDieId, rank2ChannelIdx, channelsPerDie[DIE_0]));
            HCCL_INFO("enableDieNum = %lld", enableDieNum);
        } else if (enableDieNum == DIE_NUM_2) {
            // 加入fromRank 2个die的链路
            CHK_RET(CcuAlgTemplateBase::SelectChannelToVec(
                comm, myRank, rank, rankIdToChannelDesc, DIE_0, rank2ChannelIdx, channelsPerDie[DIE_0]));
            CHK_RET(CcuAlgTemplateBase::SelectChannelToVec(
                comm, myRank, rank, rankIdToChannelDesc, DIE_1, rank2ChannelIdx, channelsPerDie[DIE_1]));
            HCCL_INFO("enableDieNum = %lld", enableDieNum);
        }
    }
    return HcclResult::HCCL_SUCCESS;
#else
    (void)comm;
    (void)subcommInfo;
    (void)rank2ChannelIdx;
    (void)myRank;
    (void)channelsPerDie;
    (void)enableDieNum;
    (void)enableDieId;
    (void)rankIdToChannelDesc;
    return HcclResult::HCCL_E_NOT_SUPPORT;
#endif
}

HcclResult ProcessFlattenLink(
    HcclComm comm, u32 myRank, const std::vector<std::vector<u32>>& subcommInfo, std::vector<HcclChannelDesc>& channels)
{
#if !defined(AICPU_COMPILE) && (CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0))
    std::map<u32, std::vector<HcclChannelDesc>> rankIdToChannelDesc;
    CHK_RET(CcuAlgTemplateBase::RestoreChannelMap(channels, rankIdToChannelDesc));
    uint32_t enableDieNum = 0;
    uint32_t enableDieId = 0;
    CHK_RET(
        CcuAlgTemplateBase::GetDieInfoFromChannelDescs(comm, rankIdToChannelDesc, myRank, enableDieNum, enableDieId));
    if (enableDieNum < 1 || enableDieNum > CCU_DIE_NUM_MAX_2) { // 目前只支持1个或2个die
        HCCL_ERROR("[ProcessFlattenLink] get channelDescs fail");
        return HcclResult::HCCL_E_INTERNAL;
    }
    std::vector<std::vector<HcclChannelDesc>> channelsPerDie;
    channelsPerDie.resize(enableDieNum);
    std::map<u32, u32> rank2ChannelIdx;
    CHK_RET(ProcessMeshInfo(
        comm, subcommInfo, rank2ChannelIdx, myRank, channelsPerDie, enableDieNum, enableDieId, rankIdToChannelDesc));
    if (enableDieNum > 1) { // 通过端口数划分channel，适配跨框die0连die1的场景，避免建链失败
        CHK_RET(CcuAlgTemplateBase::ReverseChannelPerDieIfNeed(
            comm, myRank, channelsPerDie)); // 通过端口数划分channel，适配跨框die0连die1的场景，避免建链失败
    }
    channels = channelsPerDie[0];
    return HcclResult::HCCL_SUCCESS;
#else
    (void)comm;
    (void)myRank;
    (void)subcommInfo;
    (void)channels;
    return HcclResult::HCCL_E_NOT_SUPPORT;
#endif
}

HcclResult CalcLevel1ChannelRequest(
    const OpParam& param, const TopoInfo* topoInfo, AlgHierarchyInfo& algHierarchyInfo, const AlgType& algType,
    std::vector<HcclChannelDesc>& channels)
{
    (void)param;
    channels.clear();
    SubCommInfo& subCommInfo = algHierarchyInfo.infos[COMM_LEVEL1];
    std::set<u32> connectRanks; // 非通信域rank

    switch (algType.algoLevel1) {
        case AlgTypeLevel1::ALG_LEVEL1_NB:
            CHK_RET(CalcNBChannelConnect(
                subCommInfo.localRank, subCommInfo.localRankSize, INVALID_VALUE_RANKID, connectRanks));
            break;
        case AlgTypeLevel1::ALG_LEVEL1_NHR:
            CHK_RET(CalcNHRChannelConnect(
                subCommInfo.localRank, subCommInfo.localRankSize, INVALID_VALUE_RANKID, connectRanks));
            break;
        default:
            CHK_RET(CalcRingChannelConnect(
                subCommInfo.localRank, subCommInfo.localRankSize, INVALID_VALUE_RANKID, connectRanks));
            break;
    }

    // level1走rdma的几种条件：A2单机A+X开启switch；A2多机；A3开启disableHccs；A3跨超卡数不一致
    bool isA2UsedRdma
        = topoInfo->deviceType == HcclDevType::DEV_TYPE_910B
          && (topoInfo->serverNum > 1
              || (topoInfo->serverNum == 1 && topoInfo->isDiffDeviceModule && GetExternalInputIntraRoceSwitch() > 0));
    bool isA3UsedRdma
        = topoInfo->deviceType == HcclDevType::DEV_TYPE_910_93
          && ((topoInfo->superPodNum > 1
               && (topoInfo->multiSuperPodDiffServerNumMode || topoInfo->multiModuleDiffDeviceNumMode))
              || (topoInfo->superPodNum == 1 && topoInfo->serverNum > 1 && GetExternalInputInterHccsDisable()));
    bool isUsedRdma = isA2UsedRdma || isA3UsedRdma;

    CommProtocol protocol = isUsedRdma ? CommProtocol::COMM_PROTOCOL_ROCE : CommProtocol::COMM_PROTOCOL_HCCS;
    for (u32 rank : connectRanks) {
        HcclChannelDesc channelDesc;
        CHK_RET(HcclChannelDescInit(&channelDesc, 1));
        CHK_RET(GetUserRankBySubCommRank(rank, COMM_LEVEL1, algHierarchyInfo, channelDesc.remoteRank));
        channelDesc.channelProtocol = protocol;
        channelDesc.notifyNum = NORMAL_NOTIFY_NUM;
        channels.push_back(channelDesc);
    }
    return HCCL_SUCCESS;
}

HcclResult CalcLevel2ChannelRequest(
    const OpParam& param, const TopoInfo* topoInfo, AlgHierarchyInfo& algHierarchyInfo, const AlgType& algType,
    std::vector<HcclChannelDesc>& channels)
{
    (void)param;
    (void)topoInfo;
    channels.clear();
    SubCommInfo& subCommInfo = algHierarchyInfo.infos[COMM_LEVEL2];
    std::set<u32> connectRanks; // 非通信域rank

    switch (algType.algoLevel2) {
        case AlgTypeLevel2::ALG_LEVEL2_NB:
            CHK_RET(CalcNBChannelConnect(
                subCommInfo.localRank, subCommInfo.localRankSize, INVALID_VALUE_RANKID, connectRanks));
            break;
        case AlgTypeLevel2::ALG_LEVEL2_NHR:
            CHK_RET(CalcNHRChannelConnect(
                subCommInfo.localRank, subCommInfo.localRankSize, INVALID_VALUE_RANKID, connectRanks));
            break;
        default:
            CHK_RET(CalcRingChannelConnect(
                subCommInfo.localRank, subCommInfo.localRankSize, INVALID_VALUE_RANKID, connectRanks));
            break;
    }

    // level2当前一定走rdma
    CommProtocol protocol = CommProtocol::COMM_PROTOCOL_ROCE;

    for (u32 rank : connectRanks) {
        HcclChannelDesc channelDesc;
        CHK_RET(HcclChannelDescInit(&channelDesc, 1));
        CHK_RET(GetUserRankBySubCommRank(rank, COMM_LEVEL2, algHierarchyInfo, channelDesc.remoteRank));
        channelDesc.channelProtocol = protocol;
        channelDesc.notifyNum = NORMAL_NOTIFY_NUM;
        channels.push_back(channelDesc);
    }
    return HCCL_SUCCESS;
}

// [中文导读] [AllReduce逐行 S229] 定义 GetProtocolByEngine 入口：按引擎和 CANN 版本生成协议优先级：AICPU/AICPU_TS 优先 UB_CTP，再 PCIe、UBOE，9.2+ 最后 UB_RTP。
HcclResult GetProtocolByEngine(const OpParam& param, std::vector<CommProtocol>& protocols)
// [中文导读] [AllReduce逐行 S230] 进入 GetProtocolByEngine 的实现作用域；按引擎和 CANN 版本生成协议优先级：AICPU/AICPU_TS 优先 UB_CTP，再 PCIe、UBOE，9.2+ 最后 UB_RTP。
{
    // [中文导读] [AllReduce逐行 S231] 对 protocols 清空 ，准备或更新本阶段列表。
    protocols.clear();
// [中文导读] [AllReduce逐行 S232] 编译期能力条件 CANN_VERSION_NUM >= CANN_VERSION(9, 1, 0) 决定是否生成下方协议或运行时逻辑。
#if CANN_VERSION_NUM >= CANN_VERSION(9, 1, 0)
    // HCCL不再选择UBC_TP（值5）；HCOMM仅为兼容历史调用方保留该协议。
    // [中文导读] [AllReduce逐行 S234] 续接 GetProtocolByEngine 当前语句的具体实参/字段：switch (实际通信引擎) {；由其完整表达式完成参数组装、检查或结果写回。
    switch (param.engine) {
        // [中文导读] [AllReduce逐行 S235] 处理 CommEngine::COMM_ENGINE_AICPU 的专用实现，不同类型或运算在其它 case 分开处理。
        case CommEngine::COMM_ENGINE_AICPU:
        // [中文导读] [AllReduce逐行 S236] 处理 CommEngine::AICPU_TS 引擎 的专用实现，不同类型或运算在其它 case 分开处理。
        case CommEngine::COMM_ENGINE_AICPU_TS:
            // [中文导读] [AllReduce逐行 S237] 对 protocols 追加 CommProtocol::COMM_PROTOCOL_UB_CTP，准备或更新本阶段列表。
            protocols.push_back(CommProtocol::COMM_PROTOCOL_UB_CTP);
            // [中文导读] [AllReduce逐行 S238] 对 protocols 追加 CommProtocol::COMM_PROTOCOL_PCIE，准备或更新本阶段列表。
            protocols.push_back(CommProtocol::COMM_PROTOCOL_PCIE);
            // [中文导读] [AllReduce逐行 S239] 对 protocols 追加 CommProtocol::COMM_PROTOCOL_UBOE，准备或更新本阶段列表。
            protocols.push_back(CommProtocol::COMM_PROTOCOL_UBOE);
// [中文导读] [AllReduce逐行 S240] 编译期能力条件 CANN_VERSION_NUM >= CANN_VERSION(9, 2, 0) 决定是否生成下方协议或运行时逻辑。
#if CANN_VERSION_NUM >= CANN_VERSION(9, 2, 0)
            // [中文导读] [AllReduce逐行 S241] 对 protocols 追加 CommProtocol::COMM_PROTOCOL_UB_RTP，准备或更新本阶段列表。
            protocols.push_back(CommProtocol::COMM_PROTOCOL_UB_RTP);
// [中文导读] [AllReduce逐行 S242] 结束该编译期能力/Host 边界，后续公共返回逻辑在相应构建中保留。
#endif
            // [中文导读] [AllReduce逐行 S243] 退出当前 switch 或内层循环，保留此前选中的结果，不继续后续项。
            break;
        // [中文导读] [AllReduce逐行 S244] 处理 CommEngine::CCU 引擎 的专用实现，不同类型或运算在其它 case 分开处理。
        case CommEngine::COMM_ENGINE_CCU:
            // [中文导读] [AllReduce逐行 S245] 对 protocols 追加 CommProtocol::COMM_PROTOCOL_UB_CTP，准备或更新本阶段列表。
            protocols.push_back(CommProtocol::COMM_PROTOCOL_UB_CTP);
            // [中文导读] [AllReduce逐行 S246] 退出当前 switch 或内层循环，保留此前选中的结果，不继续后续项。
            break;
        // [中文导读] [AllReduce逐行 S247] 处理 CommEngine::AIV 引擎 的专用实现，不同类型或运算在其它 case 分开处理。
        case CommEngine::COMM_ENGINE_AIV:
            // [中文导读] [AllReduce逐行 S248] 对 protocols 追加 CommProtocol::COMM_PROTOCOL_UB_MEM，准备或更新本阶段列表。
            protocols.push_back(CommProtocol::COMM_PROTOCOL_UB_MEM);
            // [中文导读] [AllReduce逐行 S249] 对 protocols 追加 CommProtocol::COMM_PROTOCOL_PCIE，准备或更新本阶段列表。
            protocols.push_back(CommProtocol::COMM_PROTOCOL_PCIE);
            // [中文导读] [AllReduce逐行 S250] 退出当前 switch 或内层循环，保留此前选中的结果，不继续后续项。
            break;
        // [中文导读] [AllReduce逐行 S251] 处理 CommEngine::COMM_ENGINE_CPU 的专用实现，不同类型或运算在其它 case 分开处理。
        case CommEngine::COMM_ENGINE_CPU:
            // level 1到level n-1使用UB协议，server内建联，最外层使用网卡建联
            // [中文导读] [AllReduce逐行 S253] 对 protocols 追加 CommProtocol::COMM_PROTOCOL_UB_CTP，准备或更新本阶段列表。
            protocols.push_back(CommProtocol::COMM_PROTOCOL_UB_CTP);
            // [中文导读] [AllReduce逐行 S254] 对 protocols 追加 CommProtocol::COMM_PROTOCOL_ROCE，准备或更新本阶段列表。
            protocols.push_back(CommProtocol::COMM_PROTOCOL_ROCE);
            // [中文导读] [AllReduce逐行 S255] 退出当前 switch 或内层循环，保留此前选中的结果，不继续后续项。
            break;
        // [中文导读] [AllReduce逐行 S256] 处理 CommEngine::COMM_ENGINE_CPU_TS 的专用实现，不同类型或运算在其它 case 分开处理。
        case CommEngine::COMM_ENGINE_CPU_TS:
            // [中文导读] [AllReduce逐行 S257] 对 protocols 追加 CommProtocol::COMM_PROTOCOL_ROCE，准备或更新本阶段列表。
            protocols.push_back(CommProtocol::COMM_PROTOCOL_ROCE);
            // [中文导读] [AllReduce逐行 S258] 退出当前 switch 或内层循环，保留此前选中的结果，不继续后续项。
            break;
        // [中文导读] [AllReduce逐行 S259] 未列出的类型或状态进入兜底分支；按下面返回码判为不支持或错误。
        default:
            // [中文导读] [AllReduce逐行 S260] 开始 HCCL_WARNING 诊断输出，记录 GetProtocolByEngine 当前阶段的参数、候选或错误；日志本身不决定返回码。
            HCCL_WARNING(
                // [中文导读] [AllReduce逐行 S261] 续接 GetProtocolByEngine 的诊断格式串，展示此行列出的字段标签；它是编译期字符串拼接，不发起额外操作。
                "[GetProtocolByEngine] Unknown engine[%s], set protocol to RESERVED",
                // [中文导读] [AllReduce逐行 S262] 为 GetProtocolByEngine 的诊断/错误宏提供实参：GetEnumToString(GetCommEngineStatusStrMap(), 实际通信引擎).c_str(，与前面的格式占位依次对应。
                GetEnumToString(GetCommEngineStatusStrMap(), param.engine).c_str());
            // [中文导读] [AllReduce逐行 S263] 退出当前 switch 或内层循环，保留此前选中的结果，不继续后续项。
            break;
    // [中文导读] [AllReduce逐行 S264] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
    }
// [中文导读] [AllReduce逐行 S265] 转入该版本或构建条件不成立时的兼容分支。
#else
    // 8.5.0 CANN 无 UB_CTP/UB_MEM 等枚举值；此函数所在的 CalcChannelRequestXxx/CreateChannelRequestByRankId 通路
    // 仅 9.0.0 新路径使用，运行时已由算子入口 GetHcommVersion() < CANN_VERSION(9, 0, 0) 分流到 HcclXxxInner，
    // 8.5.0 下不会真正走到。这里保留空桩让 libhccl.so 外部链接（hccl_test 等）能解析符号。
    // [中文导读] [AllReduce逐行 S269] 显式标记 param 在此兼容/default 分支未使用，避免编译器未使用参数警告。
    (void)param;
// [中文导读] [AllReduce逐行 S270] 结束该编译期能力/Host 边界，后续公共返回逻辑在相应构建中保留。
#endif
    // [中文导读] [AllReduce逐行 S271] 当前路径返回成功；异步原语的成功表示任务/协议提交成功，数据完成依赖相应通知或 Join。
    return HCCL_SUCCESS;
// [中文导读] [AllReduce逐行 S272] 结束 GetProtocolByEngine 实现；其返回状态或已写回字段由调用者接收。
}

// [中文导读] [AllReduce逐行 S274] 定义 CreateChannelFromLink 入口：将 RankGraph 的链路端点、地址、协议和远端 Rank 转为 HcclChannelDesc 请求；尚未申请 ChannelHandle。
HcclResult CreateChannelFromLink(
    // [中文导读] [AllReduce逐行 S275] 续接 CreateChannelFromLink 的入口参数/基类初始化：HcclComm comm, u32 本地用户 Rank, u32 rank, uint32_t 当前网络层编号, u32 idx, const CommLink& link, const std::string& funcName,；引用参数按声明的 const 限制读写。
    HcclComm comm, u32 myRank, u32 rank, uint32_t netLayer, u32 idx, const CommLink& link, const std::string& funcName,
    // [中文导读] [AllReduce逐行 S276] 续接 CreateChannelFromLink 的入口参数/基类初始化：std::vector<HcclChannelDesc>& 通道请求输出列表)；引用参数按声明的 const 限制读写。
    std::vector<HcclChannelDesc>& channels)
// [中文导读] [AllReduce逐行 S277] 进入 CreateChannelFromLink 的实现作用域；将 RankGraph 的链路端点、地址、协议和远端 Rank 转为 HcclChannelDesc 请求；尚未申请 ChannelHandle。
{
    // [中文导读] [AllReduce逐行 S278] 显式标记 comm 在此兼容/default 分支未使用，避免编译器未使用参数警告。
    (void)comm;
    // [中文导读] [AllReduce逐行 S279] 声明本阶段局部变量 HcclChannelDesc 待申请通道描述，实际值由后续查询/计算填写。
    HcclChannelDesc channelDesc;
    // [中文导读] [AllReduce逐行 S280] 清零/初始化描述的默认值；这里没有检查 HcclChannelDescInit 返回码。
    HcclChannelDescInit(&channelDesc, 1);
    // [中文导读] [AllReduce逐行 S281] 设置 目标 Peer 用户 Rank 为 rank；该值供下方当前分支使用。
    channelDesc.remoteRank = rank;
    // [中文导读] [AllReduce逐行 S282] 设置 本地链路端点描述.protocol 为 RankGraph 链路源端点Desc.protocol；该值供下方当前分支使用。
    channelDesc.localEndpoint.protocol = link.srcEndpointDesc.protocol;
    // [中文导读] [AllReduce逐行 S283] 设置 本地链路端点描述.commAddr 为 RankGraph 链路源端点Desc.commAddr；该值供下方当前分支使用。
    channelDesc.localEndpoint.commAddr = link.srcEndpointDesc.commAddr;
    // [中文导读] [AllReduce逐行 S284] 设置 本地链路端点描述.loc 为 RankGraph 链路源端点Desc.loc；该值供下方当前分支使用。
    channelDesc.localEndpoint.loc = link.srcEndpointDesc.loc;
    // [中文导读] [AllReduce逐行 S285] 设置 远端链路端点描述.protocol 为 RankGraph 链路目标端点Desc.protocol；该值供下方当前分支使用。
    channelDesc.remoteEndpoint.protocol = link.dstEndpointDesc.protocol;
    // [中文导读] [AllReduce逐行 S286] 设置 远端链路端点描述.commAddr 为 RankGraph 链路目标端点Desc.commAddr；该值供下方当前分支使用。
    channelDesc.remoteEndpoint.commAddr = link.dstEndpointDesc.commAddr;
    // [中文导读] [AllReduce逐行 S287] 设置 远端链路端点描述.loc 为 RankGraph 链路目标端点Desc.loc；该值供下方当前分支使用。
    channelDesc.remoteEndpoint.loc = link.dstEndpointDesc.loc;
    // [中文导读] [AllReduce逐行 S288] 开始 HCCL_DEBUG 诊断输出，记录 CreateChannelFromLink 当前阶段的参数、候选或错误；日志本身不决定返回码。
    HCCL_DEBUG(
        // [中文导读] [AllReduce逐行 S289] 续接 CreateChannelFromLink 的诊断格式串，展示此行列出的字段标签；它是编译期字符串拼接，不发起额外操作。
        "[CreateChannelFromLink]%s local device phyId: %u, remote device phyId: %u.", funcName.c_str(),
        // [中文导读] [AllReduce逐行 S290] 为 CreateChannelFromLink 的诊断/错误宏提供实参：本地链路端点描述.loc.device.devPhyId, 远端链路端点描述.loc.device.devPhyId，与前面的格式占位依次对应。
        channelDesc.localEndpoint.loc.device.devPhyId, channelDesc.remoteEndpoint.loc.device.devPhyId);
    // [中文导读] [AllReduce逐行 S291] 开始 HCCL_INFO 诊断输出，记录 CreateChannelFromLink 当前阶段的参数、候选或错误；日志本身不决定返回码。
    HCCL_INFO(
        // [中文导读] [AllReduce逐行 S292] 续接 CreateChannelFromLink 的诊断格式串，展示此行列出的字段标签；它是编译期字符串拼接，不发起额外操作。
        "[CreateChannelFromLink]%s Add channel request between %zu and %zu, netLayerIdx %u, "
        // [中文导读] [AllReduce逐行 S293] 续接 CreateChannelFromLink 的诊断格式串，展示此行列出的字段标签；它是编译期字符串拼接，不发起额外操作。
        "linkListIdx %u, protocol %zu",
        // [中文导读] [AllReduce逐行 S294] 为 CreateChannelFromLink 的诊断/错误宏提供实参：funcName.c_str(), 本地用户 Rank, 目标 Peer 用户 Rank, 当前网络层编号, idx, 远端链路端点描述.protocol，与前面的格式占位依次对应。
        funcName.c_str(), myRank, channelDesc.remoteRank, netLayer, idx, channelDesc.remoteEndpoint.protocol);
    // [中文导读] [AllReduce逐行 S295] 设置 通道协议 为 链路属性.linkProtocol；该值供下方当前分支使用。
    channelDesc.channelProtocol = link.linkAttr.linkProtocol;
    // [中文导读] [AllReduce逐行 S296] 设置 通道通知槽请求数量 为 NORMAL_NOTIFY_NUM；该值供下方当前分支使用。
    channelDesc.notifyNum = NORMAL_NOTIFY_NUM;
    // [中文导读] [AllReduce逐行 S297] 对 通道请求输出列表 追加 待申请通道描述，准备或更新本阶段列表。
    channels.push_back(channelDesc);
    // [中文导读] [AllReduce逐行 S298] 当前路径返回成功；异步原语的成功表示任务/协议提交成功，数据完成依赖相应通知或 Join。
    return HCCL_SUCCESS;
// [中文导读] [AllReduce逐行 S299] 结束 CreateChannelFromLink 实现；其返回状态或已写回字段由调用者接收。
}

// [中文导读] [AllReduce逐行 S301] 定义 ProcessLinkForProtocol 入口：依协议优先级筛选链路，同一协议去重本地源端点 Die；找到任一可用协议后不再尝试低优先级协议。
HcclResult ProcessLinkForProtocol(
    // [中文导读] [AllReduce逐行 S302] 续接 ProcessLinkForProtocol 的入口参数/基类初始化：HcclComm comm, const std::vector<CommProtocol>& 按引擎排序的候选协议, const std::vector<CommLink>& RankGraph 返回的链路数组,；引用参数按声明的 const 限制读写。
    HcclComm comm, const std::vector<CommProtocol>& expectedProtocols, const std::vector<CommLink>& linkList,
    // [中文导读] [AllReduce逐行 S303] 续接 ProcessLinkForProtocol 的入口参数/基类初始化：u32 本地用户 Rank, u32 remoteRank, uint32_t 当前网络层编号, std::vector<HcclChannelDesc>& 通道请求输出列表, bool& 当前协议找到链路的标志,；引用参数按声明的 const 限制读写。
    u32 myRank, u32 remoteRank, uint32_t netLayer, std::vector<HcclChannelDesc>& channels, bool& protocolFound,
    // [中文导读] [AllReduce逐行 S304] 续接 ProcessLinkForProtocol 的入口参数/基类初始化：const std::string& funcName)；引用参数按声明的 const 限制读写。
    const std::string& funcName)
// [中文导读] [AllReduce逐行 S305] 进入 ProcessLinkForProtocol 的实现作用域；依协议优先级筛选链路，同一协议去重本地源端点 Die；找到任一可用协议后不再尝试低优先级协议。
{
    // [中文导读] [AllReduce逐行 S306] 设置 当前协议找到链路的标志 为 false；该值供下方当前分支使用。
    protocolFound = false;
    // [中文导读] [AllReduce逐行 S307] 建立本阶段局部对象 std::set<uint32_t> 已选本地源端点 Die 集合，供 ProcessLinkForProtocol 下方参数组装和子调用使用。
    std::set<uint32_t> seenDie;
    // [中文导读] [AllReduce逐行 S308] 按执行引擎协议优先级枚举候选协议；边界/迭代规则为 (auto expectedProtocol : 按引擎排序的候选协议。
    for (auto expectedProtocol : expectedProtocols) {
        // [中文导读] [AllReduce逐行 S309] 遍历当前网络层查询出的链路；边界/迭代规则为 (u32 idx = 0; idx 小于 RankGraph 返回的链路数组.size(); idx++。
        for (u32 idx = 0; idx < linkList.size(); idx++) {
            // [中文导读] [AllReduce逐行 S310] 分支条件为 RankGraph 返回的链路数组[idx].linkAttr.linkProtocol 不等于 expectedProtocol；成立进入本块，未成立继续后续分支。
            if (linkList[idx].linkAttr.linkProtocol != expectedProtocol) {
                // [中文导读] [AllReduce逐行 S311] 跳过当前遍历项的剩余步骤，直接处理下一项。
                continue;
            // [中文导读] [AllReduce逐行 S312] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
            }
            // [中文导读] [AllReduce逐行 S313] 设置 EndpointAttrDieId dieId 为 0；该值供下方当前分支使用。
            EndpointAttrDieId dieId = 0;
            // [中文导读] [AllReduce逐行 S314] 查询链路对应的本地源端点 Die 身份用于同 Die 去重；失败不直接返回错误，仍可添加该匹配链路。
            HcclResult dieRet = HcclRankGraphGetEndpointInfo(
                // [中文导读] [AllReduce逐行 S315] 续接 ProcessLinkForProtocol 当前语句的具体实参/字段：comm, 本地用户 Rank, &RankGraph 返回的链路数组[idx].srcEndpointDesc, ENDPOINT_ATTR_DIE_ID, sizeof(dieId), &dieId)；由其完整表达式完成参数组装、检查或结果写回。
                comm, myRank, &linkList[idx].srcEndpointDesc, ENDPOINT_ATTR_DIE_ID, sizeof(dieId), &dieId);
            // [中文导读] [AllReduce逐行 S316] 设置 是否为该链路新增描述的标志 为 true；该值供下方当前分支使用。
            bool shouldAdd = true;
            // [中文导读] [AllReduce逐行 S317] 分支条件为 Die 信息查询返回码 等于 成功状态；成立进入本块，未成立继续后续分支。
            if (dieRet == HCCL_SUCCESS) {
                // [中文导读] [AllReduce逐行 S318] 设置 是否为该链路新增描述的标志 为 已选本地源端点 Die 集合.insert(dieId).second；该值供下方当前分支使用。
                shouldAdd = seenDie.insert(dieId).second;
            // [中文导读] [AllReduce逐行 S319] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
            }
            // [中文导读] [AllReduce逐行 S320] 分支条件为 是否为该链路新增描述的标志；成立进入本块，未成立继续后续分支。
            if (shouldAdd) {
                // [中文导读] [AllReduce逐行 S321] 执行下方完整子调用；返回值非成功时立即从当前函数返回该错误。
                CHK_RET(
                    // [中文导读] [AllReduce逐行 S322] 把链路端点转换成待申请通道描述；本行实参为 CreateChannelFromLink(comm, 本地用户 Rank, remoteRank, 当前网络层编号, idx, RankGraph 返回的链路数组[idx], funcName, 通道请求输出列表))。
                    CreateChannelFromLink(comm, myRank, remoteRank, netLayer, idx, linkList[idx], funcName, channels));
                // [中文导读] [AllReduce逐行 S323] 设置 当前协议找到链路的标志 为 true；该值供下方当前分支使用。
                protocolFound = true;
            // [中文导读] [AllReduce逐行 S324] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
            }
        // [中文导读] [AllReduce逐行 S325] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
        }
        // [中文导读] [AllReduce逐行 S326] 当前协议只要找到任意匹配链路就停止协议优先级循环，不再尝试后续低优先级协议。
        if (protocolFound) {
            // [中文导读] [AllReduce逐行 S327] 退出当前 switch 或内层循环，保留此前选中的结果，不继续后续项。
            break;
        // [中文导读] [AllReduce逐行 S328] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
        }
    // [中文导读] [AllReduce逐行 S329] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
    }
    // [中文导读] [AllReduce逐行 S330] 当前路径返回成功；异步原语的成功表示任务/协议提交成功，数据完成依赖相应通知或 Join。
    return HCCL_SUCCESS;
// [中文导读] [AllReduce逐行 S331] 结束 ProcessLinkForProtocol 实现；其返回状态或已写回字段由调用者接收。
}

HcclResult GetRankFullMeshLayers(
    HcclComm comm, const std::vector<std::vector<u32>>& subcommInfo, std::vector<uint32_t> netLayersVector, u32 myRank,
    u32& curNetLayer)
{
#ifndef AICPU_COMPILE
    for (auto netLayer : netLayersVector) {
        bool isStainPath = true;
        HCCL_INFO("netlayer=%d", netLayer);
        CommLink* linkList = nullptr;
        u32 listSize = 0;
        for (u32 rank : subcommInfo[COMM_LEVEL0]) {
            if (rank == myRank) {
                continue;
            }
            CHK_RET(HcclRankGraphGetLinks(comm, netLayer, myRank, rank, &linkList, &listSize));
            HCCL_INFO("dstrank = %d,listsize = %d", rank, listSize);
            if (listSize == 0) {
                isStainPath = false;
                break;
            }
        }
        if (isStainPath) {
            curNetLayer = netLayer;
            HCCL_INFO("curNetLayer=%d", curNetLayer);
            break;
        }
        CHK_PRT_RET(
            (curNetLayer == 0) && (netLayer != 0),
            HCCL_ERROR("[GetRankFullMeshLayers] Failed to get cur netlayer myRank=%u .", myRank),
            HcclResult::HCCL_E_INTERNAL);
    }
    return HCCL_SUCCESS;
#else
    (void)comm;
    (void)subcommInfo;
    (void)netLayersVector;
    (void)curNetLayer;

    HCCL_ERROR(
        "[GetRankFullMeshLayers] This function is not supported in AICPU mode, "
        "myRank[%u].",
        myRank);
    return HcclResult::HCCL_E_NOT_SUPPORT;
#endif
}

// [中文导读] [AllReduce逐行 S379] 定义 CalcChannelRequestMesh1D 入口：遍历一层 Mesh 的其它 Rank，逐网络层查询链路，选择首个有可用协议的层并产生通道描述请求。
HcclResult CalcChannelRequestMesh1D(
    // [中文导读] [AllReduce逐行 S380] 续接 CalcChannelRequestMesh1D 的入口参数/基类初始化：HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,；引用参数按声明的 const 限制读写。
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    // [中文导读] [AllReduce逐行 S381] 续接 CalcChannelRequestMesh1D 的入口参数/基类初始化：const std::vector<std::vector<u32>>& subcommInfo, std::vector<HcclChannelDesc>& 通道请求输出列表)；引用参数按声明的 const 限制读写。
    const std::vector<std::vector<u32>>& subcommInfo, std::vector<HcclChannelDesc>& channels)
// [中文导读] [AllReduce逐行 S382] 进入 CalcChannelRequestMesh1D 的实现作用域；遍历一层 Mesh 的其它 Rank，逐网络层查询链路，选择首个有可用协议的层并产生通道描述请求。
{
// [中文导读] [AllReduce逐行 S383] 以下资源/成本逻辑仅在 Host 编译版本执行；AICPU_COMPILE 构建跳过该段。
#ifndef AICPU_COMPILE
    // [中文导读] [AllReduce逐行 S384] 显式标记 param 在此兼容/default 分支未使用，避免编译器未使用参数警告。
    (void)param;
    // [中文导读] [AllReduce逐行 S385] 清空输出请求，避免重用上一次资源计算的 Peer 通道描述。
    channels.clear();
    // [中文导读] [AllReduce逐行 S386] 在第零层算法 Rank 列表中定位自身，确保后续 Peer 枚举基于包含本地 Rank 的通信域。
    auto it = std::find(subcommInfo[COMM_LEVEL0].begin(), subcommInfo[COMM_LEVEL0].end(), topoInfo->userRank);
    // [中文导读] [AllReduce逐行 S387] 开始条件错误返回宏：条件成立时打印下方诊断并返回所列错误码；判断对象由下一行展开。
    CHK_PRT_RET(
        // [中文导读] [AllReduce逐行 S388] 调用 end 完成当前参数所指的子步骤；本行实参为 (it 等于 subcommInfo[COMM_LEVEL0].end()),。
        (it == subcommInfo[COMM_LEVEL0].end()),
        // [中文导读] [AllReduce逐行 S389] 开始 HCCL_ERROR 诊断输出，记录 CalcChannelRequestMesh1D 当前阶段的参数、候选或错误；日志本身不决定返回码。
        HCCL_ERROR("[CollAlgFactory] [channel] Rank [%d] is not in commInfo.", topoInfo->userRank),
        // [中文导读] [AllReduce逐行 S390] 向条件返回宏提供 HcclResult::参数错误；上方检查成立才退出当前函数。
        HcclResult::HCCL_E_PARA);
    // [中文导读] [AllReduce逐行 S391] 设置 本地用户 Rank 为 本地用户 Rank；该值供下方当前分支使用。
    u32 myRank = topoInfo->userRank;
    // [中文导读] [AllReduce逐行 S392] 建立本阶段局部对象 std::vector<CommProtocol> 按引擎排序的候选协议，供 CalcChannelRequestMesh1D 下方参数组装和子调用使用。
    std::vector<CommProtocol> expectedProtocols;
    // [中文导读] [AllReduce逐行 S393] AICPU_TS 的 UB 路径按引擎取得 UB_CTP、PCIe、UBOE 等候选协议顺序。
    CHK_RET(GetProtocolByEngine(param, expectedProtocols));
    // [中文导读] [AllReduce逐行 S394] 为第零层每个用户 Rank 枚举通道请求；下一分支跳过自身。
    for (u32 rank : subcommInfo[COMM_LEVEL0]) {
        // [中文导读] [AllReduce逐行 S395] 分支条件为 rank 等于 本地用户 Rank；成立进入本块，未成立继续后续分支。
        if (rank == topoInfo->userRank) {
            // [中文导读] [AllReduce逐行 S396] 跳过当前遍历项的剩余步骤，直接处理下一项。
            continue;
        // [中文导读] [AllReduce逐行 S397] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
        }
        // [中文导读] [AllReduce逐行 S398] 设置 当前 Peer 处理前的请求数 为 通道请求输出列表.size()；该值供下方当前分支使用。
        size_t channelCountBefore = channels.size();
        // [中文导读] [AllReduce逐行 S399] 声明本阶段局部变量 uint32_t* 当前网络层编号s，实际值由后续查询/计算填写。
        uint32_t* netLayers;
        // [中文导读] [AllReduce逐行 S400] 声明本阶段局部变量 uint32_t 网络层数量，实际值由后续查询/计算填写。
        uint32_t netLayerNum;
        // [中文导读] [AllReduce逐行 S401] 查询 RankGraph 的网络层列表；失败立即退出资源计算。
        CHK_RET(HcclRankGraphGetLayers(comm, &netLayers, &netLayerNum));
        // [中文导读] [AllReduce逐行 S402] 调用 netLayersVector 完成当前参数所指的子步骤；本行实参为 std::vector<uint32_t> 网络层编号列表(当前网络层编号s, 当前网络层编号s + 网络层数量)。
        std::vector<uint32_t> netLayersVector(netLayers, netLayers + netLayerNum);
        // [中文导读] [AllReduce逐行 S403] 按 RankGraph 顺序枚举网络层；边界/迭代规则为 (auto 当前网络层编号 : 网络层编号列表。
        for (auto netLayer : netLayersVector) {
            // [中文导读] [AllReduce逐行 S404] 设置 CommLink* RankGraph 返回的链路数组 为 nullptr；该值供下方当前分支使用。
            CommLink* linkList = nullptr;
            // [中文导读] [AllReduce逐行 S405] 声明本阶段局部变量 u32 链路数组长度，实际值由后续查询/计算填写。
            u32 listSize;
            // [中文导读] [AllReduce逐行 S406] 查询当前网络层本地 Rank 到当前 Peer 的所有 CommLink。
            CHK_RET(HcclRankGraphGetLinks(comm, netLayer, myRank, rank, &linkList, &listSize));
            // [中文导读] [AllReduce逐行 S407] 分支条件为 链路数组长度 等于 0；成立进入本块，未成立继续后续分支。
            if (listSize == 0) {
                // [中文导读] [AllReduce逐行 S408] 跳过当前遍历项的剩余步骤，直接处理下一项。
                continue;
            // [中文导读] [AllReduce逐行 S409] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
            }
            // [中文导读] [AllReduce逐行 S410] 调用 links 完成当前参数所指的子步骤；本行实参为 std::vector<CommLink> links(RankGraph 返回的链路数组, RankGraph 返回的链路数组 + 链路数组长度)。
            std::vector<CommLink> links(linkList, linkList + listSize);
            // [中文导读] [AllReduce逐行 S411] 设置 当前协议找到链路的标志 为 false；该值供下方当前分支使用。
            bool protocolFound = false;
            // [中文导读] [AllReduce逐行 S412] 按协议顺序和 Die 去重筛选当前层链路；返回值非成功时立即从当前函数返回该错误。
            CHK_RET(ProcessLinkForProtocol(
                // [中文导读] [AllReduce逐行 S413] 续接本次错误检查/子调用实参：comm, 按引擎排序的候选协议, links, 本地用户 Rank, rank, 当前网络层编号, 通道请求输出列表, 当前协议找到链路的标志；返回行为由所在完整宏决定。
                comm, expectedProtocols, links, myRank, rank, netLayer, channels, protocolFound,
                // [中文导读] [AllReduce逐行 S414] 续接本次错误检查/子调用实参：std::string("[CalcChannelRequestMesh1D]"；返回行为由所在完整宏决定。
                std::string("[CalcChannelRequestMesh1D]")));
            // [中文导读] [AllReduce逐行 S415] 当前层一旦增加了通道请求，就不再继续查找此 Peer 的后续网络层。
            if (channels.size() > channelCountBefore) {
                // [中文导读] [AllReduce逐行 S416] 退出当前 switch 或内层循环，保留此前选中的结果，不继续后续项。
                break;
            // [中文导读] [AllReduce逐行 S417] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
            }
        // [中文导读] [AllReduce逐行 S418] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
        }
        // [中文导读] [AllReduce逐行 S419] 若所有网络层均未给此 Peer 增加通道描述，报内部错误；保证每个其它 Rank 至少有一个请求。
        CHK_PRT_RET(
            // [中文导读] [AllReduce逐行 S420] 调用 size 完成当前参数所指的子步骤；本行实参为 通道请求输出列表.size() 等于 当前 Peer 处理前的请求数,。
            channels.size() == channelCountBefore,
            // [中文导读] [AllReduce逐行 S421] 开始 HCCL_ERROR 诊断输出，记录 CalcChannelRequestMesh1D 当前阶段的参数、候选或错误；日志本身不决定返回码。
            HCCL_ERROR(
                // [中文导读] [AllReduce逐行 S422] 续接 CalcChannelRequestMesh1D 的诊断格式串，展示此行列出的字段标签；它是编译期字符串拼接，不发起额外操作。
                "[CalcChannelRequestMesh1D] Failed to create channel between myRank=%u and rank=%u, there is no link.",
                // [中文导读] [AllReduce逐行 S423] 为 CalcChannelRequestMesh1D 的诊断/错误宏提供实参：本地用户 Rank, rank，与前面的格式占位依次对应。
                myRank, rank),
            // [中文导读] [AllReduce逐行 S424] 向条件返回宏提供 HcclResult::内部错误；上方检查成立才退出当前函数。
            HcclResult::HCCL_E_INTERNAL);
    // [中文导读] [AllReduce逐行 S425] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
    }
// [中文导读] [AllReduce逐行 S426] 结束该编译期能力/Host 边界，后续公共返回逻辑在相应构建中保留。
#endif
    // [中文导读] [AllReduce逐行 S427] 当前路径返回成功；异步原语的成功表示任务/协议提交成功，数据完成依赖相应通知或 Join。
    return HCCL_SUCCESS;
// [中文导读] [AllReduce逐行 S428] 结束 CalcChannelRequestMesh1D 实现；其返回状态或已写回字段由调用者接收。
}

#ifndef AICPU_COMPILE
static HcclResult CalcHighestHostRoceChannels(
    HcclComm comm, u32 myRank, const std::vector<u32>& remoteRanks, const std::string& funcName,
    std::vector<HcclChannelDesc>& channels)
{
    uint32_t* netLayers = nullptr;
    uint32_t netLayerNum = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &netLayers, &netLayerNum));
    CHK_PRT_RET(
        netLayers == nullptr || netLayerNum == 0, HCCL_ERROR("%s net layer list is empty.", funcName.c_str()),
        HCCL_E_INTERNAL);
    std::vector<uint32_t> netLayersVector(netLayers, netLayers + netLayerNum);
    std::sort(netLayersVector.begin(), netLayersVector.end(), std::greater<uint32_t>());

    for (u32 remoteRank : remoteRanks) {
        if (remoteRank == myRank) {
            continue;
        }

        bool channelFound = false;
        for (u32 netLayer : netLayersVector) {
            CommLink* linkList = nullptr;
            u32 listSize = 0;
            CHK_RET(HcclRankGraphGetLinks(comm, netLayer, myRank, remoteRank, &linkList, &listSize));
            CHK_PRT_RET(
                listSize > 0 && linkList == nullptr,
                HCCL_ERROR(
                    "%s link list is null between myRank=%u and rank=%u on netLayer=%u.", funcName.c_str(), myRank,
                    remoteRank, netLayer),
                HCCL_E_INTERNAL);
            for (u32 idx = 0; idx < listSize; ++idx) {
                const CommLink& link = linkList[idx];
                if (link.linkAttr.linkProtocol != CommProtocol::COMM_PROTOCOL_ROCE
                    || link.srcEndpointDesc.loc.locType != ENDPOINT_LOC_TYPE_HOST
                    || link.dstEndpointDesc.loc.locType != ENDPOINT_LOC_TYPE_HOST) {
                    continue;
                }
                CHK_RET(CreateChannelFromLink(comm, myRank, remoteRank, netLayer, idx, link, funcName, channels));
                channelFound = true;
            }
            if (channelFound) {
                break;
            }
        }
        CHK_PRT_RET(
            !channelFound,
            HCCL_ERROR(
                "%s Failed to create HOST/ROCE channel between myRank=%u and rank=%u.", funcName.c_str(), myRank,
                remoteRank),
            HCCL_E_INTERNAL);
    }
    return HCCL_SUCCESS;
}
#endif

HcclResult CalcChannelRequestMesh1DHighestHostRoce(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    const std::vector<std::vector<u32>>& subcommInfo, std::vector<HcclChannelDesc>& channels)
{
#ifndef AICPU_COMPILE
    (void)param;
    channels.clear();
    CHK_PTR_NULL(topoInfo);
    CHK_PRT_RET(
        subcommInfo.empty() || subcommInfo[COMM_LEVEL0].empty(),
        HCCL_ERROR("[CalcChannelRequestMesh1DHighestHostRoce] subcommInfo is empty."), HCCL_E_PARA);

    const u32 myRank = topoInfo->userRank;
    auto rankIt = std::find(subcommInfo[COMM_LEVEL0].begin(), subcommInfo[COMM_LEVEL0].end(), myRank);
    CHK_PRT_RET(
        rankIt == subcommInfo[COMM_LEVEL0].end(),
        HCCL_ERROR("[CalcChannelRequestMesh1DHighestHostRoce] Rank [%u] is not in commInfo.", myRank), HCCL_E_PARA);
    CHK_RET(CalcHighestHostRoceChannels(
        comm, myRank, subcommInfo[COMM_LEVEL0], "[CalcChannelRequestMesh1DHighestHostRoce]", channels));
#endif
    return HCCL_SUCCESS;
}

HcclResult CalcChannelRequestMesh1DFullMesh(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    const std::vector<std::vector<u32>>& subcommInfo, std::vector<HcclChannelDesc>& channels)
{
#if !defined(HCCL_CANN_COMPAT_850) && !defined(AICPU_COMPILE)
    channels.clear();
    (void)param;
    auto it = std::find(subcommInfo[COMM_LEVEL0].begin(), subcommInfo[COMM_LEVEL0].end(), topoInfo->userRank);
    CHK_PRT_RET(
        (it == subcommInfo[COMM_LEVEL0].end()),
        HCCL_ERROR("[CollAlgFactory] [channel] Rank [%d] is not in commInfo.", topoInfo->userRank),
        HcclResult::HCCL_E_PARA);
    std::vector<CommProtocol> expectedProtocols;
    u32 myRank = topoInfo->userRank;
    CHK_RET(GetProtocolByEngine(param, expectedProtocols));

    uint32_t *netLayers, netLayerNum;
    uint32_t curNetLayer = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &netLayers, &netLayerNum));
    std::vector<uint32_t> netLayersVector(netLayers, netLayers + netLayerNum);
    CHK_RET(GetRankFullMeshLayers(comm, subcommInfo, netLayersVector, myRank, curNetLayer));

    for (u32 rank : subcommInfo[COMM_LEVEL0]) {
        if (rank == topoInfo->userRank) {
            continue;
        }
        CommLink* linkList = nullptr;
        u32 listSize;
        CHK_RET(HcclRankGraphGetLinks(comm, curNetLayer, myRank, rank, &linkList, &listSize));
        CHK_PRT_RET(
            (listSize == 0),
            HCCL_ERROR("[CalcChannelRequestMesh1D] These is no link between myRank=%u, dstRank=%u.", myRank, rank),
            HcclResult::HCCL_E_INTERNAL);
        std::vector<CommLink> links(linkList, linkList + listSize);
        bool protocolFound = false;
        CHK_RET(ProcessLinkForProtocol(
            comm, expectedProtocols, links, myRank, rank, curNetLayer, channels, protocolFound,
            std::string("[CalcChannelRequestMesh1D]")));
    }
    if (curNetLayer != 0) { // 通过端口数划分channel，适配跨框die0连die1的场景，避免建链失败
        CHK_RET(ProcessFlattenLink(comm, myRank, subcommInfo, channels));
    }
    return HCCL_SUCCESS;
#else
    return HCCL_E_INTERNAL;
#endif
}

#ifndef AICPU_COMPILE
static HcclResult
CheckNetLayerExists(HcclComm comm, u32 netLayer, const std::string& tag, bool linkRequired, bool& netLayerExists)
{
    netLayerExists = false;
    uint32_t* netLayers = nullptr;
    uint32_t netLayerNum = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &netLayers, &netLayerNum));
    std::vector<uint32_t> netLayersVector(netLayers, netLayers + netLayerNum);
    for (auto layer : netLayersVector) {
        if (layer == netLayer) {
            netLayerExists = true;
            break;
        }
    }

    if (!netLayerExists) {
        CHK_PRT_RET(
            linkRequired, HCCL_ERROR("[%s] netLayer[%u] does not exist in rankGraph.", tag.c_str(), netLayer),
            HcclResult::HCCL_E_INTERNAL);
        HCCL_WARNING("[%s] netLayer[%u] does not exist in rankGraph, skip.", tag.c_str(), netLayer);
    }
    return HCCL_SUCCESS;
}
#endif

static HcclResult CalcChannelRequestMesh1DByLevel(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    const std::vector<std::vector<u32>>& subcommInfo, std::vector<HcclChannelDesc>& channels, u32 netLayer,
    const std::string& tag, bool linkRequired)
{
#ifndef AICPU_COMPILE
    channels.clear();
    bool netLayerExists = false;
    CHK_RET(CheckNetLayerExists(comm, netLayer, tag, linkRequired, netLayerExists));
    if (!netLayerExists) {
        return HCCL_SUCCESS;
    }

    auto it = std::find(subcommInfo[COMM_LEVEL0].begin(), subcommInfo[COMM_LEVEL0].end(), topoInfo->userRank);
    CHK_PRT_RET(
        (it == subcommInfo[COMM_LEVEL0].end()),
        HCCL_ERROR("[%s] Rank [%u] is not in commInfo.", tag.c_str(), topoInfo->userRank), HcclResult::HCCL_E_PARA);

    u32 myRank = topoInfo->userRank;
    std::vector<CommProtocol> expectedProtocols;
    CHK_RET(GetProtocolByEngine(param, expectedProtocols));

    for (u32 rank : subcommInfo[COMM_LEVEL0]) {
        if (rank == topoInfo->userRank) {
            continue;
        }
        size_t channelCountBefore = channels.size();

        CommLink* linkList = nullptr;
        u32 listSize;
        CHK_RET(HcclRankGraphGetLinks(comm, netLayer, myRank, rank, &linkList, &listSize));

        if (listSize == 0) {
            if (linkRequired) {
                HCCL_ERROR("[%s] No intra-frame link between myRank=%u and rank=%u.", tag.c_str(), myRank, rank);
                return HcclResult::HCCL_E_INTERNAL;
            }
            HCCL_WARNING("[%s] No inter-frame link between myRank=%u and rank=%u.", tag.c_str(), myRank, rank);
            continue;
        }

        std::vector<CommLink> links(linkList, linkList + listSize);
        bool protocolFound = false;
        CHK_RET(ProcessLinkForProtocol(
            comm, expectedProtocols, links, myRank, rank, netLayer, channels, protocolFound, tag));

        if (channels.size() == channelCountBefore) {
            if (linkRequired) {
                HCCL_ERROR(
                    "[%s] No matching protocol intra-frame link between myRank=%u and rank=%u.", tag.c_str(), myRank,
                    rank);
                return HcclResult::HCCL_E_INTERNAL;
            }
            HCCL_WARNING(
                "[%s] No matching protocol inter-frame link between myRank=%u and rank=%u.", tag.c_str(), myRank, rank);
        }
    }
#else
    (void)comm;
    (void)param;
    (void)topoInfo;
    (void)subcommInfo;
    (void)channels;
    (void)netLayer;
    (void)tag;
    (void)linkRequired;
#endif
    return HCCL_SUCCESS;
}

HcclResult CalcChannelRequestMesh1DLevel0(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    const std::vector<std::vector<u32>>& subcommInfo, std::vector<HcclChannelDesc>& channels)
{
    return CalcChannelRequestMesh1DByLevel(
        comm, param, topoInfo, subcommInfo, channels, 0, "CalcChannelRequestMesh1DLevel0", true);
}

HcclResult CalcChannelRequestMesh1DLevel1(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    const std::vector<std::vector<u32>>& subcommInfo, std::vector<HcclChannelDesc>& channels)
{
    return CalcChannelRequestMesh1DByLevel(
        comm, param, topoInfo, subcommInfo, channels, 1, "CalcChannelRequestMesh1DLevel1", false);
}

HcclResult CalcChannelRequestMesh2D(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    const std::vector<std::vector<u32>>& subcommInfo, std::vector<HcclChannelDesc>& channels)
{
#ifndef AICPU_COMPILE
    channels.clear();
    u32 myRank = topoInfo->userRank; // 全局rankId

    std::set<u32> connectRanks;
    if (subcommInfo.size() == 2) { // 2D Mesh
        CHK_RET(CalcMesh2DChannelConnect(myRank, subcommInfo, connectRanks));
    }
#if CANN_VERSION_NUM >= CANN_VERSION(9, 1, 0)
    CommProtocol protocol = CommProtocol::COMM_PROTOCOL_UB_CTP;
    if (param.engine == CommEngine::COMM_ENGINE_AIV) {
        protocol = CommProtocol::COMM_PROTOCOL_UB_MEM;
    }
#else
    // 8.5.0 CANN 无 UB_CTP/UB_MEM 等枚举值；CalcChannelRequestMesh2D 整体属 9.0.0 新特性，
    // 主源已由算子入口 GetHcommVersion() 守护避免运行时调用；8.5.0 下用 HCCS 协议占位仅为可编
    CommProtocol protocol = CommProtocol::COMM_PROTOCOL_HCCS;
    (void)param;
#endif

    for (u32 rank : connectRanks) {
        HcclChannelDesc channelDesc;
        HcclChannelDescInit(&channelDesc, 1);
        channelDesc.remoteRank = rank;
        CommLink* linkList = nullptr;
        u32 listSize;
        CHK_RET(HcclRankGraphGetLinks(comm, 0, myRank, channelDesc.remoteRank, &linkList, &listSize));
        bool protocolExists = false;
        for (u32 idx = 0; idx < listSize; idx++) {
            CommLink link = linkList[idx];
            if (link.linkAttr.linkProtocol == protocol) {
                channelDesc.localEndpoint.protocol = link.srcEndpointDesc.protocol;
                channelDesc.localEndpoint.commAddr = link.srcEndpointDesc.commAddr;
                channelDesc.localEndpoint.loc = link.srcEndpointDesc.loc;
                channelDesc.remoteEndpoint.protocol = link.dstEndpointDesc.protocol;
                channelDesc.remoteEndpoint.commAddr = link.dstEndpointDesc.commAddr;
                channelDesc.remoteEndpoint.loc = link.dstEndpointDesc.loc;
                protocolExists = true;
                HCCL_INFO(
                    "[%s]Add channel request between %zu and %zu with protocol %zu type %u", __func__, myRank,
                    channelDesc.remoteRank, link.dstEndpointDesc.protocol, link.srcEndpointDesc.commAddr.type);
                break;
            }
        }
        CHK_PRT_RET(
            !protocolExists,
            HCCL_ERROR(
                "[%s] protocol[%u] not exists between %zu and %zu", __func__, protocol, myRank, channelDesc.remoteRank),
            HCCL_E_NOT_FOUND);
        channelDesc.channelProtocol = protocol;
        channelDesc.notifyNum = NORMAL_NOTIFY_NUM;
        channels.push_back(channelDesc);
    }
#endif
    return HCCL_SUCCESS;
}

HcclResult ProcessLinkForProtocolNhr(
    HcclComm comm, const std::vector<CommProtocol>& expectedProtocols, const std::vector<CommLink>& linkList,
    u32 myRank, u32 remoteRank, uint32_t netLayer, std::vector<HcclChannelDesc>& channels, bool& protocolFound)
{
    return ProcessLinkForProtocol(
        comm, expectedProtocols, linkList, myRank, remoteRank, netLayer, channels, protocolFound,
        std::string("[CalcLevel1ChannelRequestNhr]"));
}

#ifndef AICPU_COMPILE
static bool IsUbMultiChannelProtocol(CommProtocol protocol)
{
    if (protocol == CommProtocol::COMM_PROTOCOL_UB_CTP) {
        return true;
    }
#if CANN_VERSION_NUM >= CANN_VERSION(9, 2, 0)
    if (protocol == CommProtocol::COMM_PROTOCOL_UB_RTP) {
        return true;
    }
#endif
    return false;
}

static void
DuplicateUbMultiChannelDescs(std::vector<HcclChannelDesc>& channels, size_t channelCountBefore, u32 multiChannelNum)
{
    if (multiChannelNum <= 1) {
        return;
    }
    std::vector<HcclChannelDesc> newChannels(channels.begin() + channelCountBefore, channels.end());
    for (const auto& desc : newChannels) {
        u32 duplicateCount = (IsUbMultiChannelProtocol(desc.channelProtocol)) ? (multiChannelNum - 1) : 0;
        for (u32 n = 0; n < duplicateCount; ++n) {
            channels.push_back(desc);
        }
    }
}
#endif

HcclResult CalcChannelRequestNhr(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    const std::vector<std::vector<u32>>& subcommInfo, std::vector<HcclChannelDesc>& channels)
{
#ifndef AICPU_COMPILE
    (void)param;
    channels.clear();
    u32 multiChannelNum = 1;
    if (param.engine == CommEngine::COMM_ENGINE_AICPU_TS) {
        CHK_RET(GetUbMultiChannelNum(comm, multiChannelNum));
    }
    HCCL_DEBUG(" %s multiChannelNum is %u ", __func__, multiChannelNum);
    std::set<u32> connectRanks;
    u32 myRank = topoInfo->userRank;
    auto it = std::find(subcommInfo[0].begin(), subcommInfo[0].end(), myRank);
    CHK_PRT_RET(
        (it == subcommInfo[0].end()), HCCL_ERROR("[CollAlgFactory] [channel] Rank [%d] is not in commInfo.", myRank),
        HcclResult::HCCL_E_PARA);

    u32 localRank = std::distance(subcommInfo[0].begin(), it);
    u32 localRankSize = subcommInfo[0].size();
    CHK_RET(CalcNHRChannelConnect(localRank, localRankSize, INVALID_VALUE_RANKID, connectRanks));

    // 根据engine获取期望的协议类型列表
    std::vector<CommProtocol> expectedProtocols;
    CHK_RET(GetProtocolByEngine(param, expectedProtocols));

    for (u32 rankIdx : connectRanks) {
        size_t channelCountBefore = channels.size();
        uint32_t* netLayers;
        uint32_t netLayerNum;
        CHK_RET(HcclRankGraphGetLayers(comm, &netLayers, &netLayerNum));
        std::vector<uint32_t> netLayersVector(netLayers, netLayers + netLayerNum);
        bool hasLayerOne = std::find(netLayersVector.begin(), netLayersVector.end(), 1U) != netLayersVector.end();
        for (auto netLayer : netLayersVector) {
            // PCIE-SW场景，需要建立PCIE的clos链路
            bool isNeedLevel0NhrChannel
                = ((topoInfo->level0Topo == Level0Shape::CLOS || topoInfo->level0Topo == Level0Shape::MESH_1D_CLOS)
                   && topoInfo->serverNum == 1 && !hasLayerOne);
            HCCL_INFO(
                "[CalcChannelRequestNhr] isNeedLevel0NhrChannel[%d] Need to calc NHR channel in level0",
                isNeedLevel0NhrChannel);
            if (netLayerNum > 1 && netLayer == 0 && !isNeedLevel0NhrChannel) {
                continue; // 跨框场景，nhr算法只取layer1的链路
            }
            CommLink* linkList = nullptr;
            u32 listSize;
            CHK_RET(HcclRankGraphGetLinks(comm, netLayer, myRank, subcommInfo[0][rankIdx], &linkList, &listSize));

            if (listSize == 0) {
                continue;
            }

            std::vector<CommLink> links(linkList, linkList + listSize);
            bool protocolFound = false;
            CHK_RET(ProcessLinkForProtocolNhr(
                comm, expectedProtocols, links, myRank, subcommInfo[0][rankIdx], netLayer, channels, protocolFound));

            if (channels.size() > channelCountBefore) {
                break;
            }
        }

        HCCL_INFO(
            "[CalcChannelRequestNhr] myRank=%u, remoteRank=%u, channelsCreated=%zu, totalChannels=%zu.", myRank,
            subcommInfo[0][rankIdx], channels.size() - channelCountBefore, channels.size());

        CHK_PRT_RET(
            channels.size() == channelCountBefore,
            HCCL_ERROR(
                "[CalcChannelRequestNhr] Failed to create channel between myRank=%u and rank=%u, there is no link.",
                myRank, subcommInfo[0][rankIdx]),
            HcclResult::HCCL_E_INTERNAL);

        DuplicateUbMultiChannelDescs(channels, channelCountBefore, multiChannelNum);
    }
#endif
    return HCCL_SUCCESS;
}

HcclResult CalcChannelRequestNhrHighestHostRoce(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    const std::vector<std::vector<u32>>& subcommInfo, std::vector<HcclChannelDesc>& channels)
{
#ifndef AICPU_COMPILE
    (void)param;
    channels.clear();
    CHK_PTR_NULL(topoInfo);
    CHK_PRT_RET(
        subcommInfo.empty() || subcommInfo[COMM_LEVEL0].empty(),
        HCCL_ERROR("[CalcChannelRequestNhrHighestHostRoce] subcommInfo is empty."), HCCL_E_PARA);

    const u32 myRank = topoInfo->userRank;
    auto rankIt = std::find(subcommInfo[COMM_LEVEL0].begin(), subcommInfo[COMM_LEVEL0].end(), myRank);
    CHK_PRT_RET(
        rankIt == subcommInfo[COMM_LEVEL0].end(),
        HCCL_ERROR("[CalcChannelRequestNhrHighestHostRoce] Rank [%u] is not in commInfo.", myRank), HCCL_E_PARA);

    const u32 localRank = std::distance(subcommInfo[COMM_LEVEL0].begin(), rankIt);
    const u32 localRankSize = subcommInfo[COMM_LEVEL0].size();
    std::set<u32> connectRankIndexes;
    CHK_RET(CalcNHRChannelConnect(localRank, localRankSize, INVALID_VALUE_RANKID, connectRankIndexes));

    std::vector<u32> remoteRanks;
    remoteRanks.reserve(connectRankIndexes.size());
    for (u32 rankIdx : connectRankIndexes) {
        CHK_PRT_RET(
            rankIdx >= localRankSize,
            HCCL_ERROR(
                "[CalcChannelRequestNhrHighestHostRoce] Invalid rank index [%u], rank size [%u].", rankIdx,
                localRankSize),
            HCCL_E_INTERNAL);
        remoteRanks.push_back(subcommInfo[COMM_LEVEL0][rankIdx]);
    }
    CHK_RET(CalcHighestHostRoceChannels(comm, myRank, remoteRanks, "[CalcChannelRequestNhrHighestHostRoce]", channels));
#endif
    return HCCL_SUCCESS;
}

#if !defined(AICPU_COMPILE) && CANN_VERSION_NUM >= CANN_VERSION(9, 1, 0)
static bool IsEndPointEqual(EndpointDesc& endPoint0, EndpointDesc& endPoint1)
{
    HCCL_INFO(
        "endPoint0:phyId[%u], protocol[%u], addr.type[%u], addr.id[%u]", endPoint0.loc.device.devPhyId,
        endPoint0.protocol, endPoint0.commAddr.type, endPoint0.commAddr.id);
    HCCL_INFO(
        "endPoint1:phyId[%u], protocol[%u], addr.type[%u], addr.id[%u]", endPoint1.loc.device.devPhyId,
        endPoint1.protocol, endPoint1.commAddr.type, endPoint1.commAddr.id);
    if (endPoint0.protocol == CommProtocol::COMM_PROTOCOL_PCIE) {
        return (endPoint0.protocol == endPoint1.protocol) && (endPoint0.commAddr.type == endPoint1.commAddr.type);
    } else {
        return (endPoint0.protocol == endPoint1.protocol) && (endPoint0.commAddr.type == endPoint1.commAddr.type)
               && (memcmp(endPoint0.commAddr.eid, endPoint1.commAddr.eid, sizeof(endPoint0.commAddr.eid)) == 0);
    }
}

static bool IsPortEqual(EndpointDesc& endPoint0, EndpointDesc& endPoint1, bool isIsolation)
{
    HCCL_INFO(
        "[IsPortEqual] eidEndPoint0[%d], eidEndPoint1[%d], isIsolation[%d]", endPoint0.commAddr.eid[PORT_IDX],
        endPoint1.commAddr.eid[PORT_IDX], isIsolation);
    const u32 PORTVAL = 127;
    if (isIsolation) {
        return (
            (endPoint0.commAddr.eid[PORT_IDX] == endPoint1.commAddr.eid[PORT_IDX])
            && (endPoint0.commAddr.eid[PORT_IDX] != PORTVAL));
    } else {
        return (
            (endPoint0.commAddr.eid[PORT_IDX] == endPoint1.commAddr.eid[PORT_IDX])
            && (endPoint0.commAddr.eid[PORT_IDX] == PORTVAL));
    }
}
#endif // CANN_VERSION_NUM >= CANN_VERSION(9, 1, 0)

HcclResult GetTopoTypeByLink(HcclComm comm, uint32_t netLayer, CommLink& link, CommTopo& topoType)
{
#if defined(AICPU_COMPILE) || CANN_VERSION_NUM < CANN_VERSION(9, 1, 0)
    // 9.1.0 之前不使用 HcclRankGraphGetEndpointNum / GetEndpointDesc / GetTopoType 等新 API，
    // 且 CommAddr.eid 字段也不存在；整函数在 8.5.0 下不提供真实实现（上游在 9.0.0 新路径里调用，
    // 入口版本号守护后 8.5.0 永远走不到这里）。
    (void)comm;
    (void)netLayer;
    (void)link;
    (void)topoType;
    return HCCL_SUCCESS;
#else
    uint32_t* topoInstList = nullptr;
    uint32_t listSize;
    CHK_RET(HcclRankGraphGetTopoInstsByLayer(comm, netLayer, &topoInstList, &listSize)); // 获取当前rank的所有TopoInst
    HCCL_INFO("[%s][%u] listSize = %u", __func__, __LINE__, listSize);

    for (uint32_t topoInstIdx = 0; topoInstIdx < listSize; topoInstIdx++) { // 遍历topoInst
        uint32_t topoInstId = topoInstList[topoInstIdx];
        uint32_t endPointNum;
        CHK_RET(HcclRankGraphGetEndpointNum(comm, netLayer, topoInstId, &endPointNum));
        EndpointDesc* endPointDescs = (EndpointDesc*)malloc(endPointNum * sizeof(EndpointDesc));
        if (endPointDescs == nullptr) {
            HCCL_ERROR("Malloc endPointDescs failed!");
            return HCCL_E_PARA;
        }
        HcclResult ret = HCCL_SUCCESS;
        ret = HcclRankGraphGetEndpointDesc(comm, netLayer, topoInstId, &endPointNum, endPointDescs);
        if (ret != HCCL_SUCCESS) {
            free(endPointDescs);
            return ret;
        }

        ret = HcclRankGraphGetTopoType(comm, netLayer, topoInstId, &topoType);
        if (ret != HCCL_SUCCESS) {
            free(endPointDescs);
            return ret;
        }
        HCCL_DEBUG("[%s]topoInstId=%u, endPointNum=%u, topoType=%u", __func__, topoInstId, endPointNum, topoType);
        for (uint32_t endPointIdx = 0; endPointIdx < endPointNum; endPointIdx++) {
            EndpointDesc endPoint = endPointDescs[endPointIdx];
            if (IsEndPointEqual(link.srcEndpointDesc, endPoint)
                == true) { // 当前TopoInst和link的endPoint相同，说明link属于当前TopoInst
                free(endPointDescs);
                return HCCL_SUCCESS;
            }
        }
        HCCL_WARNING("[%s]No Endpoint matches on TopoInst[%u].", __func__, topoInstId);
        free(endPointDescs);
    }
    HCCL_ERROR("[%s]Cannot get TopoType by Link.", __func__);
    return HCCL_E_INTERNAL;
#endif
}

/*
 *   获取link对应的channel。对于2个rank之间，存在多条link的场景，会优先获取指定TopoType的1条channel。
 *   如果多条link都没有指定的TopoType，则返回第一条link对应的channel。
 */
HcclResult ProcessLinksForChannel(
    HcclComm comm, u32 myRank, u32 rank, std::vector<HcclChannelDesc>& channels, CommTopo priorityTopo,
    u32 topoLevelNums)
{
#ifndef AICPU_COMPILE
    uint32_t* netLayers;
    uint32_t netLayerNum;
    CHK_RET(HcclRankGraphGetLayers(comm, &netLayers, &netLayerNum));
    std::vector<uint32_t> netLayersVector(netLayers, netLayers + netLayerNum);
    for (auto netLayer : netLayersVector) {
        CommLink* linkList = nullptr;
        u32 listSize;
        CHK_RET(HcclRankGraphGetLinks(comm, netLayer, myRank, rank, &linkList, &listSize));
        HCCL_INFO("[CalcChannelRequestWithPriorTopo] netLayer=%u, linkListSize=%u", netLayer, listSize);

        if (listSize == 0) {
            HCCL_WARNING(
                "[CalcChannelRequestWithPriorTopo]There is no link between rank[%u] and rank[%u] in layer[%u].", myRank,
                rank, netLayer);
            // 多层拓扑: 跳过当前层继续查找下一层; 单层拓扑: 直接退出
            if (topoLevelNums > 1) {
                continue;
            }
            break;
        }

        uint32_t priorityLink = 0;
        CommTopo topoType;
        if (netLayer == 0) {
            HCCL_INFO("[CalcChannelRequestWithPriorTopo] netLayerNum > 1 && netLayer == 0, select the link according "
                      "to priorityTopo.");
            for (u32 idx = 0; idx < listSize; idx++) {
                CHK_RET(GetTopoTypeByLink(comm, netLayer, linkList[idx], topoType));
                if (topoType == priorityTopo) {
                    priorityLink = idx;
                    HCCL_INFO(
                        "[CalcChannelRequestWithPriorTopo] Found link[%u] with priority topotype[%u].", idx, topoType);
                    break;
                }
            }
        }

        CHK_RET(CreateChannelFromLink(
            comm, myRank, rank, netLayer, priorityLink, linkList[priorityLink], "CalcChannelRequestWithPriorTopo",
            channels));

        if (netLayer == 0) {
            CHK_RET(GetTopoTypeByLink(comm, netLayer, linkList[priorityLink], topoType));
            HCCL_INFO(
                "[CalcChannelRequestWithPriorTopo]Add channel request between %u and %u with protocol %u "
                "and topoType %u. And Priority topoType is %u.",
                myRank, rank, linkList[priorityLink].dstEndpointDesc.protocol, topoType, priorityTopo);
        }

        if (listSize > 0) {
            break;
        }
    }

#endif
    return HCCL_SUCCESS;
}

HcclResult ProcessLinksForChannelMutiJetty(
    HcclComm comm, CommProtocol& expectedProtocol, std::vector<CommLink>& linkList, u32 myRank, u32 remoteRank,
    uint32_t netLayer, std::vector<HcclChannelDesc>& channels, bool execptMesh, bool isIsolation)
{
#ifndef AICPU_COMPILE
    CommTopo topoType;
    constexpr u32 MIN_PHY_COUNT = 8;
    if (linkList.size() < MIN_PHY_COUNT) {
        // 兼容性适配
        isIsolation = false;
    }
    HCCL_INFO(
        "[ProcessLinksForChannelMutiJetty] myRank=%u, remoteRank=%u, netLayer=%u, linkList.size()=%zu, execptMesh=%d, "
        "isIsolation=%d",
        myRank, remoteRank, netLayer, linkList.size(), execptMesh, isIsolation);
    std::vector<HcclChannelDesc> tempChannels;
#if CANN_VERSION_NUM < CANN_VERSION(9, 1, 0)
    // 9.1.0 之前不使用 ProcessLinksForChannelMutiJetty 等新 API，
    // 且 CommAddr.eid 字段也不存在；整函数在 8.5.0 下不提供真实实现（上游在 9.0.0 新路径里调用，
    // 入口版本号守护后 8.5.0 永远走不到这里）。
    (void)comm;
    (void)netLayer;
    (void)linkList;
    return HcclResult::HCCL_E_NOT_SUPPORT;
#else
    for (u32 idx = 0; idx < linkList.size(); idx++) {
        if (linkList[idx].linkAttr.linkProtocol != expectedProtocol) {
            continue;
        }
        CHK_RET(GetTopoTypeByLink(comm, netLayer, linkList[idx], topoType));
        HcclChannelDesc channelDesc;
        HcclChannelDescInit(&channelDesc, 1);
        channelDesc.remoteRank = remoteRank;
        channelDesc.localEndpoint.protocol = linkList[idx].srcEndpointDesc.protocol;
        channelDesc.localEndpoint.commAddr = linkList[idx].srcEndpointDesc.commAddr;
        channelDesc.localEndpoint.loc = linkList[idx].srcEndpointDesc.loc;
        channelDesc.remoteEndpoint.protocol = linkList[idx].dstEndpointDesc.protocol;
        channelDesc.remoteEndpoint.commAddr = linkList[idx].dstEndpointDesc.commAddr;
        channelDesc.remoteEndpoint.loc = linkList[idx].dstEndpointDesc.loc;
        channelDesc.channelProtocol = linkList[idx].srcEndpointDesc.protocol;
        channelDesc.notifyNum = NORMAL_NOTIFY_NUM;
        HCCL_INFO(
            "[CalcChannelRequestMeshClos]Get channel request between %u and %u with protocol %u "
            "and topoType %u.",
            myRank, channelDesc.remoteRank, channelDesc.remoteEndpoint.protocol, topoType);
        if (topoType == CommTopo::COMM_TOPO_CLOS
            && IsPortEqual(linkList[idx].srcEndpointDesc, linkList[idx].dstEndpointDesc, isIsolation)) {
            tempChannels.push_back(channelDesc);
        } else if (topoType == CommTopo::COMM_TOPO_1DMESH && execptMesh) {
            HCCL_INFO("[CalcChannelRequestMeshClos] Clear clos channels and add mesh channel.");
            tempChannels.clear();
            tempChannels.push_back(channelDesc);
            break;
        }
    }
    channels.insert(channels.end(), tempChannels.begin(), tempChannels.end());
    HCCL_INFO(
        "[ProcessLinksForChannelMutiJetty] myRank=%u, remoteRank=%u, channel.size=%zu, ", myRank, remoteRank,
        channels.size());
#endif
#endif
    return HCCL_SUCCESS;
}

HcclResult CalcChannelRequestMesh1DWithPriorityTopo(
    HcclComm comm, const OpParam& param, const TopoInfo* topoInfo, const std::vector<std::vector<u32>>& subcommInfo,
    std::vector<HcclChannelDesc>& channels, CommTopo priorityTopo)
{
#ifndef AICPU_COMPILE
    (void)param;
    channels.clear();
    u32 myRank = topoInfo->userRank;
    auto it = std::find(subcommInfo[COMM_LEVEL0].begin(), subcommInfo[COMM_LEVEL0].end(), myRank);
    CHK_PRT_RET(
        (it == subcommInfo[COMM_LEVEL0].end()),
        HCCL_ERROR("[CollAlgFactory] [channel] Rank [%d] is not in commInfo.", myRank), HcclResult::HCCL_E_PARA);

    u32 topoLevelNums = static_cast<const TopoInfoWithNetLayerDetails*>(topoInfo)->topoLevelNums;
    for (u32 rank : subcommInfo[COMM_LEVEL0]) {
        if (rank != myRank) {
            CHK_RET(ProcessLinksForChannel(comm, myRank, rank, channels, priorityTopo, topoLevelNums));
        }
    }
    HCCL_INFO("[%s] success.", __func__);
#endif
    return HCCL_SUCCESS;
}

HcclResult CalcChannelRequestNHRWithPriorityTopo(
    HcclComm comm, const OpParam& param, const TopoInfo* topoInfo, const std::vector<std::vector<u32>>& subcommInfo,
    std::vector<HcclChannelDesc>& channels, CommTopo priorityTopo)
{
#ifndef AICPU_COMPILE
    (void)param;
    channels.clear();
    auto it = std::find(subcommInfo[COMM_LEVEL0].begin(), subcommInfo[COMM_LEVEL0].end(), topoInfo->userRank);
    CHK_PRT_RET(
        (it == subcommInfo[COMM_LEVEL0].end()),
        HCCL_ERROR("[CollAlgFactory] [channel] Rank [%d] is not in commInfo.", topoInfo->userRank),
        HcclResult::HCCL_E_PARA);

    u32 topoLevelNums = static_cast<const TopoInfoWithNetLayerDetails*>(topoInfo)->topoLevelNums;
    std::set<u32> connectRanks;
    u32 localRank = std::distance(subcommInfo[0].begin(), it);
    u32 localRankSize = subcommInfo[0].size();
    u32 myRank = topoInfo->userRank;
    CHK_RET(CalcNHRChannelConnect(localRank, localRankSize, INVALID_VALUE_RANKID, connectRanks));

    for (u32 rank : connectRanks) {
        if (rank != localRank) {
            CHK_RET(ProcessLinksForChannel(comm, myRank, subcommInfo[0][rank], channels, priorityTopo, topoLevelNums));
        }
    }
    HCCL_INFO("[%s] success.", __func__);
#endif
    return HCCL_SUCCESS;
}

HcclResult CalcChannelRequestNhrMultiJetty(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    const std::vector<std::vector<u32>>& subcommInfo, std::vector<HcclChannelDesc>& channels, bool isIsolation)
{
#ifndef AICPU_COMPILE
    (void)param;
    channels.clear();
    std::set<u32> connectRanks;
    u32 myRank = topoInfo->userRank;
    auto it = std::find(subcommInfo[0].begin(), subcommInfo[0].end(), myRank);
    CHK_PRT_RET(
        (it == subcommInfo[0].end()), HCCL_ERROR("[CollAlgFactory] [channel] Rank [%d] is not in commInfo.", myRank),
        HcclResult::HCCL_E_PARA);

    u32 localRank = std::distance(subcommInfo[0].begin(), it);
    u32 localRankSize = subcommInfo[0].size();
    CHK_RET(CalcNHRChannelConnect(localRank, localRankSize, INVALID_VALUE_RANKID, connectRanks));
#if CANN_VERSION_NUM >= CANN_VERSION(9, 1, 0)
    CommProtocol expectedProtocol = param.engine == CommEngine::COMM_ENGINE_AIV ? CommProtocol::COMM_PROTOCOL_UB_MEM :
                                                                                  CommProtocol::COMM_PROTOCOL_UB_CTP;
#else
    // 8.5.0 CANN 无 UB_CTP/UB_MEM 等枚举值；
    // 主源已由算子入口 GetHcommVersion() 守护避免运行时调用；8.5.0 下用 HCCS 协议占位仅为可编
    CommProtocol expectedProtocol = CommProtocol::COMM_PROTOCOL_HCCS;
#endif
    for (u32 rankIdx : connectRanks) {
        size_t channelCountBefore = channels.size();
        uint32_t* netLayers;
        uint32_t netLayerNum;
        CHK_RET(HcclRankGraphGetLayers(comm, &netLayers, &netLayerNum));
        std::vector<uint32_t> netLayersVector(netLayers, netLayers + netLayerNum);

        for (auto netLayer : netLayersVector) {
            CommLink* linkList = nullptr;
            u32 listSize;
            CHK_RET(HcclRankGraphGetLinks(comm, netLayer, myRank, subcommInfo[0][rankIdx], &linkList, &listSize));
            if (listSize == 0) {
                continue;
            }
            std::vector<CommLink> links(linkList, linkList + listSize);
            if (rankIdx != localRank) {
                CHK_RET(ProcessLinksForChannelMutiJetty(
                    comm, expectedProtocol, links, myRank, subcommInfo[0][rankIdx], netLayer, channels, false,
                    isIsolation));
            }
            if (channels.size() > channelCountBefore) {
                break;
            }
        }

        CHK_PRT_RET(
            channels.size() == channelCountBefore,
            HCCL_ERROR(
                "[CalcChannelRequestNhrMultiJetty] Failed to create channel between myRank=%u and rank=%u, there is no "
                "link.",
                myRank, subcommInfo[0][rankIdx]),
            HcclResult::HCCL_E_INTERNAL);
    }
#endif
    return HCCL_SUCCESS;
}

HcclResult CalcChannelRequestNhrMultiJettyUbx(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    const std::vector<std::vector<u32>>& subcommInfo, std::vector<HcclChannelDesc>& channels)
{
    u64 perDataSize = DATATYPE_SIZE_TABLE[param.DataDes.dataType];
    u64 dataSize = param.DataDes.count * perDataSize;
    bool isIsolation
        = !(IsAllConnetedWithTopo(topoInfo, 0, CommTopo::COMM_TOPO_1DMESH) || dataSize <= SMALL_SIZE_512KB);
    std::vector<HcclChannelDesc> myChannelDescs;
    CHK_RET(CalcChannelRequestNhrMultiJetty(comm, param, topoInfo, subcommInfo, myChannelDescs, isIsolation));
    for (auto channel : myChannelDescs) {
        if (channel.channelProtocol == CommProtocol::COMM_PROTOCOL_UB_CTP) {
            channels.push_back(channel);
        }
    }
    return HCCL_SUCCESS;
}

HcclResult CalcChannelRequestMeshClosMultiJetty(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    const std::vector<std::vector<u32>>& subcommInfo, std::vector<HcclChannelDesc>& channels, bool isIsolation,
    bool execptMesh)
{
#ifndef AICPU_COMPILE
    (void)param;
    channels.clear();
    auto it = std::find(subcommInfo[COMM_LEVEL0].begin(), subcommInfo[COMM_LEVEL0].end(), topoInfo->userRank);
    CHK_PRT_RET(
        (it == subcommInfo[COMM_LEVEL0].end()),
        HCCL_ERROR("[CollAlgFactory] [channel] Rank [%d] is not in commInfo.", topoInfo->userRank),
        HcclResult::HCCL_E_PARA);
    u32 myRank = topoInfo->userRank;
#if CANN_VERSION_NUM >= CANN_VERSION(9, 1, 0)
    CommProtocol expectedProtocol = param.engine == CommEngine::COMM_ENGINE_AIV ? CommProtocol::COMM_PROTOCOL_UB_MEM :
                                                                                  CommProtocol::COMM_PROTOCOL_UB_CTP;
#else
    // 8.5.0 CANN 无 UB_CTP/UB_MEM 等枚举值；
    // 主源已由算子入口 GetHcommVersion() 守护避免运行时调用；8.5.0 下用 HCCS 协议占位仅为可编
    CommProtocol expectedProtocol = CommProtocol::COMM_PROTOCOL_HCCS;
#endif
    for (u32 rank : subcommInfo[COMM_LEVEL0]) {
        if (rank == topoInfo->userRank) {
            continue;
        }
        size_t channelCountBefore = channels.size();
        uint32_t* netLayers;
        uint32_t netLayerNum;
        CHK_RET(HcclRankGraphGetLayers(comm, &netLayers, &netLayerNum));
        std::vector<uint32_t> netLayersVector(netLayers, netLayers + netLayerNum);
        for (auto netLayer : netLayersVector) {
            CommLink* linkList = nullptr;
            u32 listSize;
            CHK_RET(HcclRankGraphGetLinks(comm, netLayer, myRank, rank, &linkList, &listSize));
            if (listSize == 0) {
                continue;
            }
            std::vector<CommLink> links(linkList, linkList + listSize);
            CHK_RET(ProcessLinksForChannelMutiJetty(
                comm, expectedProtocol, links, myRank, rank, netLayer, channels, execptMesh, isIsolation));
            if (channels.size() > channelCountBefore) {
                break;
            }
        }
        CHK_PRT_RET(
            channels.size() == channelCountBefore,
            HCCL_ERROR(
                "[CalcChannelRequestMeshClos] Failed to create channel between myRank=%u and rank=%u, there is no "
                "link.",
                myRank, rank),
            HcclResult::HCCL_E_INTERNAL);
    }
#endif
    return HCCL_SUCCESS;
}

HcclResult CreateChannelRequestByRankId(
    HcclComm comm, const OpParam& param, u32 myRank, u32 remoteRank, std::vector<HcclChannelDesc>& channels,
    u32 channelRepeatNum)
{
#ifndef AICPU_COMPILE
    channels.clear();
    std::vector<CommProtocol> expectedProtocols;
    CHK_RET(GetProtocolByEngine(param, expectedProtocols));

    uint32_t* netLayers;
    uint32_t netLayerNum;
    CHK_RET(HcclRankGraphGetLayers(comm, &netLayers, &netLayerNum));
    std::vector<uint32_t> netLayersVector = std::vector<uint32_t>(netLayers, netLayers + netLayerNum);

    for (auto netLayer : netLayersVector) {
        CommLink* linkList = nullptr;
        u32 listSize;
        CHK_RET(HcclRankGraphGetLinks(comm, netLayer, myRank, remoteRank, &linkList, &listSize));
        if (listSize == 0) {
            continue;
        }
        std::vector<CommLink> links(linkList, linkList + listSize);
        bool protocolFound = false;
        for (u32 i = 0; i < channelRepeatNum; i++) {
            CHK_RET(ProcessLinkForProtocol(
                comm, expectedProtocols, links, myRank, remoteRank, netLayer, channels, protocolFound,
                std::string("[CreateChannelRequestByRankId]")));
        }

        if (channels.size() > 0) {
            break;
        }
    }
    CHK_PRT_RET(
        channels.size() == 0,
        HCCL_ERROR(
            "[CreateChannelRequestByRankId] Failed to create channel between myRank=%u and rank=%u, there is no link.",
            myRank, remoteRank),
        HcclResult::HCCL_E_INTERNAL);

#endif
    return HCCL_SUCCESS;
}
} // namespace ops_hccl
