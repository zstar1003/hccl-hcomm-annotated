/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_TEMPLATE_UTILS
#define HCCLV2_TEMPLATE_UTILS

#include <algorithm>
#include <map>
#include <vector>

#include "data_type.h"
#include "coll_operator.h"
#include "coll_alg_params.h"
#include "op_mode.h"
#include "virtual_topo.h"
#include "connected_link_mgr.h"
#include "dev_capability.h"
#include "primitive.h"
#include "prim_queue.h"
#include "instruction.h"
#include "ins_queue.h"

namespace Hccl {
constexpr int NUM_TWO = 2;
constexpr uint64_t UB_MAX_DATA_SIZE = 256 * 1024 * 1024; // Byte, UB协议一次传输的最大size

// log2 for HD
inline u32 Log2(u32 antilogarithm)
{
    u32 logarithm = 0;
    while ((antilogarithm >> (logarithm + 1)) != 0) {
        logarithm++;
    }

    return logarithm;
}

// judge if both odd or even
inline bool IsSameParity(RankId rank, u32 portId) { return ((static_cast<u32>(rank) % NUM_TWO) == (portId % NUM_TWO)); }

// roundup func for uint
inline u64 RoundUp(u64 dividend, u64 divisor) { return dividend / divisor + ((dividend % divisor != 0) ? 1 : 0); }

using BuffInfo = struct BufferInformation {
    BufferType inBuffType;
    BufferType outBuffType;
    BufferType scratBuffType;
    u64 scratchBuffSize = 0;
    u64 inBuffBaseOff = 0;
    u64 outBuffBaseOff = 0;
    u64 scratchBuffBaseOff = 0;
};

using SliceInfo = struct SliceInformation {
    u64 offset;
    u64 size;
};

struct SendRecvSliceInfo {
    SliceInfo sendSlice;
    SliceInfo recvSlice;
};

using RankSliceInfo = std::vector<std::vector<SliceInfo>>;

// for DMA Copy Elimination
using UsrData = struct UserDataInformation {
    std::vector<DataSlice> usrInSlices;
    std::vector<DataSlice> scratchInSlices;
    std::vector<DataSlice> scratchOutSlices;
    std::vector<DataSlice> usrOutSlices;
};

using A2ASendRecvInfo = struct A2ASendRecvInfoDef {
    // 存放数据长度和偏移长度
    std::vector<u64> sendLength;
    std::vector<u64> sendOffset;
    std::vector<u64> recvLength;
    std::vector<u64> recvOffset;
    // 存放数据个数和偏移个数
    std::vector<u64> sendCounts;
    std::vector<u64> sendDispls;
    std::vector<u64> recvCounts;
    std::vector<u64> recvDispls;
};

// 针对标准的RS temp，准备所需的信息
// rankSize=n 的RS的标准行为：N个Input，Reduce成1个Output;
struct TemplateInfo {
    uint64_t inputAddr;
    uint64_t outputAddr;
    uint64_t scratchAddr;

    BufferType inBuffType;
    BufferType outBuffType;

    uint64_t dataCount;
    DataType inDataType;
    DataType outDataType;
    uint64_t dataSize;

    uint64_t inStride = 0; // in Count
};

struct ParamPool {
    ParamPool(
        const CollAlgOperator& op, const CollAlgParams& params, const u64 scratchSize = 0,
        const AlgTopoInfo* topoInfo = nullptr, const RankGraph* rankGraph = nullptr)
        : op(op),
          params(params),
          scratchSize(scratchSize),
          topoInfo(topoInfo),
          rankGraph(rankGraph)
    {}
    const CollAlgOperator& op;
    const CollAlgParams& params;
    const u64 scratchSize = 0;
    const AlgTopoInfo* topoInfo = nullptr;
    const RankGraph* rankGraph = nullptr;
};

using TempFuncs = struct TemplateFunctionality {
    OpMode opMode;
    bool enableCounterNotify = false;
    bool forAllReduce = false;
    bool forAlgSeqComb = false;
    bool isForepart = false;
    bool isBottom = false;
    bool forAlgConcurrComb = false; // concurrent combination not supported yet, 2024/1/30
    bool forAlgPipeComb = false;    // pipeline combination not supported yet, 2024/1/30
    UsrData usrData;                // pass user memory info for DMA copy elimination
};

using AllignInfo = struct AllignInformation {
    bool enableAllign;
    u64 allignSize;
    DataType dataType;
};

using LinkReq = std::map<RankId, u32>;

using AlgTempResReq = struct AlgTemplateResRequirement {
    std::vector<std::tuple<QId, QId, u32>> queNotifys;
    u32 queNum = 0;
    u32 streamNum = 0;
    LinkReq links; // link requirements
    std::vector<std::pair<QId, u32>> localWaitGroupCntNotify{};
    std::vector<std::pair<QId, u32>> localBcastPostCntNotify{};
    // u64 scratchBufferSize;
};

using ResLinks = std::map<RankId, std::vector<LinkData>>;
using LinkDataIterator = std::vector<LinkData>::const_iterator;

u32 GetNHRStepNum(u32 rankSize);

HcclResult GetUnitAllignSize(const AllignInfo& allignInfo, u64& unitAlignSize);

// convert virtualRank (rankIdx of virtual Topo) to algRank (rankIdx of Alg Template)
HcclResult GetAlgRank(const RankId virtRank, const std::vector<RankId>& tempVTopo, u32& algRank);

// slice calculation shared by ar and rs/ag
HcclResult CalcRsAgSliceInfoConcurrMesh(
    const RankId myRank, const std::vector<std::vector<RankId>>& tempVTopo, const AllignInfo& allignInfo,
    const u64 dataSize, RankSliceInfo& sliceInfoVec);
HcclResult CalcRsAgSliceInfoMesh(
    const RankId myRank, const u32 tempRankSize, const AllignInfo& allignInfo, const u64 dataSize,
    RankSliceInfo& sliceInfoVec);
HcclResult CalcRsAgSliceInfoRing(
    const RankId myRank, const std::vector<std::vector<RankId>>& tempVTopo, const AllignInfo& allignInfo,
    const u64 dataSize, RankSliceInfo& sliceInfoVec);
HcclResult CalcRsAgSliceInfoNHR(
    const RankId myRank, const u32 tempRankSize, const AllignInfo& allignInfo, const u64 dataSize,
    RankSliceInfo& sliceInfoVec);
// slice calculation allreduce
HcclResult CalcSliceInfoAllReduce(
    const AllignInfo& allignInfo, const u32 rankSize, const u64 dataSize, RankSliceInfo& sliceInfoVec);
// res calculation
HcclResult CalcResLinksMesh(
    const RankId myRank, const u32 tempRankSize, const std::vector<std::vector<RankId>>& tempVTopo,
    const u32 linkNumBtwPeers, AlgTempResReq& tempResReq);
HcclResult CalcResLinksMesh2D(
    const RankId myRank, const std::vector<std::vector<RankId>>& tempVTopo, const u32 linkNumBtwPeers,
    AlgTempResReq& tempResReq);
HcclResult CalcResLinksRing(
    const RankId myRank, const u32 tempRankSize, const std::vector<std::vector<RankId>>& tempVTopo,
    AlgTempResReq& tempResReq);
HcclResult CalcResLinksNHR(
    const RankId myRank, const u32 tempRankSize, const std::vector<std::vector<RankId>>& tempVTopo,
    AlgTempResReq& tempResReq);

// get detour send recv links in 4P mesh
HcclResult GetDetourSendRecvLinksIn4P(
    const RankId myRank, const RankId neighborRank, const ResLinks& tempLinks,
    std::vector<std::vector<LinkDataIterator>>& sendRecvLinks);

u32 GetLinkNum(const RankGraph* rankGraph, RankId srcRank, RankId dstRank);

HcclResult GetLocalSendRecvInfoforAlltoall(
    const CollAlgOperator& opParam, const u32 userRank, const u32 userRankSize, A2ASendRecvInfo& localSendRecvInfo);
HcclResult GetLocalSendRecvInfoforAlltoallV(
    const CollAlgOperator& opParam, const u32 userRank, const u32 userRankSize, A2ASendRecvInfo& localSendRecvInfo);
HcclResult GetLocalSendRecvInfoforAlltoallVC(
    const CollAlgOperator& opParam, const u32 userRank, const u32 userRankSize, A2ASendRecvInfo& localSendRecvInfo);
HcclResult GetAlltoAllLocalSendRecvInfo(
    const CollAlgOperator& opParam, const u32 userRank, const u32 userRankSize, A2ASendRecvInfo& localSendRecvInfo);
HcclResult BufferTypeToAddr(const BufferType& bufferType, CollAlgOperator& op, uint64_t& addr);
HcclResult CalcDataSplitRateForLinks(const std::vector<LinkData>& links, std::vector<float>& dataSplitRate);
DataSlice CalcDataSliceForLinks(
    const DataSlice& recvSrcSliceAllLinks, std::vector<float> dataSplitRate, u32 j, DataType dataType_);
} // namespace Hccl

#endif // HCCLV2_COLL_ALG_BASE
