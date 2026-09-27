/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COLL_RUN_ALLTOALLV_TWO_LEVEL_PIPELINE_H
#define COLL_RUN_ALLTOALLV_TWO_LEVEL_PIPELINE_H
#include "coll_all_to_all_executor.h"
namespace hccl {
class CollRunAlltoAllVStaged : public CollAlltoAllExecutor {
public:
    CollRunAlltoAllVStaged(const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher);
    ~CollRunAlltoAllVStaged() override = default;

    HcclResult CheckNeedRecreateComm(u64 lastScratchMemSize, bool& needRecreateAlltoallComm) override;
    HcclResult CheckNeedCreateVirtualLinks(AlgResourceRequest& resourceRequest) override;
    HcclResult ParallelTaskLoaderProcess(
        const std::string& tag, Stream& stream, SubCommInfo& level0CommInfo, std::vector<Stream>& ringStreams) const;

private:
    HcclResult CalcStreamNum(u32& streamNum) override;
    void CalcWorkSpaceMemSize(
        const AlltoAllUserRankInfo& userRankInfo, const std::vector<SendRecvInfo>& allMeshAggregationSendRecvInfo,
        u64& workspaceMemSize, u32 meshAggregationRankSize) const;
    HcclResult CalcScratchMemSize(u64& scratchMemSize) override;

    HcclResult CalcLevel0CommInfo(
        TransportMemType inputType, TransportMemType outputType,
        std::vector<LevelNSubCommTransport>& opTransport) override;
    HcclResult CalcLevel1CommInfo(
        TransportMemType inputType, TransportMemType outputType,
        std::vector<LevelNSubCommTransport>& opTransport) override;
    HcclResult CalcLevel2CommInfo(
        TransportMemType inputType, TransportMemType outputType,
        std::vector<LevelNSubCommTransport>& opTransport) override;
    HcclResult CalStagedAlltoallVCommInfo(
        TransportMemType inputType, TransportMemType outputType, std::vector<LevelNSubCommTransport>& opTransport);
    HcclResult CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport) override;
    HcclResult KernelRun(const OpParam& param, ExecMem& execMem) override;

    HcclResult PrepareAlltoAllVStaged1(
        DeviceMem& sendBuf, DeviceMem& recvBuf, DeviceMem& scratchMem,
        std::map<u32, std::list<OneSendRecvAddrInfo>>& sendAddrInfosIntra,
        std::map<u32, std::list<OneSendRecvAddrInfo>>& recvAddrInfosIntra, Stream& stream, const std::string& tag,
        std::unique_ptr<AlgTemplateBase>& alltoallLevel0, ExecMem& execMem);
    void CalcInterMeshAggregationRecvRemoteOffset(
        const AlltoAllUserRankInfo& userRankInfo, const std::vector<SendRecvInfo>& allSendRecvInfo, u32 index,
        u64& remoteOffset, u32 meshAggregationRankSize) const;
    void CalcInterMeshAggregationAlltoAllMemInfo(
        const AlltoAllUserRankInfo& userRankInfo, const std::vector<SendRecvInfo>& allSendRecvInfo,
        std::map<u32, std::list<OneSendRecvAddrInfo>>& sendAddrInfosInter,
        std::map<u32, std::list<OneSendRecvAddrInfo>>& recvAddrInfosInter, u32 meshAggregationRankSize);
    HcclResult PrepareAlltoAllVStaged2(
        DeviceMem& recvBuf, DeviceMem& scratchMem, std::map<u32, std::list<OneSendRecvAddrInfo>>& sendAddrInfosInter,
        std::map<u32, std::list<OneSendRecvAddrInfo>>& recvAddrInfosInter, Stream& stream, const std::string& tag,
        std::unique_ptr<AlgTemplateBase>& alltoallLevel1, ExecMem& execMem) const;
};

} // namespace hccl

#endif
