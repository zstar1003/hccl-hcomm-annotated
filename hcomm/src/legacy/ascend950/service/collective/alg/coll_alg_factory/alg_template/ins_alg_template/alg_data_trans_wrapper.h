/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_ALG_DATA_TRANS_WRAPPER
#define HCCLV2_ALG_DATA_TRANS_WRAPPER

#include <vector>
#include "data_type.h"
#include "reduce_op.h"

#include "coll_alg_params.h"
#include "virtual_topo.h"
#include "connected_link_mgr.h"
#include "dev_capability.h"
#include "primitive.h"
#include "prim_queue.h"
#include "instruction.h"
#include "ins_queue.h"

namespace Hccl {
using InsQuePtr = std::shared_ptr<InsQueue>;

using SlicesList = struct SlicesListDef {
    std::vector<DataSlice> srcSlices_;
    std::vector<DataSlice> dstSlices_;

    SlicesListDef(const std::vector<DataSlice>& srcSlices, const std::vector<DataSlice>& dstSlices)
        : srcSlices_(srcSlices),
          dstSlices_(dstSlices)
    {}
};

using ReduceSlicesList = struct ReduceSlicesListDef {
    std::vector<DataSlice> srcSlices_;
    std::vector<DataSlice> dstSlices_;
    DataType dataType_;
    ReduceOp reduceOp_;

    ReduceSlicesListDef(
        const std::vector<DataSlice>& srcSlices, const std::vector<DataSlice>& dstSlices, const DataType& dataType,
        const ReduceOp& reduceOp)
        : srcSlices_(srcSlices),
          dstSlices_(dstSlices),
          dataType_(dataType),
          reduceOp_(reduceOp)
    {}
    ReduceSlicesListDef(const SlicesList& slices, const DataType& dataType, const ReduceOp& reduceOp)
        : srcSlices_(slices.srcSlices_),
          dstSlices_(slices.dstSlices_),
          dataType_(dataType),
          reduceOp_(reduceOp)
    {}
};

using TxRxSlicesList = struct TxRxSlicesListDef {
    SlicesList txSlicesList_;
    SlicesList rxSlicesList_;

    TxRxSlicesListDef(const SlicesList& txSlicesList, const SlicesList& rxSlicesList)
        : txSlicesList_(txSlicesList),
          rxSlicesList_(rxSlicesList)
    {}
};

using TxRxReduceSlicesList = struct TxRxReduceSliceInfoDef {
    SlicesList txSlicesList_;
    SlicesList rxSlicesList_;
    DataType dataType_;
    ReduceOp reduceOp_;

    TxRxReduceSliceInfoDef(
        const SlicesList& txSlicesList, const SlicesList& rxSlicesList, const DataType dataType,
        const ReduceOp reduceOp)
        : txSlicesList_(txSlicesList),
          rxSlicesList_(rxSlicesList),
          dataType_(dataType),
          reduceOp_(reduceOp)
    {}

    TxRxReduceSliceInfoDef(const TxRxSlicesList& slices, const DataType dataType, const ReduceOp reduceOp)
        : txSlicesList_(slices.txSlicesList_),
          rxSlicesList_(slices.rxSlicesList_),
          dataType_(dataType),
          reduceOp_(reduceOp)
    {}
};

using TxRxLinks = struct TxRxLinkDef {
    LinkData txLink_;
    LinkData rxLink_;

    TxRxLinkDef(const LinkData& txLink, const LinkData& rxLink) : txLink_(txLink), rxLink_(rxLink) {}
};

using DataInfo = struct DataInfoDef {
    LinkData link_;
    SlicesList slices_;

    DataInfoDef(const LinkData& link, const SlicesList& slices) : link_(link), slices_(slices) {}
};

using SendRecvInfo = struct SendRecvInfoDef {
    TxRxLinks sendRecvLinks_;
    TxRxSlicesList sendRecvSlices_;

    SendRecvInfoDef(const TxRxLinks& sendRecvLinks, const TxRxSlicesList& sendRecvSlices)
        : sendRecvLinks_(sendRecvLinks),
          sendRecvSlices_(sendRecvSlices)
    {}
};

using DataReduceInfo = struct DataReduceInfoDef {
    LinkData link_;
    SlicesList slices_;
    DataType dataType_;
    ReduceOp reduceOp_;

    DataReduceInfoDef(const LinkData& link, const SlicesList& slices, const DataType dataType, const ReduceOp reduceOp)
        : link_(link),
          slices_(slices),
          dataType_(dataType),
          reduceOp_(reduceOp)
    {}
};

using SendRecvReduceInfo = struct SendRecvReduceInfoDef {
    TxRxLinks sendRecvLinks_;
    TxRxSlicesList sendRecvSlices_;
    DataType dataType_;
    ReduceOp reduceOp_;

    SendRecvReduceInfoDef(
        const TxRxLinks& sendRecvLinks, const TxRxSlicesList& sendRecvSlices, const DataType dataType,
        const ReduceOp reduceOp)
        : sendRecvLinks_(sendRecvLinks),
          sendRecvSlices_(sendRecvSlices),
          dataType_(dataType),
          reduceOp_(reduceOp)
    {}
};

using MultiDataInfo = struct MultiDataInfoDef {
    std::vector<LinkData>& links_;
    std::vector<SlicesList>& slices_;

    MultiDataInfoDef(std::vector<LinkData>& links, std::vector<SlicesList>& slices) : links_(links), slices_(slices) {}
};

using MultiSendRecvInfo = struct MultiSendRecvInfoDef {
    std::vector<TxRxLinks>& txRxLinks_;
    std::vector<TxRxSlicesList>& txRxSlices_;

    MultiSendRecvInfoDef(std::vector<TxRxLinks>& txRxLinks, std::vector<TxRxSlicesList>& txRxSlices)
        : txRxLinks_(txRxLinks),
          txRxSlices_(txRxSlices)
    {}
};

using MultiDataReduceInfo = struct MultiDataReduceInfoDef {
    std::vector<LinkData>& links_;
    std::vector<ReduceSlicesList>& slices_;

    MultiDataReduceInfoDef(std::vector<LinkData>& links, std::vector<ReduceSlicesList>& slices)
        : links_(links),
          slices_(slices)
    {}
};

using MultiSendRecvReduceInfo = struct MultiSendRecvReduceInfoDef {
    std::vector<TxRxLinks>& txRxLinks_;
    std::vector<TxRxReduceSlicesList>& txRxSlices_;

    MultiSendRecvReduceInfoDef(std::vector<TxRxLinks>& txRxLinks, std::vector<TxRxReduceSlicesList>& txRxSlices)
        : txRxLinks_(txRxLinks),
          txRxSlices_(txRxSlices)
    {}
};

using SlicePair = struct SlicePairDef {
    DataSlice srcSlice_;
    DataSlice dstSlice_;
    DataType dataType_;
    ReduceOp reduceOp_;

    SlicePairDef(const DataSlice& srcSlice, const DataSlice& dstSlice) : srcSlice_(srcSlice), dstSlice_(dstSlice) {}
};

using TransSlicesInfo = struct TransSlicesInfoDef {
    bool reduceFlag;

    std::vector<DataSlice> srcSlices;
    std::vector<DataSlice> dstSlices;
    DataType dataType_;
    ReduceOp reduceOp_;

    bool enableCounterNotify_;

    explicit TransSlicesInfoDef(const SlicesList& slices, const bool enableCounterNotify = false)
        : reduceFlag(false),
          srcSlices(slices.srcSlices_),
          dstSlices(slices.dstSlices_),
          enableCounterNotify_(enableCounterNotify)
    {}

    TransSlicesInfoDef(
        const SlicesList& slices, const DataType dataType, const ReduceOp reduceOp,
        const bool enableCounterNotify = false)
        : reduceFlag(true),
          srcSlices(slices.srcSlices_),
          dstSlices(slices.dstSlices_),
          dataType_(dataType),
          reduceOp_(reduceOp),
          enableCounterNotify_(enableCounterNotify)
    {}

    explicit TransSlicesInfoDef(const ReduceSlicesList& slices, const bool enableCounterNotify = false)
        : reduceFlag(true),
          srcSlices(slices.srcSlices_),
          dstSlices(slices.dstSlices_),
          dataType_(slices.dataType_),
          reduceOp_(slices.reduceOp_),
          enableCounterNotify_(enableCounterNotify)
    {}
};

using MultiDataLinksDmaModeInfo = struct MultiDataLinksDmaModeInfoDef {
    DmaMode modeNeedSync_;
    DmaMode modeSet_;

    MultiDataLinksDmaModeInfoDef(const DmaMode modeNeedSync, const DmaMode modeSet)
        : modeNeedSync_(modeNeedSync),
          modeSet_(modeSet)
    {}
};

// mid-level
// sync
HcclResult TxReady(const LinkData& link, InsQuePtr queue, u32 topicId = 0, DmaMode dmaMode = DmaMode::DEFAULT);
HcclResult RxReady(const LinkData& link, InsQuePtr queue, u32 topicId = 0, DmaMode dmaMode = DmaMode::DEFAULT);
HcclResult TxFin(const LinkData& link, InsQuePtr queue, u32 topicId = 0, DmaMode dmaMode = DmaMode::DEFAULT);
HcclResult RxFin(const LinkData& link, InsQuePtr queue, u32 topicId = 0, DmaMode dmaMode = DmaMode::DEFAULT);
HcclResult TxFinAck(const LinkData& link, InsQuePtr queue, u32 topicId = 0, DmaMode dmaMode = DmaMode::DEFAULT);
HcclResult RxFinAck(const LinkData& link, InsQuePtr queue, u32 topicId = 0, DmaMode dmaMode = DmaMode::DEFAULT);

// data
HcclResult TxData(const LinkData& link, InsQuePtr queue, const SlicesList& slices, DmaMode dmaMode = DmaMode::DEFAULT);
HcclResult RxData(const LinkData& link, InsQuePtr queue, const SlicesList& slices, DmaMode dmaMode = DmaMode::DEFAULT);
HcclResult
TxReduce(const LinkData& link, InsQuePtr queue, const ReduceSlicesList& slices, DmaMode dmaMode = DmaMode::DEFAULT);
HcclResult
RxReduce(const LinkData& link, InsQuePtr queue, const ReduceSlicesList& slices, DmaMode dmaMode = DmaMode::DEFAULT);

// data with sync
HcclResult TxDataWithFin(
    const LinkData& link, InsQuePtr queue, const SlicesList& slices, u32 topicId = 0,
    DmaMode dmaMode = DmaMode::DEFAULT);
HcclResult RxDataWithFin(
    const LinkData& link, InsQuePtr queue, const SlicesList& slices, u32 topicId = 0,
    DmaMode dmaMode = DmaMode::DEFAULT);
HcclResult TxReduceWithFin(
    const LinkData& link, InsQuePtr queue, const ReduceSlicesList& slices, u32 topicId = 0,
    DmaMode dmaMode = DmaMode::DEFAULT);
HcclResult RxReduceWithFin(
    const LinkData& link, InsQuePtr queue, const ReduceSlicesList& slices, u32 topicId = 0,
    DmaMode dmaMode = DmaMode::DEFAULT);

// data with sync in counterNotify mode
HcclResult MultiTxDataWithFinCounter(
    const std::vector<LinkData>& links, const std::vector<InsQuePtr>& queues, const std::vector<SlicesList>& slices,
    u32 topicId = 0, DmaMode dmaMode = DmaMode::DEFAULT);
HcclResult MultiRxDataWithFinCounter(
    const std::vector<LinkData>& links, const std::vector<InsQuePtr>& queues, const std::vector<SlicesList>& slices,
    u32 topicId = 0, DmaMode dmaMode = DmaMode::DEFAULT);
HcclResult MultiTxReduceWithFinCounter(
    const std::vector<LinkData>& links, const std::vector<InsQuePtr>& queues,
    const std::vector<ReduceSlicesList>& slices, u32 topicId = 0, DmaMode dmaMode = DmaMode::DEFAULT);
HcclResult MultiRxReduceWithFinCounter(
    const std::vector<LinkData>& links, const std::vector<InsQuePtr>& queues,
    const std::vector<ReduceSlicesList>& slices, u32 topicId = 0, DmaMode dmaMode = DmaMode::DEFAULT);

// sync
HcclResult TxRxReady(const TxRxLinks& txRxlinks, InsQuePtr queue, u32 topicId = 0, DmaMode dmaMode = DmaMode::DEFAULT);
HcclResult TxRxFin(const TxRxLinks& txRxlinks, InsQuePtr queue, u32 topicId = 0, DmaMode dmaMode = DmaMode::DEFAULT);
HcclResult TxRxFinAck(const TxRxLinks& txRxlinks, InsQuePtr queue, u32 topicId = 0, DmaMode dmaMode = DmaMode::DEFAULT);

// data
HcclResult TxRxData(
    const TxRxLinks& txRxlinks, InsQuePtr queue, const TxRxSlicesList& txRxSlices, DmaMode dmaMode = DmaMode::DEFAULT);
HcclResult TxRxReduce(
    const TxRxLinks& txRxlinks, InsQuePtr queue, const TxRxReduceSlicesList& txRxSlices,
    DmaMode dmaMode = DmaMode::DEFAULT);

// data with sync
HcclResult TxRxDataWithFin(
    const TxRxLinks& txRxlinks, InsQuePtr queue, const TxRxSlicesList& txRxSlices, u32 topicId = 0,
    DmaMode dmaMode = DmaMode::DEFAULT);
HcclResult TxRxReduceWithFin(
    const TxRxLinks& txRxlinks, InsQuePtr queue, const TxRxReduceSlicesList& txRxSlices, u32 topicId = 0,
    DmaMode dmaMode = DmaMode::DEFAULT);

HcclResult MultiTxRxDataWithFinCounter(
    const std::vector<TxRxLinks>& links, const std::vector<InsQuePtr>& queues,
    const std::vector<TxRxSlicesList>& slices, u32 topicId = 0, DmaMode dmaMode = DmaMode::DEFAULT);
HcclResult MultiTxRxReduceWithFinCounter(
    const std::vector<TxRxLinks>& links, const std::vector<InsQuePtr>& queues,
    const std::vector<TxRxReduceSlicesList>& slices, u32 topicId = 0, DmaMode dmaMode = DmaMode::DEFAULT);

// else
HcclResult LocalCopy(InsQuePtr queue, const DataSlice& srcSlice, const DataSlice& dstSlice);
HcclResult LocalCopySlices(
    InsQuePtr queue, const std::vector<DataSlice>& srcSlices,
    const std::vector<DataSlice>& dstSlices); // 支持连续数据片融合，未来支持strideCount
HcclResult LocalReduce(
    InsQuePtr queue, const DataSlice& srcSlice, const DataSlice& dstSlice, const DataType dataType,
    const ReduceOp reduceOp);
HcclResult LocalReduceSlices(
    InsQuePtr queue, const std::vector<DataSlice>& srcSlices, const std::vector<DataSlice>& dstSlices,
    const DataType dataType, const ReduceOp reduceOp);
HcclResult AicpuReduce(
    InsQuePtr queue, const DataSlice& srcSlice, const DataSlice& dstSlice, const DataType dataType,
    const ReduceOp reduceOp);
HcclResult AicpuReduceSlices(
    InsQuePtr queue, const std::vector<DataSlice>& srcSlices, const std::vector<DataSlice>& dstSlices,
    const DataType dataType, const ReduceOp reduceOp);

HcclResult StreamSync(std::vector<InsQuePtr>& queues);

HcclResult PreSyncQues(
    const std::vector<InsQuePtr>& syncQueues, const u32 postQueIdx, u32 topicId = 0, bool enableCounterNotify = false);
HcclResult PostSyncQues(
    const std::vector<InsQuePtr>& syncQueues, const u32 waitQueIdx, u32 topicId = 0, bool enableCounterNotify = false);

// high-level
HcclResult Send(
    const DataInfo& sendInfo, InsQuePtr queue, u32 topicId = 0, bool needNetFinAck = true,
    DmaMode dmaMode = DmaMode::DEFAULT);
HcclResult Recv(
    const DataInfo& recvInfo, InsQuePtr queue, u32 topicId = 0, bool needNetFinAck = true,
    DmaMode dmaMode = DmaMode::DEFAULT);
HcclResult SendRecv(
    const SendRecvInfo& sendRecvInfo, InsQuePtr queue, u32 topicId = 0, bool needNetFinAck = true,
    DmaMode dmaMode = DmaMode::DEFAULT);

HcclResult SendReduce(
    const DataReduceInfo& sendReduceInfo, InsQuePtr queue, u32 topicId = 0, bool needNetFinAck = true,
    DmaMode dmaMode = DmaMode::DEFAULT);
HcclResult RecvReduce(
    const DataReduceInfo& recvReduceInfo, InsQuePtr queue, u32 topicId = 0, bool needNetFinAck = true,
    DmaMode dmaMode = DmaMode::DEFAULT);
HcclResult SendRecvReduce(
    const SendRecvReduceInfo& sendRecvReduceInfo, InsQuePtr queue, u32 topicId = 0, bool needNetFinAck = true,
    DmaMode dmaMode = DmaMode::DEFAULT);

// Inter-rank CounterNotify is supported when device supports poll cqe
HcclResult MultiSendCounter(
    const MultiDataInfo& sendInfo, std::vector<InsQuePtr>& queues, u32 topicId = 0, DmaMode dmaMode = DmaMode::DEFAULT);
HcclResult MultiRecvCounter(
    const MultiDataInfo& recvInfo, std::vector<InsQuePtr>& queues, u32 topicId = 0, DmaMode dmaMode = DmaMode::DEFAULT);
HcclResult MultiSendRecvCounter(
    const MultiSendRecvInfo& sendRecvInfo, std::vector<InsQuePtr>& queues, u32 topicId = 0,
    DmaMode dmaMode = DmaMode::DEFAULT);

HcclResult MultiSendReduceCounter(
    const MultiDataReduceInfo& sendInfo, std::vector<InsQuePtr>& queues, u32 topicId = 0,
    DmaMode dmaMode = DmaMode::DEFAULT);
HcclResult MultiRecvReduceCounter(
    const MultiDataReduceInfo& recvInfo, std::vector<InsQuePtr>& queues, u32 topicId = 0,
    DmaMode dmaMode = DmaMode::DEFAULT);
HcclResult MultiSendRecvReduceCounter(
    const MultiSendRecvReduceInfo& sendRecvInfo, std::vector<InsQuePtr>& queues, u32 topicId = 0,
    DmaMode dmaMode = DmaMode::DEFAULT);

// send/recv through multi links (support detour case)
HcclResult SendThruMultiLinks(
    const std::vector<DataInfo>& sendInfo, std::vector<InsQuePtr>& queues, u32 topicId = 0, bool needNetFinAck = true,
    DmaMode dmaMode = DmaMode::DEFAULT);
HcclResult RecvThruMultiLinks(
    const std::vector<DataInfo>& recvInfo, std::vector<InsQuePtr>& queues, u32 topicId = 0, bool needNetFinAck = true,
    DmaMode dmaMode = DmaMode::DEFAULT);
HcclResult SendRecvThruMultiLinks(
    const std::vector<SendRecvInfo>& sendRecvInfo, std::vector<InsQuePtr>& queues, u32 topicId = 0,
    bool needNetFinAck = true, DmaMode dmaMode = DmaMode::DEFAULT);

HcclResult SendReduceThruMultiLinks(
    const std::vector<DataReduceInfo>& sendReduceInfo, std::vector<InsQuePtr>& queues, u32 topicId = 0,
    bool needNetFinAck = true, DmaMode dmaMode = DmaMode::DEFAULT);
HcclResult RecvReduceThruMultiLinks(
    const std::vector<DataReduceInfo>& recvReduceInfo, std::vector<InsQuePtr>& queues, u32 topicId = 0,
    bool needNetFinAck = true, DmaMode dmaMode = DmaMode::DEFAULT);
HcclResult SendRecvReduceThruMultiLinks(
    const std::vector<SendRecvReduceInfo>& sendRecvReduceInfo, std::vector<InsQuePtr>& queues, u32 topicId = 0,
    bool needNetFinAck = true, DmaMode dmaMode = DmaMode::DEFAULT);

// auxiliary functions
HcclResult GetDMAMode(const DmaMode setMode, const PortDeploymentType linkPortType, DmaMode& mode);
bool IsContinuousSlice(const DataSlice& nxtSlice, const DataSlice& currSlice);

void TransSlice(const LinkData& link, InsQuePtr queue, const SlicePair& txRxSlice, DmaMode dmaMode, bool reduceFlag);
HcclResult TransSlicesLists(const LinkData& link, InsQuePtr queue, const TransSlicesInfo& slices, DmaMode dmaMode);

HcclResult WriteSlicesListsWithFin(const LinkData& link, InsQuePtr queue, const TransSlicesInfo& slices, u32 topicId);

// for high-level wrapper function
HcclResult ProceedMultiLinks(
    const std::vector<DataInfo>& dataInfo, const std::vector<InsQuePtr>& queues,
    const MultiDataLinksDmaModeInfo& dmaModeInfo, std::vector<InsQuePtr>& syncQues, bool& hasDiffDmaMode);
HcclResult ProceedMultiLinks(
    const std::vector<DataReduceInfo>& dataInfo, const std::vector<InsQuePtr>& queues,
    const MultiDataLinksDmaModeInfo& dmaModeInfo, std::vector<InsQuePtr>& syncQues, bool& hasDiffDmaMode);
} // namespace Hccl

#endif // !HCCLV2_ALG_DATA_TRANS_WRAPPER
