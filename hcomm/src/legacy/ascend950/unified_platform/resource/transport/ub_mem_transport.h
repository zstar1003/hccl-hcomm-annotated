/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef UB_MEM_TRANSPORT_H
#define UB_MEM_TRANSPORT_H
#include "base_mem_transport.h"
#include "local_cnt_notify.h"
#include "local_ub_rma_buffer.h"
#include "task_param.h"
#include "virtual_topo.h"
#include "ub_local_notify.h"

namespace Hccl {

class UbMemTransport : public BaseMemTransport {
public:
    UbMemTransport(
        CommonLocRes& commonLocRes, Attribution& attr, const LinkData& linkData, const Socket& socket,
        RdmaHandle rdmaHandle1, LocCntNotifyRes& locCntNotifyRes1, bool isRecvFirst);

    UbMemTransport(
        CommonLocRes& commonLocRes, Attribution& attr, const LinkData& linkData, const Socket& socket,
        RdmaHandle rdmaHandle1, LocCntNotifyRes& locCntNotifyRes1,
        std::function<void(u32 streamId, u32 taskId, const TaskParam& taskParam)> callback);

    std::string Describe() const override;
    HcclResult Describe(std::string& dfxMsg);

    HcclResult StatusMachine();
    TransportStatus GetStatus() override;

    std::vector<char> GetUniqueId() override;

    std::vector<char> GetUniqueIdV2();
    std::vector<char> PackConnData();

    vector<char>& GetRmtCntNotifyDesc() override // 仅UB 支持
    {
        return rmtCntNotifyDesc;
    }

    void SetConnVec(std::vector<RmaConnection*>& connectVec) override { commonLocRes.connVec = connectVec; }

    void Post(u32 index, const Stream& stream) override;

    void Wait(u32 index, const Stream& stream, u32 timeout) override;

    void Read(const RmaBufferSlice& locSlice, const RmtRmaBufferSlice& rmtSlice, const Stream& stream) override;

    void ReadReduce(
        const RmaBufferSlice& locSlice, const RmtRmaBufferSlice& rmtSlice, const ReduceIn& reduceIn,
        const Stream& stream) override;

    void Write(const RmaBufferSlice& locSlice, const RmtRmaBufferSlice& rmtSlice, const Stream& stream) override;

    void WriteReduce(
        const RmaBufferSlice& locSlice, const RmtRmaBufferSlice& rmtSlice, const ReduceIn& reduceIn,
        const Stream& stream) override;

    void WriteWithNotify(
        const RmaBufferSlice& locSlice, const RmtRmaBufferSlice& rmtSlice, const WithNotifyIn& withNotify,
        const Stream& stream) override;

    void WriteReduceWithNotify(
        const RmaBufferSlice& locSlice, const RmtRmaBufferSlice& rmtSlice, const ReduceIn& reduceIn,
        const WithNotifyIn& withNotify, const Stream& stream) override;

    u32 GetCurrentStatus() { return static_cast<u32>(baseStatus); }

    HcclResult GetRemoteMems(uint32_t* memNum, CommMem** remoteMem, char*** memInfos);
    HcclResult CheckSocketStatus(std::string socketOpreator);
    HcclResult UpdateMemInfo(std::vector<LocalRmaBuffer*>& bufferVecTemp);

    // hostUb使用
    HcclResult GetRemoteSeg(const void* addr, u64 len, u64* seg);

    HcclResult Init();
    HcclResult DeInit() const;

private:
    RdmaHandle rdmaHandle;

    LocCntNotifyRes locCntNotifyRes;

    MemoryBuffer GetLocMemBuffer(const RmaBufferSlice& locSlice) const;
    MemoryBuffer GetRmtMemBuffer(const RmtRmaBufferSlice& rmtSlice) const;
    MemoryBuffer GetRmtNotifyMemBuffer(u32 index);
    MemoryBuffer GetRmtCntNotifyMemBuffer(const WithNotifyIn& withNotify);

    static constexpr u64 NORMAL_NOTIFY_VAL = 1;

    MAKE_ENUM(UbStatus, INIT, RECV_SIZE, SEND_DATA, RECV_DATA, SEND_FIN, RECV_FIN, PROCESS_DATA, SET_READY, READY)
    UbStatus ubStatus{UbStatus::INIT};
    bool isRecvFirst_{false};

    u32 cntNotifyNum{0};
    u32 cntNotifyDescSize{0};
    vector<char> rmtCntNotifyDesc;

    using RemoteBufferVec = std::vector<std::unique_ptr<RemoteUbRmaBuffer>>;
    using LocalBufferVec = std::vector<LocalUbRmaBuffer*>;

    MAKE_ENUM(UbRmtBufType, NOTIFY, BUFFER, CNT_NOTIFY)

    std::mutex remoteMemsMutex_;                              // 远端内存列表互斥锁
    RemoteBufferVec rmtNotifyVec;                             // 远端普通 notify
    RemoteBufferVec rmtBufferVec;                             // 远端 buffer
    RemoteBufferVec rmtCntNotifyVec;                          // 远端 cnt Notify
    LocalBufferVec locBufferVec;                              // 本端 buffer
    bool cacheValid_ = false;                                 // 当前缓存是否有效
    std::vector<CommMem> remoteUserMems_;                     // 内存基本信息缓存
    std::vector<std::string> memInfoCopies_;                  // 储存 Tag 字符串副本
    std::vector<char*> memInfoPointers_;                      // Tag 缓存
    std::unique_ptr<Hccl::BaseLocalNotify> drainNotify_;      // 本端drain notify
    std::unique_ptr<Hccl::LocalUbRmaBuffer> drainBuffer_;     // 本端常量1 buffer
    std::unique_ptr<Hccl::RemoteUbRmaBuffer> rmtDrainBuffer_; // 对端常量1 buffer

    HcclResult BuildDrainResource();

    HcclResult SendAll();
    HcclResult RecvDataSize();
    HcclResult SendExchangeData();
    HcclResult RecvExchangeData();

    HcclResult SendFinish();
    HcclResult RecvFinish();

    void BufferVecPack(BinaryStream& binaryStream, std::vector<LocalRmaBuffer*>& bufferVec);
    void CntNotifyVecPack(BinaryStream& binaryStream);

    HcclResult DrainBufPack(BinaryStream& binaryStream);
    HcclResult DrainBufUnpack(BinaryStream& binaryStream);

    void CntNotifyDescPack(BinaryStream& binaryStream);
    HcclResult CntNotifyDescUnpack(BinaryStream& binaryStream);

    HcclResult
    RmtBufferVecUnpackProc(u32 locNum, BinaryStream& binaryStream, RemoteBufferVec& bufferVec, UbRmtBufType type);
    HcclResult ConnVecUnpackProc(BinaryStream& binaryStream, bool& needSendFinish);

    void FillRmtRmaBufferVec(RemoteRmaBuffer* rmaBuffer, UbRmtBufType type);

    void SubmitNotify(const MemoryBuffer& rmtNotify, u64 data, const Stream& stream);

    void SubmitWriteEmptyWithNotify(const WithNotifyIn& withNotify, const Stream& stream);

    void SubmitWriteWithNotify(
        const MemoryBuffer& rmt, const MemoryBuffer& loc, u64 data, const MemoryBuffer& rmtNotify,
        const Stream& stream);

    void SubmitWriteReduceWithNotify(
        const MemoryBuffer& rmt, const MemoryBuffer& loc, const ReduceIn& reduceIn, u64 data,
        const MemoryBuffer& rmtNotify, const Stream& stream);

    std::vector<char> GetNotifyUniqueIds();
    std::vector<char> GetRmtBufferUniqueIds(RemoteBufferVec& bufferVec, UbRmtBufType type) const;
    std::vector<char> GetLocBufferUniqueIds(LocalBufferVec& bufferVec, UbRmtBufType type) const;
    std::vector<char> GetSingleRmtBufferUniqueId(u64 addr, u64 size, u32 tokenId, u32 tokenValue, u32 notifyId) const;
    std::vector<char> GetSingleLocBufferUniqueId(u64 addr, u64 size, u32 tokenId, u32 tokenValue) const;
    std::vector<char> GetDrainUniqueIds() const;
    std::vector<char> GetConnUniqueIds();

    bool IsResReady();
    bool IsConnsReady();
    HcclResult RecvDataProcess(bool& needSendFinish);

    HcclResult HandleInitStatus();
    HcclResult HandleSendAllStatus();
    HcclResult HandleRecvSizeStatus();
    HcclResult HandleRecvDataStatus();
    HcclResult HandleProcessDataStatus();
    HcclResult HandleSendFinStatus();
    HcclResult HandleRecvFinStatus();
    HcclResult HandleSetReadyStatus();
    vector<char> recvData{};
    vector<char> recvFinishMsg{};
    vector<char> sendData{};
    vector<char> sendDataPack_{};
    vector<char> sendFinishMsg{};

    void SaveDfxTaskInfo(const TaskParam& taskParam);
};
} // namespace Hccl
#endif
