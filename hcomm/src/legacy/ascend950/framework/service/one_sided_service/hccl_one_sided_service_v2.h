/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_ONE_SIDED_SERVICE_V2_H
#define HCCL_ONE_SIDED_SERVICE_V2_H

#include <shared_mutex>

#include "hccl_one_sided_conn_v2.h"
#include "kernel_param_lite.h"
#include "stream.h"
#include "dev_buffer.h"
#include "coll_alg_params.h"
#include "hccl_net_dev.h"

namespace Hccl {

class HcclOneSidedService {
public:
    using LocalUbRmaBufferMgr = hcomm::RmaBufferMgr<BufferKey<uintptr_t, u64>, std::shared_ptr<LocalUbRmaBuffer>>;
    using RankId = u32;

    HcclOneSidedService(CommunicatorImpl& comm);
    ~HcclOneSidedService();

    HcclResult RegMem(void* addr, u64 size, HcclMemType type, RankId remoteRankId, HcclMemDesc& localMemDesc);
    HcclResult DeregMem(const HcclMemDesc& localMemDesc);

    HcclResult ExchangeMemDesc(
        RankId remoteRankId, const HcclMemDescs& localMemDescs, HcclMemDescs& remoteMemDescs, u32& actualNumOfRemote);

    HcclResult EnableMemAccess(const HcclMemDesc& remoteMemDesc, HcclMem& remoteMem);
    HcclResult DisableMemAccess(const HcclMemDesc& remoteMemDesc);

    HcclResult BatchPut(RankId remoteRankId, const HcclOneSideOpDesc* desc, u32 descNum, const rtStream_t stream);
    HcclResult BatchGet(RankId remoteRankId, const HcclOneSideOpDesc* desc, u32 descNum, const rtStream_t stream);
    LinkData GetLinkData(RankId remoteRankId);

    void AddOpCounterMems();
    DevBuffer* GetOpCounterBuf();

private:
    bool isOpModeReady_{false};
    u32 registeredMemCnt_{0};
    std::unordered_map<RankId, std::shared_ptr<HcclOneSidedConn>> oneSidedConns_{};
    mutable std::shared_mutex oneSidedConnsMutex_;

    std::unordered_map<std::string, std::shared_ptr<LocalUbRmaBuffer>> desc2LocalRdmaRmaBufferMap_{};
    LocalUbRmaBufferMgr localUbRmaBufferMgr_{};

    CommunicatorImpl* comm_{nullptr};
    std::map<RankId, LinkData> linkDataMap_{};

    shared_ptr<DevBuffer> devBatchPutGetLocalBufs{nullptr};
    shared_ptr<DevBuffer> devBatchPutGetRemoteBufs{nullptr};

    std::shared_ptr<DevBuffer> counterBuf{nullptr};
    std::unordered_map<std::string, HcclBuf> desc2HcclBufMapLocalUb_{};
    std::unordered_map<std::string, HcclNetDev> desc2netDevMap_{};

private:
    HcclResult CheckLink(LinkData linkData) const;
    HcclResult CreateConnection(std::shared_ptr<HcclOneSidedConn>& tempConn, LinkData linkData);

    HcclResult RmaMemDescCopyFromStr(RmaMemDesc& rmaMemDesc, const std::vector<char>& memDescStr) const
    {
        if (memcpy_s(rmaMemDesc.memDesc, TRANSPORT_EMD_ESC_SIZE, memDescStr.data(), memDescStr.size()) != EOK) {
            return HCCL_E_INTERNAL;
        }
        return HCCL_SUCCESS;
    }

    // 从 memDesc 转换为 string
    std::string RmaMemDescCopyToStr(const RmaMemDesc& rmaMemDesc) const
    {
        return std::string(rmaMemDesc.memDesc, TRANSPORT_EMD_ESC_SIZE);
    }
    HcclResult
    BatchPutGetDevBufs(const HcclOneSideOpDesc* desc, u32 descNum, std::shared_ptr<HcclOneSidedConn> oneSidedConn);

    HcclResult BatchOpKernelLaunch(
        OpType opType, RankId remoteRankId, const HcclOneSideOpDesc* desc, u32 descNum, shared_ptr<Stream> stream);

    void AddPostToUserStream(const Stream& stream) const;

    void AddWaitToUserStream(const Stream& stream) const;

    std::vector<char> PackOpData(const CollAlgOpReq& req) const;

    void FillOneSidedOperator(OpType type, RankId remoteRankId, const HcclOneSideOpDesc* desc) const;

    DevBuffer* PackResToKernelLanuch(CollAlgOpReq& opReq);

    void SetOneSidedKernelLaunchParam(HcclKernelLaunchParam& param, const DevBuffer* mem) const;

    void OneSidedAicpuKernelLaunch(HcclKernelLaunchParam& param, Stream& stream) const;

    std::unordered_map<std::string, std::shared_ptr<DevBuffer>> OneSidedLoadMap;
};
} // namespace Hccl

#endif // HCCL_ONE_SIDED_SERVICE_V2_H
