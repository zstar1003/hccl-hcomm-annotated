/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_CCU_CONTEXT_ALL_TO_ALL_V_MESH_1D_H_
#define HCCLV2_CCU_CONTEXT_ALL_TO_ALL_V_MESH_1D_H_

#include <vector>

#include <ios>
#include "log.h"
#include "ccu_context_alg_base.h"
#include "ccu_datatype.h"
#include "hccl_params_pub.h"

namespace Hccl {

class CcuContextAllToAllVMesh1D : public CcuContextAlgBase {
public:
    CcuContextAllToAllVMesh1D(
        const CcuCtxArg& arg, const std::vector<CcuTransport*>& transports, const CcuTransportGroup& group);
    ~CcuContextAllToAllVMesh1D() override {}

    void Algorithm() override;
    std::vector<uint64_t> GeneArgs(const CcuTaskArg& arg) override;
    static void RefreshArgs(CollOpParams opParams, u32 rankSize, std::vector<uint64_t>& args, const u32 myRank);

protected:
    // a2a 对每个对端的发送接收信息
    struct A2AsingleSendRecvInfo {
        CcuRep::Variable tailSize;
        CcuRep::Variable loopNum;
        CcuRep::Variable sendOffset;
        CcuRep::Variable recvOffset;
        GroupOpSize tailGoSize;
    };
    void CreateVariables();
    void LoadAll2allSendRecvInfo(std::vector<A2AsingleSendRecvInfo>& sendRecvInfo);
    void LoadArgs();
    void PreSync();
    void PostSync();
    void DoAll2AllVMultiLoop();
    void CalcGroupSrcDst();

private:
    std::vector<CcuRep::Variable> input_;
    std::vector<CcuRep::Variable> output_;
    std::vector<CcuRep::Variable> token_;
    CcuRep::Variable srcOffset_;
    CcuRep::Variable dstOffset_;
    std::vector<A2AsingleSendRecvInfo> sendRecvInfo_;

    CcuRep::Variable a2avXnAddr_;
    bool loadFromMem = false;

    std::vector<CcuRep::Memory> src_;
    std::vector<CcuRep::Memory> dst_;

    uint16_t selfBit_{0};
    uint16_t allBit_{0};
    uint16_t allOtherBit_{0};

    CcuRep::MaskSignal locMask_;
    CcuRep::Variable completedRankCount_;
    CcuRep::Variable xnMaxTransportSize_;
    GroupOpSize xnMaxTransportGoSize_;
    GroupOpSize localTailGoSize_;
    CcuRep::Variable xnConst1_;

    // 以下参数用于加载 MC2 参数
    CcuRep::Variable xnLength_;
};
} // namespace Hccl

#endif // HCCLV2_CCU_CONTEXT_ALL_TO_ALL_V_MESH_1D_H_
