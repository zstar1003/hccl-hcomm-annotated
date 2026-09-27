/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIV_MC2_COMPONT_H
#define AIV_MC2_COMPONT_H

#include <memory>
#include "communicator_impl.h"
#include "ins_exe_que.h"
#include "mc2_type.h"

namespace Hccl {

class AivMc2Compont {
public:
    explicit AivMc2Compont(CommunicatorImpl* comm) : comm(comm) {}
    ~AivMc2Compont();
    void AllocCommResource(void* mc2Tiling, void** commContext);

private:
    void GenerateCommContext(void** commContext);
    void AivMC2AllocCommRes(Mc2Tiling* mc2TilingPtr) const;
    void AivMC2AllocCommResV2(Mc2InitTilingInner* mc2TilingPtr) const;
    void FillCollOperator(const Mc2CommConfig& config) const;
    void FillCollOperatorV2(const Mc2CcTilingInner& config) const;

private:
    void GenerateAivMemoryCommContext(HcclCombinOpParam& combinOpParam);
    void GenerateAivUrmaCommContext(HcclCombinOpParam& combinOpParam) const;

    CommunicatorImpl* comm;
    std::shared_ptr<DevBuffer> workspaceBuffer{nullptr};
    std::shared_ptr<DevBuffer> combinOpParamBuffer{nullptr};
    std::shared_ptr<DevBuffer> comParamBuffer{nullptr};
    std::shared_ptr<DevBuffer> comSyncBuffer{nullptr};
};
} // namespace Hccl
#endif // AIV_MC2_COMPONT_H
