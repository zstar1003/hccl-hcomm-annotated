/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIV_INS_PREPROCESSOR_H
#define AIV_INS_PREPROCESSOR_H

#include "ins_queue.h"
#include "communicator_impl.h"
#include "mc2_type.h"
#include "connections_builder.h"

namespace Hccl {

class AivInsPreprocessor {
public:
    using InsIterator = HierarchicalQueue<Instruction, InsQueue>::Iterator;

    explicit AivInsPreprocessor(CommunicatorImpl* comm) : comm(comm) {}

    ~AivInsPreprocessor();

    void Preprocess(std::shared_ptr<InsQueue>& insQueue) const;
    std::vector<HcclAiRMAWQ> GetWqs() const;
    std::vector<HcclAiRMACQ> GetCqs() const;
    void SetProtocol(uint8_t protocol);
    uint8_t GetProtocol() const;

private:
    void InsPreprocess(InsIterator& insIter) const;
    void BatchBuildTransports(const vector<LinkData>& links) const;
    void BatchBuildUrmaTransports(const vector<LinkData>& links) const;

private:
    CommunicatorImpl* comm;
    uint8_t protocol_{0};
};

} // namespace Hccl

#endif // AIV_INS_PREPROCESSOR_H
