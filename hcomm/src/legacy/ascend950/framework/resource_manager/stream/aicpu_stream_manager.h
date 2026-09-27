/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_AICPU_STREAM_MANAGER_H
#define HCCLV2_AICPU_STREAM_MANAGER_H

#include <vector>
#include <memory>
#include "stream.h"

namespace Hccl {

class AicpuStreamManager {
public:
    explicit AicpuStreamManager() {}
    ~AicpuStreamManager();

    void AllocStreams(u32 num);

    void Clear();

    u32 SizeOfStreams() const { return streams.size(); }

    void AllocFreeStream();

    void AclGraphCaptureFreeStream(const Stream* mainStream) const;

    Stream* GetFreeStream() const { return freeStream.get(); }

    std::vector<char> GetPackedData();

    std::vector<Stream*>& GetStreams() { return stream_pointers; }

private:
    HcclResult CaptureFreeStream(const Stream* mainStream, const Stream* slaveStream) const;
    std::vector<std::unique_ptr<Stream>> streams;
    std::unique_ptr<Stream> freeStream{nullptr};
    std::vector<Stream*> stream_pointers;
};

} // namespace Hccl
#endif
