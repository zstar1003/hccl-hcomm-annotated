/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_STREAM_LITE_MGR_H
#define HCCLV2_STREAM_LITE_MGR_H

#include <vector>
#include "hccl/base.h"
#include "stream_lite.h"
namespace Hccl {
class StreamLiteMgr {
public:
    StreamLite* GetMaster();
    StreamLite* GetSlave(u32 index);
    u32 SizeOfSlaves();
    std::vector<StreamLite*> GetAllStreams();

    void Reset();

    void ParsePackedData(std::vector<char>& givenData);
    ~StreamLiteMgr();

private:
    std::vector<std::unique_ptr<StreamLite>> streams;

    void ParseLiteData(std::vector<char>& data, u32 num, u32 sizePerDto);
};
} // namespace Hccl
#endif
