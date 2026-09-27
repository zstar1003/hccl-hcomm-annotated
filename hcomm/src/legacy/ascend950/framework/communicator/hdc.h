/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_HDC_H
#define HCCLV2_HDC_H

#include <mutex>
#include <memory>
#include "hdc_param.h"
#include "dev_buffer.h"
#include "host_buffer.h"
#include "types.h"

namespace Hccl {
// HDCommunicate为host侧提供host和device之间的单向通道；如接收端未及时读取buffer中的数据，可能会导致数据丢失，需要使用者自行确保收发两端的应答确认机制。

HcclResult HrtDrvMemCpy(void* dst, uint64_t destMax, const void* src, uint64_t count);

class HDCommunicate {
public:
    HDCommunicate() = default;
    HDCommunicate(u32 deviceLogicId, u32 flag, u32 buffLen = 4096);
    ~HDCommunicate();

    HcclResult Init();

    struct HDCommunicateParams GetCommunicateParams() const;

    HcclResult Put(u32 offset, u32 length, u8* value);

    HcclResult Get(u32 offset, u32 length, u8* value);

private:
    HcclResult VerifyDeviceMemoryRegisterSupport();
    HcclResult AllocShm();
    HcclResult AllocReadCache();
    HcclResult Write(u32 offset, u32 length, u8* value);
    HcclResult Read(u32 offset, u32 length, u8* value);
    HcclResult UpdateCache(u32 timeoutSec);

    std::unique_ptr<DevBuffer> devMem;
    std::unique_ptr<HostBuffer> hostMem;
    std::unique_ptr<DevBuffer> devCache;
    std::unique_ptr<HostBuffer> hostCache;
    u32 deviceLogicId{0xFFFFFFFF};
    u32 flag{HCCLV2_HDC_TYPE_D2H};
    u32 buffLen{0};

    void* readCacheAddr{nullptr};
    u32* headCntAddr{nullptr};
    u32* tailCntAddr{nullptr};
    u32* devHeadCntAddr{nullptr};
    u32* devTailCntAddr{nullptr};
    bool supportDevMemReg{true}; // device内存直接映射到host，可以提升性能。目前暂不支持
    std::mutex shmLock;
};
} // namespace Hccl
#endif // HCCLV2_HDC_H
