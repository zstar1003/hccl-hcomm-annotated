/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_HDC_PUB_H
#define HCCL_HDC_PUB_H

#include <mutex>
#include <memory>
#include <shared_mutex>
#include "hccl/base.h"
#include "mem_device_pub.h"
#include "mem_host_pub.h"

#define HCCL_HDC_TYPE_D2H 0
#define HCCL_HDC_TYPE_H2D 1

namespace hccl {
struct HDCommunicateParams {
    u64 hostAddr{0};
    u64 deviceAddr{0};
    u64 readCacheAddr{0};
    u32 devMemSize{0};
    u32 buffLen{0};
    u32 flag{0};
};

// NOTE:
// HDCommunicate提供host和device之间的单向通道；如接收端未及时读取buffer中的数据，可能会导致数据丢失，需要使用者自行确保收发两端的应答确认机制。

class HDCommunicate {
public:
    HDCommunicate(u32 deviceLogicId, u32 flag, u32 buffLen = 4096);
    HDCommunicate();
    ~HDCommunicate();

    HcclResult InitHost();

    struct HDCommunicateParams GetCommunicateParams();

    HcclResult InitDevice(const struct HDCommunicateParams& params);

    HcclResult Put(u32 offset, u32 length, u8* value);

    HcclResult Get(u32 offset, u32 length, u8* value);

private:
    HcclResult VerifyDeviceMemoryRegisterSupport();
    HcclResult AllocShm(u32 devid, DeviceMem& devShm, HostMem& hostShm);
    HcclResult AllocReadCache(u32 flag, void*& readCacheAddr);
    HcclResult Write(u32 offset, u32 length, u8* value);
    HcclResult Read(u32 offset, u32 length, u8* value);
    HcclResult UpdateCache(u32 timeoutSec);
    DeviceMem devMem_;
    HostMem hostMem_;
    DeviceMem devCache_;
    HostMem hostCache_;
    u32 deviceLogicId_;
    u32 flag_;
    u32 buffLen_;

    void* readCacheAddr_{nullptr};
    u32* headCntAddr_{nullptr};
    u32* tailCntAddr_{nullptr};
    u32* devHeadCntAddr_{nullptr};
    u32* devTailCntAddr_{nullptr};
    bool isHost_{true};
    bool supportDevMemReg_{true};
    std::shared_mutex lock_;
};

} // namespace hccl
#endif // HCCL_HDC_PUB_H
