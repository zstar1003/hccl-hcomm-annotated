/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef PRIVATE_TYPES_H
#define PRIVATE_TYPES_H
#include "hccl_common.h"
#include "transport_heterog_def.h"
#include "adapter_hccp.h"
using HcclRtPointAttr = void*; // 获取指针属性，主要是页表大小
using BinHandle = void*;
constexpr u32 NOTIFY_INFO_LENGTH = 256;

constexpr u32 HETEROG_RANK_TABLE_MAX_LEN = 10240; // ranktable max len 10 * 1024 = 10KB
namespace hccl {

using MemMsg = struct TagMemMsg {
    void* addr;
    u64 len;
    MemType memType;
    s32 mrRegFlag;
    u64 offset;
    u32 rkey;
    u32 lkey;
    u32 notifyId;

    TagMemMsg()
        : addr(nullptr),
          len(0),
          memType(MemType::MEM_TYPE_RESERVED),
          mrRegFlag(0),
          offset(0),
          rkey(0),
          lkey(0),
          notifyId(INVALID_UINT)
    {}
};

struct TransportResourceInfo {
    static constexpr s32 QP_FLAG_RC = 0;
    static constexpr s32 NORMAL_QP_MODE = 0;
    static constexpr s32 DEFAULT_DEVICE_LOGIC_ID = 0;
    static constexpr s32 DEFAULT_LKEY_VALUE = 0;
    static constexpr s32 DEFAULT_BATCH_NUM = 128;

    const std::unique_ptr<MrManager>& mrManager;
    const std::unique_ptr<LocklessRingMemoryAllocate<HcclMessageInfo>>& pMsgInfosMem;
    const std::unique_ptr<LocklessRingMemoryAllocate<HcclRequestInfo>>& pReqInfosMem;
    const std::unique_ptr<HeterogMemBlocksManager>& memBlocksManager;
    const std::unique_ptr<LocklessRingMemoryAllocate<RecvWrInfo>>& pRecvWrInfosMem;
    SrqInfo tagSrqInfo;
    SrqInfo dataSrqInfo;
    s32 flag;
    u32 lkey;
    s32 qpMode;
    s32 deviceLogicId;
    bool isHdcMode;
    u32 memBlockNum;
    bool remoteIsHdc;
    bool isESMode;
    bool isGlobalMrmanagerInit;
    u32 hdcHostWqeBatchNum;
    bool isRawConn{false};
    TransportResourceInfo() = delete;
    TransportResourceInfo(
        const std::unique_ptr<MrManager>& mrManager,
        const std::unique_ptr<LocklessRingMemoryAllocate<HcclMessageInfo>>& pMsgInfosMem,
        const std::unique_ptr<LocklessRingMemoryAllocate<HcclRequestInfo>>& pReqInfosMem,
        const std::unique_ptr<HeterogMemBlocksManager>& memBlocksManager,
        const std::unique_ptr<LocklessRingMemoryAllocate<RecvWrInfo>>& pRecvWrInfosMem)
        : mrManager(mrManager),
          pMsgInfosMem(pMsgInfosMem),
          pReqInfosMem(pReqInfosMem),
          memBlocksManager(memBlocksManager),
          pRecvWrInfosMem(pRecvWrInfosMem),
          tagSrqInfo(SrqInfo()),
          dataSrqInfo(SrqInfo()),
          flag(QP_FLAG_RC),
          lkey(0),
          qpMode(NORMAL_QP_MODE),
          deviceLogicId(DEFAULT_DEVICE_LOGIC_ID),
          isHdcMode(false),
          memBlockNum(MEM_BLOCK_NUM),
          remoteIsHdc(false),
          isESMode(false),
          isGlobalMrmanagerInit(false),
          hdcHostWqeBatchNum(DEFAULT_BATCH_NUM)
    {}
    TransportResourceInfo(const struct TransportResourceInfo& that)
        : mrManager(that.mrManager),
          pMsgInfosMem(that.pMsgInfosMem),
          pReqInfosMem(that.pReqInfosMem),
          memBlocksManager(that.memBlocksManager),
          pRecvWrInfosMem(that.pRecvWrInfosMem)
    {
        tagSrqInfo = (that.tagSrqInfo);
        dataSrqInfo = (that.dataSrqInfo);
        flag = (that.flag);
        lkey = (that.lkey);
        qpMode = (that.qpMode);
        deviceLogicId = (that.deviceLogicId);
        isHdcMode = (that.isHdcMode);
        memBlockNum = (that.memBlockNum);
        remoteIsHdc = false;
        isESMode = (that.isESMode);
        isGlobalMrmanagerInit = (that.isGlobalMrmanagerInit);
        hdcHostWqeBatchNum = (that.hdcHostWqeBatchNum);
        isRawConn = (that.isRawConn);
    }
};
// 全局工作空间类型
enum class GlobalWorkSpaceType {
    OVERFLOW_DETECT_MODE = 0,
};

template <typename T>
inline std::vector<u8> CustomTypeToVectorByte(T& data)
{
    std::vector<u8> v((reinterpret_cast<u8*>(&data)), (reinterpret_cast<u8*>(&data) + sizeof(T)));
    return v;
}
} // namespace hccl

#endif /* PRIVATE_TYPES_H */
