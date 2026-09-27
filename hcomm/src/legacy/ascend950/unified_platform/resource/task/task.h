/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_TASK_H
#define HCCLV2_TASK_H

#include <map>
#include <string>
#include "data_type.h"
#include "reduce_op.h"
#include "orion_adapter_rts.h"
#include "rts_cnt_notify.h"
#include "rts_1ton_cnt_notify.h"
#include "ipc_local_notify.h"
#include "ipc_remote_notify.h"

namespace Hccl {
using namespace std;
class BaseLocalNotify;
class IpcRemoteNotify;

MAKE_ENUM(MemcpyKind, D2D, H2D, D2H, H2H, ADDR_D2D)

MAKE_ENUM(
    TaskType, LOCAL_RECORD, REMOTE_RECORD, WAIT, WAIT_VALUE, POST_BITS, WAIT_BITS, POST_VALUE, LOCAL_COPY, LOCAL_REDUCE,
    P2P_MEMCPY, SDMA_REDUCE, RDMA_SEND, UB_SEND, LOCAL_ADDR_COPY, UB_DIRECT_SEND, WRITE_VALUE // 二级指针拷贝
)

const std::map<MemcpyKind, rtMemcpyKind_t> MEMCPY_KIND_RT_MAP
    = {{MemcpyKind::D2D, RT_MEMCPY_DEVICE_TO_DEVICE},
       {MemcpyKind::H2D, RT_MEMCPY_HOST_TO_DEVICE},
       {MemcpyKind::D2H, RT_MEMCPY_DEVICE_TO_HOST},
       {MemcpyKind::H2H, RT_MEMCPY_HOST_TO_HOST},
       {MemcpyKind::ADDR_D2D, RT_MEMCPY_ADDR_DEVICE_TO_DEVICE}};

inline rtMemcpyKind_t RtMemcpyKindGet(MemcpyKind kind) { return MEMCPY_KIND_RT_MAP.at(kind); }

const std::map<DataType, aclDataType> DATA_TYPE_RT_MAP = {
    {DataType::INT8, ACL_INT8},           {DataType::INT16, ACL_INT16},
    {DataType::INT32, ACL_INT32},         {DataType::FP16, ACL_FLOAT16},
    {DataType::FP32, ACL_FLOAT},          {DataType::INT64, ACL_DT_UNDEFINED}, // does not support now
    {DataType::UINT64, ACL_DT_UNDEFINED},                                      // does not support now
    {DataType::UINT8, ACL_DT_UNDEFINED},                                       // does not support now
    {DataType::UINT16, ACL_DT_UNDEFINED},                                      // does not support now
    {DataType::UINT32, ACL_DT_UNDEFINED},                                      // does not support now
    {DataType::FP64, ACL_DT_UNDEFINED},                                        // does not support now
    {DataType::BFP16, ACL_BF16},          {DataType::INT128, ACL_DT_UNDEFINED} // does not support now
};

inline aclDataType RtDataTypeGet(DataType type) { return DATA_TYPE_RT_MAP.at(type); }

const std::map<ReduceOp, aclrtReduceKind> REDUCE_OP_RT_MAP
    = {{ReduceOp::SUM, ACL_RT_MEMCPY_SDMA_AUTOMATIC_SUM},
       {ReduceOp::MAX, ACL_RT_MEMCPY_SDMA_AUTOMATIC_MAX},
       {ReduceOp::MIN, ACL_RT_MEMCPY_SDMA_AUTOMATIC_MIN}};

inline aclrtReduceKind RtReduceOpGet(ReduceOp reduceOp) { return REDUCE_OP_RT_MAP.at(reduceOp); }

class BaseTask {
public:
    explicit BaseTask(TaskType type) : type(type), taskId(0), streamId(0) {};
    virtual ~BaseTask() = default;
    virtual std::string Describe() const = 0;

    const TaskType& GetType() const { return type; }

    void SetTaskId(u32 id) { taskId = id; }

    void SetStreamId(u32 id) { streamId = id; }

    inline u32 GetStreamId() const { return streamId; }

    inline u32 GetTaskId() const { return taskId; }

protected:
    TaskType type;
    u32 taskId;
    u32 streamId;
};

class TaskLocalCopy : public BaseTask {
public:
    TaskLocalCopy(u64 dstAddr, u64 srcAddr, u64 size, MemcpyKind kind)
        : BaseTask(TaskType::LOCAL_COPY),
          dstAddr(dstAddr),
          srcAddr(srcAddr),
          kind(kind),
          size(size)
    {}
    std::string Describe() const override;

    inline u64 GetDstAddr() const { return dstAddr; }

    inline u64 GetSrcAddr() const { return srcAddr; }

    inline const MemcpyKind& GetKind() const { return kind; }

    inline u64 GetSize() const { return size; }

private:
    u64 dstAddr;
    u64 srcAddr;
    MemcpyKind kind;
    u64 size;
};

class TaskP2pMemcpy : public BaseTask {
public:
    TaskP2pMemcpy(u64 dstAddr, u64 srcAddr, u64 size, MemcpyKind kind)
        : BaseTask(TaskType::P2P_MEMCPY),
          dstAddr(dstAddr),
          srcAddr(srcAddr),
          kind(kind),
          size(size)
    {}
    std::string Describe() const override;

    u64 GetDstAddr() const { return dstAddr; }
    u64 GetSrcAddr() const { return srcAddr; }

    inline const MemcpyKind& GetKind() const { return kind; }

    inline u64 GetSize() const { return size; }

private:
    u64 dstAddr;
    u64 srcAddr;
    MemcpyKind kind;
    u64 size;
};

class TaskRemoteRecord : public BaseTask {
public:
    explicit TaskRemoteRecord(IpcRemoteNotify* notify) : BaseTask(TaskType::REMOTE_RECORD), notify(notify) {}
    std::string Describe() const override;

    inline const IpcRemoteNotify* GetNotify() const { return notify; }

private:
    IpcRemoteNotify* notify;
};

class TaskWait : public BaseTask {
public:
    explicit TaskWait(BaseLocalNotify* notify) : BaseTask(TaskType::WAIT), notify(notify) {}
    std::string Describe() const override;
    inline const BaseLocalNotify* GetNotify() const { return notify; }

private:
    BaseLocalNotify* notify;
};

class TaskWaitValue : public BaseTask {
public:
    explicit TaskWaitValue(RtsCntNotify* notify, u32 value)
        : BaseTask(TaskType::WAIT_VALUE),
          notify(notify),
          value(value)
    {}
    std::string Describe() const override;
    inline const RtsCntNotify* GetNotify() const { return notify; }
    u32 GetValue() const { return value; }

private:
    RtsCntNotify* notify;
    u32 value;
};

class TaskPostBits : public BaseTask {
public:
    explicit TaskPostBits(RtsCntNotify* notify, u32 bitValue)
        : BaseTask(TaskType::POST_BITS),
          notify(notify),
          bitValue(bitValue)
    {}

    std::string Describe() const override;

    inline const RtsCntNotify* GetNotify() const { return notify; }
    u32 GetValue() const { return bitValue; }

private:
    RtsCntNotify* notify;
    u32 bitValue;
};

class TaskLocalRecord : public BaseTask {
public:
    explicit TaskLocalRecord(BaseLocalNotify* notify) : BaseTask(TaskType::LOCAL_RECORD), notify(notify) {}

    std::string Describe() const override;

    inline const BaseLocalNotify* GetNotify() const { return notify; }

private:
    BaseLocalNotify* notify;
};

class TaskSdmaReduce : public BaseTask {
public:
    TaskSdmaReduce(u64 dstAddr, u64 srcAddr, u64 size, DataType dataType, ReduceOp reduceOp)
        : BaseTask(TaskType::SDMA_REDUCE),
          dstAddr(dstAddr),
          srcAddr(srcAddr),
          size(size),
          dataType(dataType),
          reduceOp(reduceOp)
    {}
    std::string Describe() const override;

    u64 GetDstAddr() const { return dstAddr; }
    u64 GetSrcAddr() const { return srcAddr; }

    inline u64 GetSize() const { return size; }

    inline u64 GetDataCount() const { return size / DataTypeSizeGet(dataType); };

    inline const DataType& GetDataType() const { return dataType; }

    inline const ReduceOp& GetReduceOp() const { return reduceOp; }

private:
    u64 dstAddr;
    u64 srcAddr;
    u64 size;
    DataType dataType;
    ReduceOp reduceOp;
};

class TaskLocalReduce : public BaseTask {
public:
    TaskLocalReduce(u64 srcAddr1, u64 srcAddr2, u64 dstAddr, u64 size, DataType dataType, ReduceOp reduceOp)
        : BaseTask(TaskType::LOCAL_REDUCE),
          srcAddr1(srcAddr1),
          srcAddr2(srcAddr2),
          dstAddr(dstAddr),
          size(size),
          dataType(dataType),
          reduceOp(reduceOp)
    {}
    std::string Describe() const override;

    u64 GetSrcAddr1() const { return srcAddr1; }
    u64 GetSrcAddr2() const { return srcAddr2; }
    u64 GetDstAddr() const { return dstAddr; }

    inline u64 GetDataCount() const { return size / DataTypeSizeGet(dataType); };

    inline const DataType& GetDataType() const { return dataType; }

    inline const ReduceOp& GetReduceOp() const { return reduceOp; }

private:
    u64 srcAddr1;
    u64 srcAddr2;
    u64 dstAddr;
    u64 size;
    DataType dataType;
    ReduceOp reduceOp;
};

class TaskRdmaSend : public BaseTask {
public:
    TaskRdmaSend(u32 dbIndex, u64 dbInfo)
        : BaseTask(TaskType::RDMA_SEND),
          dbIndex(dbIndex),
          dbInfo(dbInfo),
          isTemplateMode(false)
    {}
    TaskRdmaSend(u32 qpn, u32 wqeIndex)
        : BaseTask(TaskType::RDMA_SEND),
          qpn(qpn),
          wqeIndex(wqeIndex),
          isTemplateMode(true)
    {}
    std::string Describe() const override;
    inline u32 GetQpn() const { return qpn; }

    inline u32 GetWqeIndex() const { return wqeIndex; }

    inline u32 GetDbIndex() const { return dbIndex; }

    inline u64 GetDbInfo() const { return dbInfo; }

    inline bool IsTemplateMode() const { return isTemplateMode; }

private:
    u32 qpn{0};      // 910A offload
    u32 wqeIndex{0}; // 910A offload
    u32 dbIndex;     // 910A2/A3 opbase/offload, 910A opbase
    u64 dbInfo;      // 910A2/A3 opbase/offload, 910A opbase
    bool isTemplateMode;
};

class TaskUbDbSend : public BaseTask {
public:
    TaskUbDbSend(u32 jettyId, u32 funcId, u32 piVal, u32 dieId)
        : BaseTask(TaskType::UB_SEND),
          jettyId(jettyId),
          funcId(funcId),
          piVal(piVal),
          dieId(dieId)
    {}
    std::string Describe() const override;
    inline u32 GetJettyId() const { return jettyId; }

    inline u32 GetFuncId() const { return funcId; }

    inline u32 GetPiVal() const { return piVal; }

    inline u32 GetDieId() const { return dieId; }

private:
    u32 jettyId;
    u32 funcId;
    u32 piVal;
    u32 dieId;
};

class TaskLocalAddrCopy : public BaseTask {
public:
    TaskLocalAddrCopy(u64 dstAddr, u64 srcAddr, u64 size)
        : BaseTask(TaskType::LOCAL_ADDR_COPY),
          dstAddr(dstAddr),
          srcAddr(srcAddr),
          size(size)
    {}
    std::string Describe() const override;

    u64 GetDstAddr() const { return dstAddr; }
    u64 GetSrcAddr() const { return srcAddr; }

    inline u64 GetSize() const { return size; }

private:
    u64 dstAddr;
    u64 srcAddr;
    u64 size;
};

constexpr u32 DWQE_MAX_LEN = 128;

class TaskUbDirectSend : public BaseTask {
public:
    TaskUbDirectSend(u32 funcId, u32 dieId, u32 jettyId, u32 dwqeSize, const u8* dwqe);

    std::string Describe() const override;

    u32 GetJettyId() const { return jettyId; }

    u32 GetFuncId() const { return funcId; }

    u32 GetDieId() const { return dieId; }

    u32 GetDwqeSize() const { return dwqeSize; }

    const u8* GetDwqePtr() const { return dwqe; }

private:
    u32 funcId;
    u32 dieId;
    u32 jettyId;
    u32 dwqeSize{0};
    u8 dwqe[DWQE_MAX_LEN]{0};
};

class TaskWriteValue : public BaseTask {
public:
    TaskWriteValue(u64 dbAddr, u32 piVal) : BaseTask(TaskType::WRITE_VALUE), dbAddr(dbAddr), piVal(piVal) {}

    std::string Describe() const override;

    u64 GetDbAddr() const { return dbAddr; }

    u32 GetPiVal() const { return piVal; }

private:
    u64 dbAddr;
    u32 piVal;
};

class TaskPostValue : public BaseTask {
public:
    explicit TaskPostValue(Rts1ToNCntNotify* notify, u32 value)
        : BaseTask(TaskType::POST_VALUE),
          notify(notify),
          value(value)
    {}

    std::string Describe() const override;

    inline const Rts1ToNCntNotify* GetNotify() const { return notify; }
    u32 GetValue() const { return value; }

private:
    Rts1ToNCntNotify* notify;
    u32 value;
};

class TaskWaitBits : public BaseTask {
public:
    explicit TaskWaitBits(Rts1ToNCntNotify* notify, u32 bitValue)
        : BaseTask(TaskType::WAIT_BITS),
          notify(notify),
          bitValue(bitValue)
    {}
    std::string Describe() const override;
    inline const Rts1ToNCntNotify* GetNotify() const { return notify; }
    u32 GetValue() const { return bitValue; }

private:
    Rts1ToNCntNotify* notify;
    u32 bitValue;
};

} // namespace Hccl

#endif
