/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TRANSPORT_PUB_H
#define TRANSPORT_PUB_H

#include <initializer_list>
#include <vector>
#include <hccl/hccl_types.h>
#include "hccl_common.h"
#include "sal_pub.h"
#include "adapter_pub.h"
#include "stream_pub.h"
#include "dispatcher.h"
#include "hccl_socket.h"
#include "notify_pool.h"
#include "local_notify.h"
#include "remote_notify.h"
#include "hccl_mem_defs.h"
#include "hcomm_primitives.h"

enum class DBMode : s32 { INVALID_DB = -1, HW_DB = 0, SW_DB };

struct HcclAiRMAWQ {
    u32 wqn;
    u64 bufAddr;
    u32 wqeSize;
    u32 depth;
    u64 headAddr;
    u64 tailAddr;
    DBMode dbMode; // 0-hw/1-sw
    u64 dbAddr;
    u32 sl;
    HcclAiRMAWQ()
        : wqn(0),
          bufAddr(0),
          wqeSize(0),
          depth(0),
          headAddr(0),
          tailAddr(0),
          dbMode(DBMode::INVALID_DB),
          dbAddr(0),
          sl(0)
    {}
};

struct HcclAiRMACQ {
    u32 cqn;
    u64 bufAddr;
    u32 cqeSize;
    u32 depth;
    u64 headAddr;
    u64 tailAddr;
    DBMode dbMode; // 0-hw/1-sw
    u64 dbAddr;
    HcclAiRMACQ()
        : cqn(0),
          bufAddr(0),
          cqeSize(0),
          depth(0),
          headAddr(0),
          tailAddr(0),
          dbMode(DBMode::INVALID_DB),
          dbAddr(0)
    {}
};

struct HcclAiRMAQueueInfo {
    struct HcclAiRMAWQ sq;
    struct HcclAiRMAWQ rq;
    struct HcclAiRMACQ scq;
    struct HcclAiRMACQ rcq;
};

#pragma pack(push)
#pragma pack(4)
struct HcclQpInfoV2 {
    u64 qpPtr;
    u32 sqIndex;
    u32 dbIndex;
    u16 retryCnt{0};
    u16 retryTime{0};

    HcclQpInfoV2() : qpPtr(0), sqIndex(0), dbIndex(0), retryCnt(0), retryTime(0) {}
    HcclQpInfoV2(const HcclQpInfoV2& other)
        : qpPtr(other.qpPtr),
          sqIndex(other.sqIndex),
          dbIndex(other.dbIndex),
          retryCnt(other.retryCnt),
          retryTime(other.retryTime)
    {}
    HcclQpInfoV2(HcclQpInfoV2&& other)
        : qpPtr(other.qpPtr),
          sqIndex(other.sqIndex),
          dbIndex(other.dbIndex),
          retryCnt(other.retryCnt),
          retryTime(other.retryTime)
    {}
    HcclQpInfoV2& operator=(const HcclQpInfoV2& other)
    {
        if (&other != this) {
            qpPtr = other.qpPtr;
            sqIndex = other.sqIndex;
            dbIndex = other.dbIndex;
            retryCnt = other.retryCnt;
            retryTime = other.retryTime;
        }
        return *this;
    }
    HcclQpInfoV2& operator=(HcclQpInfoV2&& other)
    {
        if (&other != this) {
            qpPtr = other.qpPtr;
            sqIndex = other.sqIndex;
            dbIndex = other.dbIndex;
            retryCnt = other.retryCnt;
            retryTime = other.retryTime;
        }
        return *this;
    }
};
#pragma pack(pop)

struct AddrKey {
    u64 addr = 0;
    u32 key = 0;
    u32 notifyId = INVALID_UINT;
};

/**
 * AICPU TS RoCE：下发 device 的 MR 元数据；亦为 TransportDeviceIbverbs 区间查表 value。
 * addr：同 RmaBuffer::GetAddr()（HOST 为主机 VA，DEVICE 为设备 VA）。
 * devAddr：MR 用设备 VA（GetDevAddr()）；HOST 映射后与 addr 不同，DEVICE 时与 addr 相同。
 */
struct RoceMemDetails {
    u64 addr = 0;
    u64 devAddr = 0;
    u64 size = 0;
    u32 key = 0;
};

struct MemDetails {
    u64 size = 0;
    u64 addr = 0;
    u32 key = 0;

    MemDetails() {}
    MemDetails(const MemDetails& that) : size(that.size), addr(that.addr), key(that.key) {}

    MemDetails(MemDetails&& that) : size(that.size), addr(that.addr), key(that.key) {}
    MemDetails& operator=(const MemDetails& that)
    {
        if (&that != this) {
            size = that.size;
            addr = that.addr;
            key = that.key;
        }
        return *this;
    }

    MemDetails operator=(MemDetails&& that)
    {
        if (&that != this) {
            size = that.size;
            addr = that.addr;
            key = that.key;
        }
        return *this;
    }
};

namespace hccl {

class TransportBase;
struct RxMemoryInfo;
struct TxMemoryInfo;
enum class UserMemType;
class DeviceMem;
class MemNameRepository;

enum class MachineType { MACHINE_SERVER_TYPE, MACHINE_CLIENT_TYPE, MACHINE_RESERVED_TYPE };
enum class LinkMode { LINK_SIMPLEX_MODE, LINK_DUPLEX_MODE, LINK_RESERVED_MODE };

// signal record 使用的value的内存信息
// 使用SDMDA或者RDMA进行notify record时需要将该内存copy到远端 notify 寄存器
using HcclSignalRecordBuff = struct HcclSignalRecordBuffDef {
    u64 address{0}; //  signal 地址
    u64 length{0};  //  signal 长度
};

constexpr u32 HCCL_TRANSPORT_RELATIONSHIP_SAME_CHIP = 0x1U << 0;   // transport 的两端rank位于同一个NPU芯片内
constexpr u32 HCCL_TRANSPORT_RELATIONSHIP_SAME_SERVER = 0x1U << 1; // transport 的两端rank位于同一个服务器内
constexpr u32 HCCL_TRANSPORT_RELATIONSHIP_SAME_SUPERPOD = 0x1U << 2; // transport 的两端rank位于同一个超节点内

// transport 使用的基础信息，包括链路类型、和远端的位置关系等
using TransportAttr = struct TransportAttrDef {
    hccl::LinkType linkType{hccl::LinkType::LINK_RESERVED}; // 链路类型，HCCS,
    u32 relationship{0}; // 和remote的位置关系{同芯片，同节点，跨节点}
    HcclSignalRecordBuff signalRecordBuff;
};

constexpr u64 MAX_EXCHANGE_DATA_LEN = 2ULL * 1024 * 1024; // 自定义交换数据限制2MB

// 传参数的时候都填充，link自己使用的时候区分
// RDMA:  machineType+serverId+local_rank_id+remote_rank_id+collectiveId
// TCP:   machineType+serverId+local_rank_id+remote_rank_id+collectiveId
// PCIE:  local_rank_id+remote_rank_id+collectiveId+localDeviceId+remoteDeviceId
using MachinePara = struct TagMachinePara {
public:
    MachineType machineType{MachineType::MACHINE_RESERVED_TYPE}; // client或者server
    LinkMode linkMode{LinkMode::LINK_RESERVED_MODE};
    std::string collectiveId{""}; // 本节点所在的通信域ID
    std::string tag{""};
    std::string serverId; // 本端server id

    HcclIpAddress localIpAddr;  // 本端rank ip
    HcclIpAddress remoteIpAddr; // 对端rank ip

    u32 localSocketPort;
    u32 remoteSocketPort;

    s32 localDeviceId{-1};  // 本端device physical id
    s32 remoteDeviceId{-1}; // 对端device physical id
    s32 deviceLogicId{0};

    u32 localUserrank{INVALID_VALUE_RANKID};  // 本端user rank
    u32 remoteUserrank{INVALID_VALUE_RANKID}; // 对端user rank

    u32 localWorldRank{INVALID_VALUE_RANKID};  // 本端world group rank
    u32 remoteWorldRank{INVALID_VALUE_RANKID}; // 对端world group rank

    NICDeployment nicDeploy{NICDeployment::NIC_DEPLOYMENT_DEVICE};
    DevType deviceType{DevType::DEV_TYPE_COUNT};

    std::vector<std::shared_ptr<HcclSocket>> sockets;
    std::vector<u8> exchangeInfo; // 自定义交换数据,限制MAX_EXCHANGE_DATA_LEN = 2MB

    DeviceMem inputMem{DeviceMem()};
    DeviceMem outputMem{DeviceMem()};
    std::vector<DeviceMem> mem{};

    // 自定义算子交换内存
    std::vector<DeviceMem> userDeviceMem{};
    std::vector<HostMem> userHostMem{};
    bool isIndOp{false};

    // link特性位图: bit0:0x1支持WRITE操作（源端发起数据传输，优选项）
    // bit1:0x2支持READ操作（目的端发起数据传输）。如果同时支持link优先选用目的端发起数据传输
    u64 linkAttribute{0x1}; // 初始设置为WRITE操作，从源端发起数据传输;

    bool supportDataReceivedAck{false};
    bool isAicpuModeEn{false};
    std::vector<std::uint16_t> srcPorts; // 多qp配置的源端口号
    u32 notifyNum{0};
    QPMode qpMode{QPMode::INVALID}; // 是否为普通QP模式
    u32 tc{HCCL_COMM_TRAFFIC_CLASS_CONFIG_NOT_SET};
    u32 sl{HCCL_COMM_SERVICE_LEVEL_CONFIG_NOT_SET};
    u32 retryCnt{INVALID_UINT};
    u32 retryInterval{INVALID_UINT};
    LinkTypeInServer specifyLink{LinkTypeInServer::RESERVED_LINK_TYPE}; // 指定链路类型
    bool enableAtomicWrite{false};                                      // 使能atomicWrite
    QueueDepthAttr queueDepthAttr{};                                    // QP深度配置
    bool userMemEnable{true};
    bool drainEnable{false};
    // DispatcherCtxPtr；设备侧 TS Roce 等场景传入，WriteCommon 内写入线程局部 dispatcher
    void* dctxPtr{nullptr};
    bool isNewOneSide{false};
    u32 localBufSize{0};
    u32 remoteBufSize{0};
    HcclMemEx* localBufMem{nullptr};
    HcclMemEx* remoteBufMem{nullptr};
    TagMachinePara() {}

    TagMachinePara(const struct TagMachinePara& that)
    {
        machineType = (that.machineType);
        linkMode = (that.linkMode);
        serverId = (that.serverId);
        localIpAddr = (that.localIpAddr);
        remoteIpAddr = (that.remoteIpAddr);
        localDeviceId = (that.localDeviceId);
        remoteDeviceId = (that.remoteDeviceId);
        localUserrank = (that.localUserrank);
        remoteUserrank = (that.remoteUserrank);
        localWorldRank = (that.localWorldRank);
        remoteWorldRank = (that.remoteWorldRank);
        collectiveId = (that.collectiveId);
        deviceType = (that.deviceType);
        tag = (that.tag);
        inputMem = (that.inputMem);
        outputMem = (that.outputMem);
        mem = (that.mem);
        userDeviceMem = (that.userDeviceMem);
        userHostMem = (that.userHostMem);
        isIndOp = (that.isIndOp);
        linkAttribute = (that.linkAttribute);
        sockets = (that.sockets);
        exchangeInfo = (that.exchangeInfo);
        supportDataReceivedAck = (that.supportDataReceivedAck);
        nicDeploy = (that.nicDeploy);
        localSocketPort = that.localSocketPort;
        remoteSocketPort = that.remoteSocketPort;
        isAicpuModeEn = that.isAicpuModeEn;
        deviceLogicId = that.deviceLogicId;
        srcPorts = that.srcPorts;
        notifyNum = that.notifyNum;
        qpMode = that.qpMode;
        tc = that.tc;
        sl = that.sl;
        retryCnt = that.retryCnt;
        retryInterval = that.retryInterval;
        specifyLink = that.specifyLink;
        enableAtomicWrite = that.enableAtomicWrite;
        queueDepthAttr = that.queueDepthAttr;
        userMemEnable = that.userMemEnable;
        drainEnable = that.drainEnable;
        dctxPtr = that.dctxPtr;
        isNewOneSide = (that.isNewOneSide);
        localBufSize = (that.localBufSize);
        remoteBufSize = (that.remoteBufSize);
        localBufMem = (that.localBufMem);
        remoteBufMem = (that.remoteBufMem);
    }

    struct TagMachinePara& operator=(struct TagMachinePara& that)
    {
        if (&that != this) {
            machineType = (that.machineType);
            linkMode = (that.linkMode);
            serverId = (that.serverId);
            localIpAddr = (that.localIpAddr);
            remoteIpAddr = (that.remoteIpAddr);
            localDeviceId = (that.localDeviceId);
            remoteDeviceId = (that.remoteDeviceId);
            localUserrank = (that.localUserrank);
            remoteUserrank = (that.remoteUserrank);
            localWorldRank = (that.localWorldRank);
            remoteWorldRank = (that.remoteWorldRank);
            collectiveId = (that.collectiveId);
            deviceType = (that.deviceType);
            tag = (that.tag);
            inputMem = (that.inputMem);
            outputMem = (that.outputMem);
            mem = (that.mem);
            userDeviceMem = (that.userDeviceMem);
            userHostMem = (that.userHostMem);
            isIndOp = (that.isIndOp);
            linkAttribute = (that.linkAttribute);
            sockets = (that.sockets);
            exchangeInfo = (that.exchangeInfo);
            supportDataReceivedAck = (that.supportDataReceivedAck);
            localSocketPort = that.localSocketPort;
            remoteSocketPort = that.remoteSocketPort;
            isAicpuModeEn = that.isAicpuModeEn;
            deviceLogicId = that.deviceLogicId;
            srcPorts = that.srcPorts;
            notifyNum = that.notifyNum;
            qpMode = that.qpMode;
            tc = that.tc;
            sl = that.sl;
            retryCnt = that.retryCnt;
            retryInterval = that.retryInterval;
            specifyLink = that.specifyLink;
            enableAtomicWrite = that.enableAtomicWrite;
            queueDepthAttr = that.queueDepthAttr;
            userMemEnable = that.userMemEnable;
            drainEnable = that.drainEnable;
            dctxPtr = that.dctxPtr;
            isNewOneSide = (that.isNewOneSide);
            localBufSize = (that.localBufSize);
            remoteBufSize = (that.remoteBufSize);
            localBufMem = (that.localBufMem);
            remoteBufMem = (that.remoteBufMem);
        }

        return *this;
    }
};

struct TransportPara {
    std::chrono::milliseconds timeout;
    NICDeployment nicDeploy;
    u32 localDieID;
    u32 dstDieID;
    HcclIpAddress* selfIp;
    HcclIpAddress* peerIp;
    u32 peerPort;
    u32 selfPort;
    u32 index;
    bool isRootRank;
    u32 devLogicId;
    u32 proxyDevLogicId;
    s32 qpMode = 0;
    bool isHdcMode = false;
    bool remoteIsHdc = false;
    bool isESPs = false;
    bool virtualFlag = false;
};

struct TransportDeviceNormalData {
    MemDetails remoteInputMem{};
    MemDetails remoteOutputMem{};
    MemDetails localInputMem{};
    MemDetails localOutputMem{};
    struct HcclQpInfoV2 qpInfo {};
    QPMode qpMode{QPMode::INVALID};
    void Print()
    {
        HCCL_DEBUG(
            "remoteInputMem: addr[%llu], size[%llu], key[%u]", remoteInputMem.addr, remoteInputMem.size,
            remoteInputMem.key);
        HCCL_DEBUG(
            "remoteOutputMem: addr[%llu], size[%llu], key[%u]", remoteOutputMem.addr, remoteOutputMem.size,
            remoteOutputMem.key);
        HCCL_DEBUG(
            "localInputMem: addr[%llu], size[%llu], key[%u]", localInputMem.addr, localInputMem.size,
            localInputMem.key);
        HCCL_DEBUG(
            "localOutputMem: addr[%llu], size[%llu], key[%u]", localOutputMem.addr, localOutputMem.size,
            localOutputMem.key);
        HCCL_DEBUG("qpInfo: qpPtr[%llu], sqIndex[%u], dbIndex[%u]", qpInfo.qpPtr, qpInfo.sqIndex, qpInfo.dbIndex);
        HCCL_DEBUG("qpMode[%d]", static_cast<int32_t>(qpMode));
        return;
    }
};

struct TransportDeviceP2pData {
    void* inputBufferPtr;
    void* outputBufferPtr;
    std::shared_ptr<LocalNotify> ipcPreWaitNotify;
    std::shared_ptr<LocalNotify> ipcPostWaitNotify;
    std::vector<std::shared_ptr<LocalNotify>> userLocalNotify;
    std::shared_ptr<RemoteNotify> ipcPreRecordNotify;
    std::shared_ptr<RemoteNotify> ipcPostRecordNotify;
    std::vector<std::shared_ptr<RemoteNotify>> userRemoteNotify;
    TransportAttr transportAttr;

    TransportDeviceP2pData() {}
    TransportDeviceP2pData(
        void* inputBufferPtr, void* outputBufferPtr, std::shared_ptr<LocalNotify> ipcPreWaitNotify,
        std::shared_ptr<LocalNotify> ipcPostWaitNotify, std::vector<std::shared_ptr<LocalNotify>> userLocalNotify,
        std::shared_ptr<RemoteNotify> ipcPreRecordNotify, std::shared_ptr<RemoteNotify> ipcPostRecordNotify,
        std::vector<std::shared_ptr<RemoteNotify>> userRemoteNotify, TransportAttr& transportAttr)
        : inputBufferPtr(inputBufferPtr),
          outputBufferPtr(outputBufferPtr),
          ipcPreWaitNotify(ipcPreWaitNotify),
          ipcPostWaitNotify(ipcPostWaitNotify),
          userLocalNotify(userLocalNotify),
          ipcPreRecordNotify(ipcPreRecordNotify),
          ipcPostRecordNotify(ipcPostRecordNotify),
          userRemoteNotify(userRemoteNotify),
          transportAttr(transportAttr)
    {}
};

struct TransportDeviceIbverbsData {
    void* inputBufferPtr;
    void* outputBufferPtr;
    MemDetails localInputMem;
    MemDetails localOutputMem;
    std::shared_ptr<LocalNotify> ackNotify;
    std::shared_ptr<LocalNotify> dataAckNotify;
    std::shared_ptr<LocalNotify> dataNotify;
    std::vector<std::vector<std::shared_ptr<LocalNotify>>> userLocalNotify;
    uint64_t localNotifyValueAddr;
    AddrKey remoteAckNotifyDetails;
    AddrKey remoteDataNotifyDetails;
    AddrKey remoteDataAckNotifyDetails;
    std::vector<std::vector<AddrKey>> userRemoteNotifyDetails;
    uint32_t notifyValueKey;
    std::vector<struct HcclQpInfoV2> qpInfo;
    uint32_t remoteInputKey;
    uint32_t remoteOutputKey;
    uint32_t notifySize;
    u32 multiQpThreshold;
    u32 qpsPerConnection;
    bool useAtomicWrite = false;
    std::vector<RoceMemDetails> localRoceMemDetailsList;
    std::vector<RoceMemDetails> remoteRoceMemDetailsList;
    bool useMemDetailsMgr{false};
    uint64_t remoteNotifyValueAddr{0}; // 对端 NOTIFY_SRC_MEM 地址 (Fence Read 读取源)
    uint32_t remoteNotifyValueKey{0};  // 对端 notify rkey
    uint64_t localDataNotifyAddr{0};   // 本端dataNotify addr
    uint32_t localDataNotifyKey{0};    // 本端dataNotifyKey
    TransportDeviceIbverbsData() {}
    TransportDeviceIbverbsData(
        void* inputBufferPtr, void* outputBufferPtr, MemDetails localInputMem, MemDetails localOutputMem,
        std::shared_ptr<LocalNotify> ackNotify, std::shared_ptr<LocalNotify> dataAckNotify,
        std::shared_ptr<LocalNotify> dataNotify, std::vector<std::vector<std::shared_ptr<LocalNotify>>> userLocalNotify,
        uint64_t localNotifyValueAddr, AddrKey remoteAckNotifyDetails, AddrKey remoteDataNotifyDetails,
        AddrKey remoteDataAckNotifyDetails, std::vector<std::vector<AddrKey>> userRemoteNotifyDetails,
        uint32_t notifyValueKey, std::vector<struct HcclQpInfoV2> qpInfo, uint32_t remoteInputKey,
        uint32_t remoteOutputKey, uint32_t notifySize, u32 multiQpThreshold, u32 qpsPerConnection, bool useAtomicWrite,
        uint64_t remoteNotifyValueAddr, uint32_t remoteNotifyValueKey, uint64_t localDataNotifyAddr,
        uint32_t localDataNotifyKey)
        : inputBufferPtr(inputBufferPtr),
          outputBufferPtr(outputBufferPtr),
          localInputMem(localInputMem),
          localOutputMem(localOutputMem),
          ackNotify(ackNotify),
          dataAckNotify(dataAckNotify),
          dataNotify(dataNotify),
          userLocalNotify(userLocalNotify),
          localNotifyValueAddr(localNotifyValueAddr),
          remoteAckNotifyDetails(remoteAckNotifyDetails),
          remoteDataNotifyDetails(remoteDataNotifyDetails),
          remoteDataAckNotifyDetails(remoteDataAckNotifyDetails),
          userRemoteNotifyDetails(userRemoteNotifyDetails),
          notifyValueKey(notifyValueKey),
          qpInfo(qpInfo),
          remoteInputKey(remoteInputKey),
          remoteOutputKey(remoteOutputKey),
          notifySize(notifySize),
          multiQpThreshold(multiQpThreshold),
          qpsPerConnection(qpsPerConnection),
          useAtomicWrite(useAtomicWrite),
          remoteNotifyValueAddr(remoteNotifyValueAddr),
          remoteNotifyValueKey(remoteNotifyValueKey),
          localDataNotifyAddr(localDataNotifyAddr),
          localDataNotifyKey(localDataNotifyKey)
    {}

    TransportDeviceIbverbsData(const TransportDeviceIbverbsData& that)
        : inputBufferPtr(that.inputBufferPtr),
          outputBufferPtr(that.outputBufferPtr),
          localInputMem(that.localInputMem),
          localOutputMem(that.localOutputMem),
          ackNotify(that.ackNotify),
          dataAckNotify(that.dataAckNotify),
          dataNotify(that.dataNotify),
          userLocalNotify(that.userLocalNotify),
          localNotifyValueAddr(that.localNotifyValueAddr),
          remoteAckNotifyDetails(that.remoteAckNotifyDetails),
          remoteDataNotifyDetails(that.remoteDataNotifyDetails),
          remoteDataAckNotifyDetails(that.remoteDataAckNotifyDetails),
          userRemoteNotifyDetails(that.userRemoteNotifyDetails),
          notifyValueKey(that.notifyValueKey),
          qpInfo(that.qpInfo),
          remoteInputKey(that.remoteInputKey),
          remoteOutputKey(that.remoteOutputKey),
          notifySize(that.notifySize),
          multiQpThreshold(that.multiQpThreshold),
          qpsPerConnection(that.qpsPerConnection),
          useAtomicWrite(that.useAtomicWrite),
          localRoceMemDetailsList(that.localRoceMemDetailsList),
          remoteRoceMemDetailsList(that.remoteRoceMemDetailsList),
          useMemDetailsMgr(that.useMemDetailsMgr),
          remoteNotifyValueAddr(that.remoteNotifyValueAddr),
          remoteNotifyValueKey(that.remoteNotifyValueKey),
          localDataNotifyAddr(that.localDataNotifyAddr),
          localDataNotifyKey(that.localDataNotifyKey)
    {}
};
using CqeInfo = struct tagCqeInfo {
    struct timeval time;
    uint32_t status = 0;
    HcclIpAddress remoteIp;
    char reserved[32];
    tagCqeInfo() {}
    tagCqeInfo(const struct timeval& time, const uint32_t status, const HcclIpAddress& remoteIp)
        : time(time),
          status(status),
          remoteIp(remoteIp)
    {}
};

class Transport {
public:
    struct Buffer {
        const void* addr{nullptr};
        u64 size{0};

        Buffer() : addr(nullptr), size(0) {}
        Buffer(const void* addr, u64 size) : addr(addr), size(size) {}
    };

    Transport() {};
    explicit Transport(TransportBase* pimpl) : pimpl_(pimpl) {};
    Transport(
        TransportType type, TransportPara& para, const HcclDispatcher dispatcher,
        const std::unique_ptr<NotifyPool>& notifyPool, MachinePara& machinePara,
        const TransportDeviceP2pData& transDevP2pData = TransportDeviceP2pData(),
        const TransportDeviceIbverbsData& transDevIbverbsData = TransportDeviceIbverbsData());

    ~Transport();

    HcclResult Stop();
    HcclResult Resume();
    HcclResult Init();
    HcclResult DeInit();

    HcclResult TxDataSignal(Stream& stream);
    HcclResult RxDataSignal(Stream& stream);

    HcclResult TxAsync(UserMemType dstMemType, u64 dstOffset, const void* src, u64 len, Stream& stream);
    HcclResult TxAsync(std::vector<TxMemoryInfo>& txMems, Stream& stream);

    HcclResult TxWithReduce(
        UserMemType dstMemType, u64 dstOffset, const void* src, u64 len, const HcclDataType datatype,
        HcclReduceOp redOp, Stream& stream);
    HcclResult TxWithReduce(
        const std::vector<TxMemoryInfo>& txWithReduceMems, const HcclDataType datatype, HcclReduceOp redOp,
        Stream& stream);
    HcclResult RxWithReduce(
        UserMemType recvSrcMemType, u64 recvSrcOffset, void* recvDst, u64 recvLen, void* reduceSrc, void* reduceDst,
        u64 reduceDataCount, HcclDataType reduceDatatype, HcclReduceOp reduceOp, Stream& stream, const u64 reduceAttr);
    HcclResult RxWithReduce(
        const std::vector<RxWithReduceMemoryInfo>& rxWithReduceMems, HcclDataType reduceDatatype, HcclReduceOp reduceOp,
        Stream& stream, const u64 reduceAttr);
    bool IsSupportTransportWithReduce();

    HcclResult RxAsync(UserMemType srcMemType, u64 srcOffset, void* dst, u64 len, Stream& stream);
    HcclResult RxAsync(std::vector<RxMemoryInfo>& rxMems, Stream& stream);
    HcclResult DataReceivedAck(Stream& stream);

    HcclResult TxAck(Stream& stream);
    HcclResult RxAck(Stream& stream);

    HcclResult TxPrepare(Stream& stream);
    HcclResult RxPrepare(Stream& stream);

    HcclResult TxDone(Stream& stream);
    HcclResult RxDone(Stream& stream);

    HcclResult TxData(UserMemType dstMemType, u64 dstOffset, const void* src, u64 len, Stream& stream);
    HcclResult RxData(UserMemType srcMemType, u64 srcOffset, void* dst, u64 len, Stream& stream);

    // 保证send语义完成
    HcclResult TxWaitDone(Stream& stream);
    // 保证recv语义完成
    HcclResult RxWaitDone(Stream& stream);
    // TxWaitDone、RxWaitDone共同出现保证sendrecv语义完成

    HcclResult Post(u32 notifyIdx, Stream& stream);
    HcclResult Wait(u32 notifyIdx, Stream& stream, const u32 timeOut = NOTIFY_INVALID_WAIT_TIME);

    u32 GetNotifyNum();
    HcclResult GetIndOpRemoteMemDetails(MemDetails** remoteMem, uint32_t* memNum, HcclMemType memType);
    HcclResult GetIndOpRemoteMem(HcclMem** remoteMem, uint32_t* memNum);
    HcclResult GetLocalNotify(std::vector<HcclSignalInfo>& localNotify);
    HcclResult GetRemoteNotify(std::vector<HcclSignalInfo>& localNotify);
    HcclResult GetRemoteMem(UserMemType memType, void** remotePtr);
    HcclResult GetRemoteMem(std::vector<void*>* remotePtrVec);
    HcclResult GetRemoteMemKey(UserMemType memType, uint32_t* remoteMemKey);
    HcclResult GetLocalRdmaNotify(std::vector<HcclSignalInfo>& rdmaNotify);
    HcclResult GetDrainLocalDataNotify(void*& localAddr, uint32_t& lkey, HcclSignalInfo& dataNotify);
    HcclResult GetRemoteRdmaNotifyAddrKey(std::vector<AddrKey>& rdmaNotifyAddr);
    HcclResult GetLocalNotifyValueAddrKey(std::vector<AddrKey>& notifyValue);
    HcclResult GetLocalMemDetails(UserMemType memType, MemDetails& memDetails);
    HcclResult GetAiQpInfo(std::vector<HcclQpInfoV2>& aiQpInfo);
    HcclResult GetAiRMAQueueInfo(std::vector<HcclAiRMAQueueInfo>& aiRMAQueueInfo);
    HcclResult GetTransportId(u32& id);
    HcclResult GetChipId(s64& chipId);
    virtual HcclResult GetRemoteMemSize(UserMemType memType, u64& size);
    HcclResult GetTxAckDevNotifyInfo(HcclSignalInfo& notifyInfo);
    HcclResult GetRxAckDevNotifyInfo(HcclSignalInfo& notifyInfo);
    HcclResult GetTxDataSigleDevNotifyInfo(HcclSignalInfo& notifyInfo);
    HcclResult GetRxDataSigleDevNotifyInfo(HcclSignalInfo& notifyInfo);

    hccl::LinkType GetLinkType() const;
    bool IsSpInlineReduce() const;
    bool GetSupportDataReceivedAck() const;
    void SetSupportDataReceivedAck(bool supportDataReceivedAck);
    u32 GetRemoteRank();

    HcclResult ConnectAsync(u32& status);
    HcclResult ConnectQuerry(u32& status);
    void Break();

    void EnableUseOneDoorbell();

    bool GetUseOneDoorbellValue();

    HcclResult GetTransportAttr(TransportAttr& attr);

    HcclResult TxEnv(const void* ptr, const u64 len, Stream& stream);
    HcclResult RxEnv(Stream& stream);
    bool IsTransportRoce();

    HcclResult WriteAsync(struct Buffer& remoteBuf, struct Buffer& localBuf, Stream& stream);
    HcclResult WriteSync(struct Buffer& remoteBuf, struct Buffer& localBuf, Stream& stream);

    HcclResult WriteReduceAsync(
        struct Buffer& remoteBuf, struct Buffer& localBuf, const HcclDataType datatype, HcclReduceOp redOp,
        Stream& stream);

    HcclResult ReadAsync(struct Buffer& localBuf, struct Buffer& remoteBuf, Stream& stream);
    HcclResult ReadSync(struct Buffer& localBuf, struct Buffer& remoteBuf, Stream& stream);
    HcclResult ReadReduceSync(
        struct Buffer& localBuf, struct Buffer& remoteBuf, const HcclDataType datatype, HcclReduceOp redOp,
        Stream& stream);

    HcclResult BatchTransferAsync(const HcommBatchTransferDesc* transferDescs, uint32_t descNum, Stream& stream);

    HcclResult PostReady(Stream& stream);
    HcclResult WaitReady(Stream& stream);

    HcclResult PostFin(Stream& stream);
    HcclResult WaitFin(Stream& stream);

    HcclResult PostFinAck(Stream& stream);
    HcclResult WaitFinAck(Stream& stream);

    HcclResult SetStopFlag(bool value);
    HcclResult Fence();
    HcclResult UpdateRemoteAddr(void* remoteIn, void* remoteOut);
    HcclResult Drain(Stream& stream);
    HcclResult InitDrainNotifyInfo();
    HcclResult GetDrainRemSrcMem(void*& remoteAddr, uint32_t& remoteKey, uint32_t& size);
    static HcclResult
    GetTransportErrorCqe(const HcclNetDevCtx netDevCtx, std::vector<std::pair<Transport*, CqeInfo>>& infos, u32& num);
    inline TransportType GetTransportType() const { return type_; }

    std::vector<u8> GetExchangeInfo();
    static HcclResult HcclBatchRead(
        const TransportDeviceNormalData& ibvData, struct MemDetails* localMems, struct MemDetails* remoteMems,
        u32 memNum, u64& dbInfo);
    static HcclResult HcclBatchWrite(
        const TransportDeviceNormalData& ibvData, struct MemDetails* localMems, struct MemDetails* remoteMems,
        u32 memNum, u64& dbInfo);
    static HcclResult SetDeviceUnavailable(u32 deviceId);

    bool GetIsUseAtomicWrite();

    HcclResult GetSpecificNotify(HcclSignalInfo& notifyInfo, bool& isValid, const std::string& notifyName);

private:
    void CreateTransportRoce(
        TransportType type, TransportPara& para, const HcclDispatcher dispatcherPtr,
        const std::unique_ptr<NotifyPool>& notifyPool, MachinePara& machinePara);
    TransportBase* pimpl_ = nullptr;
    const TransportType type_ = TransportType::TRANS_TYPE_RESERVED;

    static std::mutex mapMutex_;
    static std::unordered_map<TransportBase*, Transport*> transportMap_;
};

using LINK = std::shared_ptr<Transport>;
} // namespace hccl

#endif /* TRANSPORT_BASE_H */
