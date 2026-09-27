/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef PING_MESH_PUB_H
#define PING_MESH_PUB_H

#include <map>
#include <thread>
#include <atomic>
#include "hccl_ip_address.h"
#include "hccl_socket.h"
#include "hdc_pub.h"
#include "network/hccp_common.h"
#include "network/hccp_ping.h"
#include "hccn_rping.h"
#include "orion_adapter_hccp.h"
#include "dispatcher_task_types.h"
#include "universal_concurrent_map.h"

namespace hccl {
constexpr u32 ONE_MILLISEC = 1000;
constexpr u32 RPING_INTERFACE_OPCODE = 71;
constexpr u32 RPING_INTERFACE_VERSION = 1;
constexpr u32 RPING_PAYLOAD_LEN_MAX = 1500;
constexpr u64 TSD_EXT_PARA_NUM = 2;
constexpr u32 RPING_SERVICE_LEVEL_DEFAULT = 4;
constexpr u32 RPING_TRAFFIC_CLASS_DEFAULT = 132;

// 判断类型相关函数
bool IsSupportHCCLV2(const char* socNamePtr);
HcclResult GetAddrType(u32* addrtype);
// HCCN接口需要的结构体
struct RpingInput {
    HcclIpAddress sip;
    HcclIpAddress dip;
    int srcPort;  // UDP源端口号，用户配置，参与hash选路
    int reserved; // 保留字段，用于对齐，暂不使用
    int sl;       // 指定队列
    int tc;       // 主要是修改DSCP
    int port;     // 监听端口
    u32 len;
    u32 addrType; /* address type, 0: ip, 1: eid */ // todo: 是否要添加需要确定
    char payload[RPING_PAYLOAD_LEN_MAX];
};

MAKE_ENUM(HrtNetworkMode, PEER, HDC)

struct HRaInfo {
    HrtNetworkMode mode;
    uint32_t phyId;
    HRaInfo(HrtNetworkMode mode, uint32_t phyId) : mode(mode), phyId(phyId) {}
};

struct RpingOutput {
    u32 txPkt; // rping发包总数
    u32 rxPkt; // rping收包总数
    u32 minRTT;
    u32 maxRTT;
    u32 avgRTT;
    u32 state;
    u32 reserved[2];
};

// 需要重填的payload头只有前136字节，后面是固定的
struct RpingPayloadHead {
    union {
        char srcIp[64];  /* local(client) ip */
        char srcEid[16]; /* local(client) eid */
    };
    union {
        char dstIp[64];  /* remote(target) ip  */
        char dstEid[16]; /* remote(target) eid */
    };
    u32 payloadLen;
    u32 resvd[3];
    u64 timestamp[8];
    u32 rpingBatchId;
    u32 addrType; /* address type, 0: ip, 1: eid */
    u8 reserved[40];
};

union RpingIpHead {
    struct {
        u32 versionTclassFlow; // 4bit version, 8bit tclass, 20bit flow label
        u16 payLen;            // ub报文的payload长度
        u8 nextHdr;            // 下一个头部的类型
        u8 hopLimit;           // 最大跳数
        u8 srcIp[16];          // sip的gid，128bit
        u8 dstIp[16];          // dip的gid，128bit
    } ipv6;

    struct {
        u8 rsvd[20];           // 前20字节为空
        u32 verIhlTosTlen;     // 4bit version, 4bit IHL, 8bit type of service, 16bit total length
        u32 IdFlagsFragOffset; // 16bit identification, 3bit flags, 13bit fragment offset
        u8 tol;                // time to live
        u8 protocol;           // 协议类型
        u16 headChecksum;
        u32 srcIp;
        u32 dstIp;
    } ipv4;
};

struct RpingEidHead {
    u32 version;       // 32bit version
    u32 type;          // 32bit type
    u32 ser_version;   // 32bit ser_version
    u32 padding1;      // 32bit padding
    u8 info_size1;     // 8bit的info_size
    u8 srcEid[16];     // sip的Eid，128bit
    u32 uasid1;        // 32bit的uasid
    u32 jetty_id1;     // 32bit的jetty_id值
    u32 padding2;      // 32bit的padding
    u8 resvd[7];       // 56bit的reserved
    u32 s_token_value; // 32bit的s_token_value
    u32 dst_version;   // 32bit的dst_version
    u32 padding3;      // 32bit的padding
    u8 info_size2;     // 4bit的info_size
    u8 dstEid[16];     // dip的Eid，128bit
    u32 uasid2;        // 32bit的uasid2
    u32 jetty_id2;     // 32bit的jetty_id2
    u8 reserv[7];
    u32 client_jetty_token_value; // 32bit的client_jetty_token_value
    u64 times[8];
    u32 taskId;
    u8 reserved[44];
};

enum class RpingState { UNINIT, INITED, READY, RUN, STOP, RESERVED };

enum class RpingLinkState { CONNECTED, CONNECTING, DISCONNECTED, TIMEOUT, ERROR };

enum WhiteListStatus { WHITE_LIST_CLOSE = 0, WHITE_LIST_OPEN = 1 };

class PingMesh {
private:
    PingInitInfo initInfo_{};                      // 初始化信息
    void* pingHandle_ = nullptr;                   // 记录hccp侧的pingmesh句柄
    std::shared_ptr<HcclSocket> socket_ = nullptr; // 记录server端的socket信息，用于建立rdma链路
    HcclIpAddress ipAddr_;                         // 记录device的ip信息
    u8* payload_ = nullptr;                        // client侧记录的payload信息
    RpingState rpingState_ = RpingState::UNINIT;   // 记录client状态
    int rpingTargetNum_ = 0;                       // 记录client目标数量
    u32 pktNum_ = 0;                               // 记录最近一次BatchPingStart的发包数量
    std::mutex socketMapsMtx_;
    std::map<std::string, std::shared_ptr<HcclSocket>> socketMaps_; // 记录client端的socket信息
    UniversalConcurrentMap<std::string, PingQpInfo> rdmaInfoMaps_;  // 记录target的rdma或者ub信息
    std::unique_ptr<std::thread> connThread_;                       // server端等待socket建链的背景线程
    HcclNetDevCtx netCtx_ = nullptr;                                // 记录网络上下文信息
    std::shared_ptr<HDCommunicate> hdcD2H_ = nullptr;               // 从device侧获取数据的接口
    s32 deviceLogicId_ = 0;
    u32 devicePhyId_ = 0;
    bool isDeinited_ = false;
    bool isSocketClosed_ = false;
    std::atomic<bool> connThreadStop_{false}; // 侦听的子线程的结束条件
    bool isUsePayload_ = false;
    UniversalConcurrentMap<std::string, u32> payloadLenMap_; // 记录自定义payload的长度
    HccnRpingMode mode = HCCN_RPING_MODE_ROCE;               // ROCEorUB,

    HcclResult RpingSendInitInfo(
        u32 deviceId, u32 port, HcclIpAddress ipAddr, PingInitInfo initInfo, std::shared_ptr<HcclSocket> socket);
    HcclResult
    RpingRecvTargetInfo(void* clientNetCtx, u32 port, HcclIpAddress ipAddr, PingInitInfo& recvInfo, u32 timeout);
    HcclResult StartSocketThread(u32 deviceId, HcclIpAddress ipAddr, u32 port);
    HcclResult HccnCloseSubProc(u32 deviceId);
    HcclResult HccnRaInit(u32 deviceId);
    HcclResult HccnTargetAttrInter(
        u32 targetNumInter, RpingInput* inputInter, HccnRpingAddTargetConfig* configInter, PingTargetInfo* targetInter);
    HcclResult HccnTarRemoveAttrInter(
        u32 targetNumInter, RpingInput* inputInter, PingTargetCommInfo* targetInter,
        std::shared_ptr<HcclSocket>& socketInter);
    HcclResult RpingResultInfoInit(
        PingTargetResult* resultInfo, UniversalConcurrentMap<std::string, PingQpInfo>& rdmaInfoMaps, RpingInput* input,
        u32 targetNum);
    HcclResult HccnSupportedAndGetphyid(u32 deviceId, LinkType netMode);
    HcclResult HccnRpingOpenTsd(u32 deviceId, u32 mode, u32 port, u32 nodeNum, u32 bufferSize, u32 sl, u32 tc);

public:
    PingMesh();
    ~PingMesh();
    HcclResult HccnRpingInit(
        u32 deviceId, u32 mode, HcclIpAddress ipAddr, u32 port, u32 nodeNum, u32 bufferSize,
        u32 sl = RPING_SERVICE_LEVEL_DEFAULT, u32 tc = RPING_TRAFFIC_CLASS_DEFAULT);
    HcclResult HccnRpingDeinit(u32 deviceId);
    HcclResult HccnRpingAddTarget(u32 deviceId, u32 targetNum, RpingInput* input, HccnRpingAddTargetConfig* config);
    HcclResult HccnRpingRemoveTarget(u32 deviceId, u32 targetNum, RpingInput* input);
    HcclResult HccnRpingGetTarget(u32 deviceId, u32 targetNum, RpingInput* input, int* targetStat);
    HcclResult HccnRpingBatchPingStart(u32 deviceId, u32 pktNum, u32 interval, u32 timeout);
    HcclResult HccnRpingBatchPingStop(u32 deviceId);
    HcclResult HccnRpingGetResult(u32 deviceId, u32 targetNum, RpingInput* input, RpingOutput* output);
    HcclResult HccnRpingRefillPayloadHead(u8* originalHead, u32 payloadNum);
    HcclResult HccnRpingRefillUbPayloadHead(u8* originalHead, u32 payloadNum);
    HcclResult HccnRpingGetPayload(u32 deviceId, void** payload, u32* payloadLen, HccnRpingMode mode);

    inline s32 GetDeviceLogicId() { return deviceLogicId_; }
    inline HccnRpingMode GetMode() { return mode; }
    inline void init(HccnRpingInitAttr* attr) { this->mode = attr->mode; }
};
} // namespace hccl
#endif
