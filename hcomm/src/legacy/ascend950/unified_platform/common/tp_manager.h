/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_TP_MANAGER_H
#define HCCLV2_TP_MANAGER_H

#include <cstdint>
#include <functional>
#include <mutex>
#include <vector>
#include <unordered_map>

#include "hccl_types.h"
#include "ip_address.h"
#include "orion_adapter_hccp.h"

namespace Hccl {

constexpr uint8_t TA_GEAR_INDEX_0 = 0;
constexpr uint8_t TA_GEAR_INDEX_1 = 1;
constexpr uint8_t TA_GEAR_INDEX_2 = 2;
constexpr uint8_t TA_GEAR_INDEX_3 = 3;

constexpr uint8_t TA_HW_GEAR0_BASE = 0;
constexpr uint8_t TA_HW_GEAR1_BASE = 8;
constexpr uint8_t TA_HW_GEAR2_BASE = 16;
constexpr uint8_t TA_HW_GEAR3_BASE = 24;

static constexpr uint32_t TA_TIMEOUT_MS_GEAR0 = 512;
static constexpr uint32_t TA_TIMEOUT_MS_GEAR1 = 4000;
static constexpr uint32_t TA_TIMEOUT_MS_GEAR2 = 8000;
static constexpr uint32_t TA_TIMEOUT_MS_GEAR3 = 32000;

constexpr uint8_t AT_GEAR_MIN = 0;
constexpr uint8_t AT_GEAR_MAX = 3;
constexpr uint8_t AT_GEAR_DEFAULT = 2;
constexpr uint32_t AT_TIMEOUT_MAP[4] = {16, 128, 1000, 4000};

/// 与 GetTpInfo / ReleaseTpInfo 中 info、req 两级 map 的 qos 键一致（param.qos 低 8 位）
using QosKey = uint32_t;

/*
 * TP信息，当前申请TpHandle，不感知具体TP信息，当前仅支持TP与CTP
 * tpHandle: 对应管控面的TPID与相关资源，URMA通过引用计数管理申请和销毁TP
 */
using TpHandle = uint64_t;
struct TpInfo {
    TpHandle tpHandle{0};
    uint32_t mappedJettyPriority{0};
    bool hasMappedJettyPriority{false};

    TpInfo() = default;
    TpInfo(const TpHandle handle) : tpHandle(handle) {}
};

struct TpAttrInfo {
    struct TpAttr tpAttr {};

    TpAttrInfo() = default;
    TpAttrInfo(const struct TpAttr& attr) : tpAttr(attr) {}
};

struct GetTpAttrParam {
    TpHandle tpHandle{0};
    uint32_t attrBitmap{0};

    GetTpAttrParam() = default;
    GetTpAttrParam(const TpHandle handle, const uint32_t bitmap) : tpHandle(handle), attrBitmap(bitmap) {}

    std::string Describe() const
    {
        return Hccl::StringFormat("GetTpAttrParam[tpHandle=0x%llx, attrBitmap=0x%x]", tpHandle, attrBitmap);
    }
};

class TpManager {
public:
    static TpManager& GetInstance(const int32_t deviceLogicId);
    void Init();
    /// `isSync==true`：同步路径（HOST_NET ctx + HrtRa* 同步接口），一次返回 SUCCESS。
    /// `isSync==false`：异步三阶段，轮询返回 HCCL_E_AGAIN。
    HcclResult GetTpInfo(const RaUbGetTpInfoParam& param, TpInfo& tpInfo, bool isSync = false);
    // UB_RTP 的多个 Jetty 可能复用同一 TP。PSN 必须跟随 TP 缓存稳定复用，不能依赖异步完成顺序。
    HcclResult GetOrSetLocalPsn(const RaUbGetTpInfoParam& param, const TpInfo& tpInfo, uint32_t& localPsn);
    // unimport jetty 会 URMA 销毁 tp 资源，hccl 配套删除记录
    HcclResult ReleaseTpInfo(const RaUbGetTpInfoParam& param, const TpInfo& tpInfo);
    HcclResult GetTpAttr(const GetTpAttrParam& param, TpAttrInfo& tpAttrInfo, RdmaHandle rdmaHandle);
    HcclResult ReleaseTpAttr(const TpHandle tpHandle, const TpAttrInfo& tpAttrInfo);

    static HcclResult GetTpTotalTimeout(const TpAttrInfo& tpAttrInfo, uint32_t& tpTimeOutMs);
    static uint32_t TaHwValueToMs(uint8_t hwValue);
    static uint8_t FindMinTaHwValue(uint32_t tpTotalTimeoutMs);

    /// 哨兵值：调用方未传入 TA 超时值，CalcTaTimeout 将回退到协议默认值
    static constexpr uint8_t TA_TIMEOUT_NOT_SET = 0xFFU;

    /// 计算 Jetty 异步创建超时挡位（hw_value, 0-31）
    /// 入参 taTimeOut 由调用方按协议从环境变量获取后传入；TA_TIMEOUT_NOT_SET 表示未传入，使用协议默认值
    /// 入参 tpTimeOutMs 由 GetTpTotalTimeout 计算得到，用于环境变量时间 < TP 总超时时的自动升挡
    static uint8_t CalcTaTimeout(TpProtocol tpProtocol, uint8_t taTimeOut, uint32_t tpTimeOutMs);

private:
    bool initFlag{false};
    uint32_t devLogicId{0};
    uint32_t devPhyId{0};

    struct TpInfoCtx {
        TpInfo tpInfo{};
        uint32_t useCnt{0};
        uint32_t localPsn{0};
        bool hasLocalPsn{false};

        TpInfoCtx() = default;
        TpInfoCtx(const TpInfo& info, const uint32_t cnt) : tpInfo(info), useCnt(cnt) {}
    };

    struct TpAttrCtx {
        TpAttrInfo tpAttrInfo{};
        uint32_t useCnt{0};

        TpAttrCtx() = default;
        TpAttrCtx(const TpAttrInfo& info, const uint32_t cnt) : tpAttrInfo(info), useCnt(cnt) {}
    };

    /*
     * Request上下文，保存查询TP信息相关调用异步接口出参
     * handle: 异步接口调用handle，用于查询处理结果
     * tpInfoNum: 查询到的TP信息个数，当前为复用TP，只会申请1个
     * dataBuffer: 查询到的TP信息数据，原始数据保留缓冲区
     */
    struct RequestCtx {
        enum class ReqPhase : uint8_t { WAIT_LIST = 0, WAIT_TP_ATTR = 1 };
        ReqPhase phase{ReqPhase::WAIT_LIST};
        RequestHandle handle{0};
        uint32_t tpInfoNum{0};
        std::vector<char_t> dataBuffer;
        TpAttr tpAttr{};
        uint32_t tpAttrBitmap{0};
    };

    struct TpAttrRequestCtx {
        RequestHandle handle{0};
        struct TpAttr tpAttr {};
        uint32_t attrBitmap{0};
    };

    /// 三级索引：先按本端 IP，再按对端 IP，最后按 QoS 键（`QosKey`：`param.qos & 0xFF`，与 next `TpMgr` 一致）。
    using InfoQosMap = std::unordered_map<uint32_t, TpInfoCtx>;
    using InfoRmtMap = std::unordered_map<IpAddress, InfoQosMap>;
    using InfoCtxMap = std::unordered_map<IpAddress, InfoRmtMap>;
    using ReqQosMap = std::unordered_map<uint32_t, RequestCtx>;
    using ReqRmtMap = std::unordered_map<IpAddress, ReqQosMap>;
    using ReqCtxMap = std::unordered_map<IpAddress, ReqRmtMap>;

    using TpAttrCtxMap = std::unordered_map<TpHandle, TpAttrCtx>;
    using TpAttrReqCtxMap = std::unordered_map<TpHandle, TpAttrRequestCtx>;

    InfoCtxMap ctpInfoMap;
    ReqCtxMap ctpReqMap;

    InfoCtxMap tpInfoMap;
    ReqCtxMap tpReqMap;

    InfoCtxMap uboeInfoMap;
    ReqCtxMap uboeReqMap;

    InfoCtxMap ubRtpInfoMap;
    ReqCtxMap ubRtpReqMap;

    TpAttrCtxMap tpAttrCtxMap;
    TpAttrReqCtxMap tpAttrReqCtxMap;

    std::mutex ctpInfoMutex;
    std::mutex ctpReqMutex;

    std::mutex tpInfoMutex;
    std::mutex tpReqMutex;

    std::mutex uboeInfoMutex;
    std::mutex uboeReqMutex;

    std::mutex ubRtpInfoMutex;
    std::mutex ubRtpReqMutex;

    std::mutex tpAttrCtxMutex;
    std::mutex tpAttrReqMutex;

    TpManager() = default;
    ~TpManager() = default;
    TpManager(const TpManager& that) = delete;
    TpManager& operator=(const TpManager& that) = delete;

    HcclResult FindAndGetTpInfo(const RaUbGetTpInfoParam& param, TpInfo& tpInfo);
    HcclResult RunSyncGetTpInfo(const RaUbGetTpInfoParam& param, TpInfo& tpInfo);
    HcclResult RunAsyncGetTpInfo(const RaUbGetTpInfoParam& param, TpInfo& tpInfo);
    HcclResult StoreTpInfoResult(const RaUbGetTpInfoParam& param, TpInfo& tpInfo);
    HcclResult SyncGetFirstTpAttrForSlPolicy(
        const RaUbGetTpInfoParam& param, uint64_t firstTpHandle, TpAttr& tpAttr, uint32_t& attrBitmap) const;
    HcclResult AdvanceDeviceWaitListPhase(
        const RaUbGetTpInfoParam& param, RequestCtx& reqCtx, ReqQosMap& qosReqMap, ReqQosMap::iterator it,
        std::unique_lock<std::mutex>& reqCtxLock, TpInfo& tpInfo);
    void StartGetTpInfoListRequest(const RaUbGetTpInfoParam& param, RequestCtx& reqCtx, bool isSync) const;
    HcclResult FindAndGetTpAttr(const TpHandle tpHandle, TpAttrInfo& tpAttrInfo);
    HcclResult
    StartGetTpAttrRequest(const GetTpAttrParam& param, TpAttrRequestCtx& reqCtx, RdmaHandle rdmaHandle) const;
    HcclResult
    HandleCompletedTpAttrRequest(const TpAttrRequestCtx reqCtx, const TpHandle tpHandle, TpAttrInfo& tpAttrInfo);

    void StartGetTpAttrForFirstTpDevice(const RaUbGetTpInfoParam& param, RequestCtx& reqCtx) const;
    HcclResult
    HandleCompletedRequest(const RequestCtx reqCtx, const RaUbGetTpInfoParam& param, TpInfo& tpInfo, bool withSlPolicy);
    HcclResult MapTpInfoFromTpAttr(
        const RaUbGetTpInfoParam& param, const RequestCtx& reqCtx, TpInfo& outTpInfo, bool isSync) const;

    bool CheckRequestResult(RequestHandle& reqHandle) const;
    InfoCtxMap& GetInfoCtxMap(const TpProtocol tpProtocol);
    ReqCtxMap& GetReqCtxMap(const TpProtocol tpProtocol);
    std::mutex& GetInfoCtxMutex(const TpProtocol tpProtocol);
    std::mutex& GetReqCtxMutex(const TpProtocol tpProtocol);
};

/// UbConnection 释放 TpInfo：Release 键须与 GetTpInfo 时的业务 QoS 一致
void ReleaseUbConnectionTp(
    int32_t devLogicId, const IpAddress& locAddr, const IpAddress& rmtAddr, TpProtocol tpProtocol, TpInfo& tpInfo,
    uint32_t requestQos);

} // namespace Hccl

#endif // HCCLV2_TP_MANAGER_H
