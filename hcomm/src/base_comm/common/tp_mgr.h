/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TP_MGR_H
#define TP_MGR_H

#include <cstdint>
#include <functional>
#include <mutex>
#include <vector>
#include <unordered_map>

#include "hccl_types.h"
#include "hcomm_adapter_hccp.h"
#include "hccp_ctx_tp.h"
#include "orion_adpt_utils.h"

namespace hcomm {

// ==================== TA 档位映射表 ====================
// TA 芯片挡位 = hw_value / 8
// 挡位 0 (0-7):   512ms
// 挡位 1 (8-15):  1000ms (1s)
// 挡位 2 (16-23): 8000ms (8s)
// 挡位 3 (24-31): 32000ms (32s)
// 各挡位对应的最小硬件配置值
constexpr uint8_t TA_HW_GEAR0_BASE = 0;
constexpr uint8_t TA_HW_GEAR1_BASE = 8;
constexpr uint8_t TA_HW_GEAR2_BASE = 16;
constexpr uint8_t TA_HW_GEAR3_BASE = 24;

constexpr uint8_t TA_GEAR_INDEX_0 = 0;
constexpr uint8_t TA_GEAR_INDEX_1 = 1;
constexpr uint8_t TA_GEAR_INDEX_2 = 2;
constexpr uint8_t TA_GEAR_INDEX_3 = 3;

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

using GetTpInfoParam = struct GetTpInfoParamDef {
    CommAddr locAddr{};
    CommAddr rmtAddr{};
    TpProtocol tpProtocol{TpProtocol::CTP};
    /// 参与 TP/SL 分组与缓存键（0–7），连接侧已归一化
    uint32_t qos{EnvConfig::UB_QOS_DEFAULT};
    /// 非 0 时与 sl_available 推导的 M 取 min 作为可用档位数上限
    uint32_t slLevelCount{0};
    /// 环回等场景：首 TPID + 掩码内最小 SL
    bool loopFirstTpLowestSl{false};
    /// 仅 CCU 设备环回 GetTpInfo：与通信域 hcclQos 解耦，SL 来自 GetTpAttr.slBitmap；写回 SL 经
    /// HrtRaSetTpAttrAsync（HDC）
    bool ccuLoopbackGetTpInfo{false};

    explicit GetTpInfoParamDef() = default;
    GetTpInfoParamDef(const CommAddr& locAddr, const CommAddr& rmtAddr, TpProtocol tpProtocol)
        : locAddr(locAddr),
          rmtAddr(rmtAddr),
          tpProtocol(tpProtocol)
    {}

    std::string Describe() const
    {
        Hccl::IpAddress locIpAddr{}, rmtIpAddr{};
        (void)CommAddrToIpAddress(locAddr, locIpAddr);
        (void)CommAddrToIpAddress(rmtAddr, rmtIpAddr);
        return Hccl::StringFormat(
            "RaUbGetTpInfoParam[locAddr=%s, rmtAddr=%s, tpProtocol=%s, qos=%u, loopFirstTpLowestSl=%d, ccuLoop=%d]",
            locIpAddr.Describe().c_str(), rmtIpAddr.Describe().c_str(), tpProtocol.Describe().c_str(), qos,
            static_cast<int>(loopFirstTpLowestSl), static_cast<int>(ccuLoopbackGetTpInfo));
    }
};

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
    explicit TpInfo(const TpHandle handle) : tpHandle(handle) {}
};

struct TpAttrInfo {
    struct TpAttr tpAttr {};

    TpAttrInfo() = default;
    TpAttrInfo(const struct TpAttr& attr) : tpAttr(attr) {}
};

using GetTpAttrParam = struct GetTpAttrParamDef {
    TpHandle tpHandle{0};
    uint32_t attrBitmap{0};

    explicit GetTpAttrParamDef() = default;
    GetTpAttrParamDef(const TpHandle tpHandle, const uint32_t attrBitmap)
        : tpHandle(tpHandle),
          attrBitmap(attrBitmap) {};

    std::string Describe() const
    {
        return Hccl::StringFormat(
            "GetTpAttrParam[tpHandle=0x%llx, attrBitmap=0x%x]", static_cast<unsigned long long>(tpHandle), attrBitmap);
    }
};

class TpMgr {
public:
    static TpMgr& GetInstance(const uint32_t devicePhyId);
    HcclResult GetTpInfo(const GetTpInfoParam& param, TpInfo& tpInfo);
    HcclResult GetTpAttr(const GetTpAttrParam& param, TpAttrInfo& tpAttrInfo, CtxHandle ctxHandle);
    // unimport jetty 会 URMA 销毁 tp 资源，hccl 配套删除记录
    HcclResult ReleaseTpInfo(const GetTpInfoParam& param, const TpInfo& tpInfo);
    HcclResult ReleaseTpAttr(const TpHandle tpHandle, const TpAttrInfo& tpAttrInfo);
    static HcclResult GetTpTotalTimeout(const TpAttrInfo& tpAttrInfo, uint32_t& tpTimeOutMs);

    /// 哨兵值：调用方未传入 TA 超时值，CalcTaTimeout 将回退到协议默认值
    static constexpr uint8_t TA_TIMEOUT_NOT_SET = 0xFFU;

    /// 计算 Jetty 异步创建超时挡位（hw_value, 0-31）
    /// 入参 taTimeOut 由调用方按协议从环境变量获取后传入；TA_TIMEOUT_NOT_SET 表示未传入，使用协议默认值
    /// 入参 tpTimeOutMs 由 GetTpTotalTimeout 计算得到，用于环境变量时间 < TP 总超时时的自动升挡
    static uint8_t CalcTaTimeout(TpProtocol tpProtocol, uint8_t taTimeOut, uint32_t tpTimeOutMs);

private:
    struct TpInfoCtx {
        TpInfo tpInfo{};
        uint32_t useCnt{0};

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
    enum class ReqPhase : uint8_t { WAIT_LIST = 0, WAIT_TP_ATTR = 1 };

    struct RequestCtx {
        ReqPhase phase{ReqPhase::WAIT_LIST};
        RequestHandle handle{0};
        uint32_t tpInfoNum{0};
        std::vector<char> dataBuffer;
        TpAttr tpAttr{};
        uint32_t tpAttrBitmap{0};
    };

    struct TpAttrRequestCtx {
        RequestHandle handle{0};
        struct TpAttr tpAttr {};
    };

    using TpAttrCtxMap = std::unordered_map<TpHandle, TpAttrCtx>;
    using TpAttrReqCtxMap = std::unordered_map<TpHandle, TpAttrRequestCtx>;

    /// 三级索引：先按本端 IP，再按对端 IP，最后按 QoS 档（0–7，与 GetTpInfo/TP-SL 策略里用的档位一致）。
    using InfoQosMap = std::unordered_map<uint32_t, TpInfoCtx>;
    using InfoRmtMap = std::unordered_map<Hccl::IpAddress, InfoQosMap>;
    using InfoCtxMap = std::unordered_map<Hccl::IpAddress, InfoRmtMap>;
    using ReqQosMap = std::unordered_map<uint32_t, RequestCtx>;
    using ReqRmtMap = std::unordered_map<Hccl::IpAddress, ReqQosMap>;
    using ReqCtxMap = std::unordered_map<Hccl::IpAddress, ReqRmtMap>;

private:
    TpMgr() = default;
    ~TpMgr() = default;
    TpMgr(const TpMgr& that) = delete;
    TpMgr& operator=(const TpMgr& that) = delete;

    HcclResult FindAndGetTpInfo(const GetTpInfoParam& param, TpInfo& tpInfo);
    HcclResult FindAndGetTpAttr(const TpHandle tpHandle, TpAttrInfo& tpAttrInfo);
    HcclResult LookupInfoCtxEntry(
        InfoCtxMap& infoMap, const Hccl::IpAddress& locAddr, const Hccl::IpAddress& rmtAddr, const QosKey qosKey,
        InfoCtxMap::iterator& lit, InfoRmtMap::iterator& rit, InfoQosMap::iterator& qosIt) const;
    HcclResult
    PollGetTpInfoReqCtx(std::unique_lock<std::mutex>& reqCtxLock, const GetTpInfoParam& param, TpInfo& tpInfo);
    HcclResult BeginGetTpInfoListRequest(const GetTpInfoParam& param, ReqQosMap& qosMap, const QosKey qosKey) const;
    HcclResult AdvanceGetTpInfoWaitList(
        const GetTpInfoParam& param, RequestCtx& reqCtx, ReqQosMap& qosMap, const ReqQosMap::iterator it,
        std::unique_lock<std::mutex>& reqCtxLock, TpInfo& tpInfo);

    HcclResult StartGetTpInfoListRequest(const GetTpInfoParam& param, RequestCtx& reqCtx) const;
    HcclResult StartGetTpAttrForFirstTp(const GetTpInfoParam& param, RequestCtx& reqCtx) const;
    HcclResult StartGetTpAttrRequest(const GetTpAttrParam& param, TpAttrRequestCtx& reqCtx, CtxHandle ctxHandle) const;
    HcclResult BuildTpInfoAndCommitQosAttr(
        const GetTpInfoParam& param, const RequestCtx& reqCtx, const struct HccpTpInfo* baseInfoPtr,
        const uint32_t tpListIndex, const uint32_t mappedSl, TpInfo& tpInfo) const;
    HcclResult CommitTpInfoToCache(const GetTpInfoParam& param, TpInfo& tpInfo);
    HcclResult HandleCompletedRequest(RequestCtx reqCtx, const GetTpInfoParam& param, TpInfo& tpInfo);
    HcclResult
    HandleCompletedTpAttrRequest(const TpAttrRequestCtx reqCtx, const TpHandle tpHandle, TpAttrInfo& tpAttrInfo);

    InfoCtxMap& GetInfoCtxMap(const TpProtocol tpProtocol);
    ReqCtxMap& GetReqCtxMap(const TpProtocol tpProtocol);
    std::mutex& GetInfoCtxMutex(const TpProtocol tpProtocol);
    std::mutex& GetReqCtxMutex(const TpProtocol tpProtocol);

private:
    bool initFlag_{false};
    uint32_t devPhyId_{0};

    InfoCtxMap ctpInfoMap_;
    ReqCtxMap ctpReqMap_;

    InfoCtxMap rtpInfoMap_;
    ReqCtxMap rtpReqMap_;

    InfoCtxMap uboeInfoMap_;
    ReqCtxMap uboeReqMap_;

    TpAttrCtxMap tpAttrCtxMap_;
    TpAttrReqCtxMap tpAttrReqCtxMap_;

    std::mutex ctpInfoMutex_;
    std::mutex ctpReqMutex_;

    std::mutex rtpInfoMutex_;
    std::mutex rtpReqMutex_;

    std::mutex uboeInfoMutex_;
    std::mutex uboeReqMutex_;

    std::mutex tpAttrCtxMutex_;
    std::mutex tpAttrReqMutex_;
};

} // namespace hcomm

#endif // TP_MGR_H
