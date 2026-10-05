/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "cast_utils.h"
#include "ub_transport_lite_impl.h"
#include "binary_stream.h"
#include "ub_conn_lite_mgr.h"
#include "exception_util.h"
#include "internal_exception.h"
#include "communicator_impl_lite_manager.h"
#include "dfx_profiling_handler_lite.h"

namespace Hccl {
constexpr u32 UB_WQE_MAX_SIZE = 128; // 针对WriteWithNotify类型WQE，最大是128Byte
constexpr u32 UB_INLINE_WRITE_SIZE = 4;
constexpr u32 UB_RELAX_ORDER = 0X01; // Relax Order表示当前SQE与后续Strong Order SQE有保序要求
constexpr u32 UB_STRONG_ORDER = 0X02; // Strong Order表示当前SQE有保序要求，该SQE不能超越前面的Relax Order SQE
constexpr u32 UB_NO_COMPLETION = 0; // 表示当前报文和前面报文没有completion序要求，报文对应的CQE可以乱序上报
constexpr u32 UB_COMPLETION = 1; // 表示当前报文和前面报文有completion序要求，报文对应的CQE需要保序上报
constexpr u8 UB_FENCE_ENABLED = 1; // fence使能
UbTransportLiteImpl::UbTransportLiteImpl(
    std::vector<char>& uniqueId, std::function<void(u32 streamId, u32 taskId, const TaskParam& taskParam)> callback)
{
    callback_ = callback;
    // [header...][notifyUniqueId...][rmtNotifyUniqueId...][rmtBufferUniqueIds...]
    BinaryStream binaryStream(uniqueId);
    u32 theType;
    binaryStream >> theType;
    binaryStream >> notifyNum;
    binaryStream >> bufferNum;
    binaryStream >> rmtbufferNum;
    binaryStream >> connNum;
    linkType_ = (theType == static_cast<u32>(TransportType::UB)) ? DfxLinkType::UB : DfxLinkType::UBoE;

    std::vector<char> notifyUniqueIds;
    binaryStream >> notifyUniqueIds;
    ParseLocNotifyVec(notifyUniqueIds);

    std::vector<char> rmtNotifyUniqueIds;
    binaryStream >> rmtNotifyUniqueIds;
    ParseRmtBufferVec(rmtNotifyUniqueIds, RmaUbBufType::NOTIFY);

    std::vector<char> rmtBufferUniqueIds;
    binaryStream >> rmtBufferUniqueIds;
    ParseRmtBufferVec(rmtBufferUniqueIds, RmaUbBufType::BUFFER);

    std::vector<char> connUniqueIds;
    binaryStream >> connUniqueIds;
    ParseConnVec(connUniqueIds);
}
UbTransportLiteImpl::UbTransportLiteImpl(std::vector<char>& uniqueId) { Init(uniqueId); }

void UbTransportLiteImpl::Init(std::vector<char>& uniqueId)
{
    BinaryStream binaryStream(uniqueId);
    u32 theType;
    binaryStream >> theType;
    binaryStream >> notifyNum;
    binaryStream >> bufferNum;
    binaryStream >> rmtbufferNum;
    binaryStream >> connNum;
    linkType_ = (theType == static_cast<u32>(TransportType::UB)) ? DfxLinkType::UB : DfxLinkType::UBoE;

    std::vector<char> notifyUniqueIds;
    binaryStream >> notifyUniqueIds;
    ParseLocNotifyVec(notifyUniqueIds);

    std::vector<char> rmtNotifyUniqueIds;
    binaryStream >> rmtNotifyUniqueIds;
    ParseRmtBufferVec(rmtNotifyUniqueIds, RmaUbBufType::NOTIFY);

    std::vector<char> locBufferUniqueIds;
    binaryStream >> locBufferUniqueIds;
    ParseLocBufferMap(locBufferUniqueIds);

    std::vector<char> rmtBufferUniqueIds;
    binaryStream >> rmtBufferUniqueIds;
    ParseRmtBufferVec(rmtBufferUniqueIds, RmaUbBufType::BUFFER);

    // 解析drain相关的资源信息
    std::vector<char> drainBufferUniqueIds;
    binaryStream >> drainBufferUniqueIds;
    ParseDrainResource(drainBufferUniqueIds);

    std::vector<char> connUniqueIds;
    binaryStream >> connUniqueIds;
    ParseConnVec(connUniqueIds);
}

UbTransportLiteImpl::~UbTransportLiteImpl()
{
    for (auto& it : connUniqueIdVec) {
        DECTOR_TRY_CATCH("UbTransportLiteImpl", UbConnLiteMgr::GetInstance().Clear(it));
    }
}

std::string UbTransportLiteImpl::Describe() const
{
    std::string desc = "UbTransportLiteImpl[";

    u32 idx = 0;
    desc += "locNotifyVec=[";
    for (auto& it : locNotifyVec) {
        desc += StringFormat("idx=%u, %s;", idx, it->Describe().c_str());
        idx++;
    }

    idx = 0;
    desc += "], rmtNotifyVec=[";
    for (auto& it : rmtNotifyVec) {
        desc += StringFormat("idx=%u, %s;", idx, it.Describe().c_str());
        idx++;
    }

    idx = 0;
    desc += "], rmtBufferVec=[";
    for (auto& it : rmtBufferVec) {
        desc += StringFormat("idx=%u, %s;", idx, it.Describe().c_str());
        idx++;
    }

    idx = 0;
    desc += "], connVec=[";
    for (auto& it : connVec) {
        desc += StringFormat("idx=%u, %s;", idx, it->Describe().c_str());
        idx++;
    }

    desc += "]]";
    return desc;
}

void UbTransportLiteImpl::ParseLocNotifyVec(std::vector<char>& data)
{
    if (notifyNum == 0) {
        HCCL_WARNING("UbTransportLiteImpl::ParseLocNotifyVec num is 0");
        return;
    }
    u32 notifySizePerDto = data.size() / notifyNum;

    for (u32 idx = 0; idx < notifyNum; idx++) {
        auto start = data.begin() + idx * notifySizePerDto;
        auto end = start + notifySizePerDto;
        std::vector<char> dto(start, end);
        locNotifyVec.push_back(std::make_unique<NotifyLite>(dto));
        HCCL_INFO("locNotify idx=%u, %s", idx, locNotifyVec.back()->Describe().c_str());
    }
}

void UbTransportLiteImpl::ParseRmtBufferVec(std::vector<char>& data, RmaUbBufType rmtType)
{
    u32 num = 0;
    if (rmtType == RmaUbBufType::NOTIFY) {
        num = notifyNum;
    } else {
        num = rmtbufferNum;
    }

    if (num == 0) {
        HCCL_WARNING("UbTransportLiteImpl::ParseRmtBufferVec %s num is 0", rmtType.Describe().c_str());
        return;
    }

    u32 rmtBufferSizePerDto = data.size() / num;
    HCCL_INFO("Parse %s num=%u, sizePerDto=%u", rmtType.Describe().c_str(), num, rmtBufferSizePerDto);
    BinaryStream binaryStream(data);

    for (u32 idx = 0; idx < num; idx++) {
        RmtUbBufLite ubBufLite;
        binaryStream >> ubBufLite.addr;
        binaryStream >> ubBufLite.size;
        binaryStream >> ubBufLite.tokenId;
        binaryStream >> ubBufLite.tokenValue;
        binaryStream >> ubBufLite.notifyId;
        HCCL_INFO("idx=%u, %s %s", idx, rmtType.Describe().c_str(), ubBufLite.Describe().c_str());
        if (rmtType == RmaUbBufType::NOTIFY) {
            rmtNotifyVec.push_back(ubBufLite);
        } else {
            rmtBufferMap[static_cast<uintptr_t>(ubBufLite.addr)] = ubBufLite;
            rmtBufferVec.push_back(ubBufLite);
        }
    }
}

void UbTransportLiteImpl::ParseLocBufferMap(std::vector<char>& data)
{
    u32 num = bufferNum;

    if (num == 0) {
        HCCL_WARNING("UbTransportLiteImpl::ParseLocBufferMap num is 0");
        return;
    }

    u32 rmtBufferSizePerDto = data.size() / num;
    HCCL_INFO("ParseLocBufferMap num=%u, sizePerDto=%u", num, rmtBufferSizePerDto);
    BinaryStream binaryStream(data);

    for (u32 idx = 0; idx < num; idx++) {
        LocUbBufLite ubBufLite;
        binaryStream >> ubBufLite.addr;
        binaryStream >> ubBufLite.size;
        binaryStream >> ubBufLite.tokenId;
        binaryStream >> ubBufLite.tokenValue;
        HCCL_INFO("idx=%u, LocBuffer %s", idx, ubBufLite.Describe().c_str());
        locBufferMap[static_cast<uintptr_t>(ubBufLite.addr)] = ubBufLite;
    }
}

void UbTransportLiteImpl::ParseDrainResource(std::vector<char>& data)
{
    if (data.size() == 0) {
        HCCL_WARNING("UbTransportLiteImpl::ParseDrainResource is null");
        return;
    }

    BinaryStream binaryStream(data);
    binaryStream >> drainNotify_.addr;
    binaryStream >> drainNotify_.size;
    binaryStream >> drainNotify_.tokenId;
    binaryStream >> drainNotify_.tokenValue;
    binaryStream >> drainNotify_.notifyId;
    HCCL_INFO("drain notify %s", drainNotify_.Describe().c_str());

    binaryStream >> rmtDrainBuffer_.addr;
    binaryStream >> rmtDrainBuffer_.size;
    binaryStream >> rmtDrainBuffer_.tokenId;
    binaryStream >> rmtDrainBuffer_.tokenValue;
    binaryStream >> rmtDrainBuffer_.notifyId;
    HCCL_INFO("drain remote buffer %s", rmtDrainBuffer_.Describe().c_str());
}

void UbTransportLiteImpl::ParseConnVec(std::vector<char>& data)
{
    if (connNum == 0) {
        HCCL_WARNING("UbTransportLiteImpl::ParseConnVec num is 0");
        return;
    }
    u32 connSizePerDto = data.size() / connNum;
    HCCL_INFO("Parse ConnVec num=%u, connSizePerDto=%u", connNum, connSizePerDto);
    for (u32 idx = 0; idx < connNum; idx++) {
        auto start = data.begin() + idx * connSizePerDto;
        auto end = start + connSizePerDto;
        std::vector<char> connUniqueId(start, end);
        connUniqueIdVec.push_back(connUniqueId);
        // connLite的复用由 ubConnLiteMgr管理
        auto lite = UbConnLiteMgr::GetInstance().Get(connUniqueId);
        connVec.push_back(lite);
        HCCL_INFO("[%s]idx=%u, %s", __func__, idx, lite->Describe().c_str());
    }
    CheckConnVec("after ParseConnVec");
}

// [中文导读] [AllReduce逐行 S260] UbTransportLiteImpl::BuildUbDbSendTask的接口声明：承载任务的执行流、UB jetty的die/function/jetty标识、UB jetty生产指针（16位自然增长）；这些参数属于本函数调用边界。
void UbTransportLiteImpl::BuildUbDbSendTask(const StreamLite& stream, const UbJettyLiteId& jettyLiteId, u32 pi)
// [中文导读] [AllReduce逐行 S261] 进入UbTransportLiteImpl::BuildUbDbSendTask函数体：把 jetty 标识和 UB PI 交给当前 StreamLite 的 RTSQ 生成 Doorbell SQE。
{
    // [中文导读] [AllReduce逐行 S262] 返回当前StreamLite持有的具体执行队列；构造携带jetty标识与UB PI的Doorbell SQE，刷新当前RTSQ提交状态；传入/处理承载任务的执行流的GetRtsq字段、UB jetty的die/function/jetty标识、UB jetty生产指针（16位自然增长）。
    stream.GetRtsq()->UbDbSend(jettyLiteId, pi);
// [中文导读] [AllReduce逐行 S263] 结束UbTransportLiteImpl::BuildUbDbSendTask函数体；控制流返回外层。
}

void UbTransportLiteImpl::BuildNotifyWaitTask(const StreamLite& stream, u32 notifyId)
{
    stream.GetRtsq()->NotifyWait(notifyId);
}

Buffer UbTransportLiteImpl::GetRmtBuffer(u32 index)
{
    if (UNLIKELY(index >= rmtBufferVec.size())) {
        THROW<InternalException>(StringFormat(
            "UbTransportLiteImpl::GetRmtBuffer out-of-bounds. index=%u, size=%u", index, rmtBufferVec.size()));
    }
    return Buffer(rmtBufferVec[index].addr, rmtBufferVec[index].size);
}

// [中文导读] [AllReduce逐行 S279] UbTransportLiteImpl::GetRmtNotifySliceLite的接口声明：条目下标；这些参数属于本函数调用边界。
RmtRmaBufSliceLite UbTransportLiteImpl::GetRmtNotifySliceLite(u32 index)
// [中文导读] [AllReduce逐行 S280] 进入UbTransportLiteImpl::GetRmtNotifySliceLite函数体：按远端通知索引取得地址/token/notifyId 组成通知 slice。
{
    // [中文导读] [AllReduce逐行 S281] 设置RmtUbBufLite& lite为/按`rmtNotifyVec[index]`（通道远端通知资源数组、条目下标）。
    RmtUbBufLite& lite = rmtNotifyVec[index];
    // ub conn lite 不关心rkey , rkey 设定为0
    // [中文导读] [AllReduce逐行 S283] 直接返回`RmtRmaBufSliceLite(lite.addr, lite.size, 0, lite.tokenId, lite.tokenValue, lite.notifyId)`；调用RmtRmaBufSliceLite。
    return RmtRmaBufSliceLite(lite.addr, lite.size, 0, lite.tokenId, lite.tokenValue, lite.notifyId);
// [中文导读] [AllReduce逐行 S284] 结束UbTransportLiteImpl::GetRmtNotifySliceLite函数体；控制流返回外层。
}

// [中文导读] [AllReduce逐行 S286] UbTransportLiteImpl::GetRmtRmaBufSliceLite的接口声明：在远端已交换内存范围中解析地址/长度/token；无匹配抛异常；这些参数属于本函数调用边界。
RmtRmaBufSliceLite UbTransportLiteImpl::GetRmtRmaBufSliceLite(const Buffer& rmtBuf)
// [中文导读] [AllReduce逐行 S287] 进入UbTransportLiteImpl::GetRmtRmaBufSliceLite函数体：在远端已交换内存范围中解析地址/长度/token；无匹配抛异常。
{
    // [中文导读] [AllReduce逐行 S288] 设置auto it为/按`rmtBufferMap.upper_bound(rmtBuf.GetAddr())`（远端交换得到的注册范围映射的upper_bound字段）；定位第一个起始地址严格大于请求地址的内存登记项；读取缓冲区起始地址。
    auto it = rmtBufferMap.upper_bound(rmtBuf.GetAddr());

    // [中文导读] [AllReduce逐行 S290] 在`(it != rmtBufferMap.begin())`（远端交换得到的注册范围映射的begin字段）条件下重复执行后续等待或分片处理。
    while (it != rmtBufferMap.begin()) {
        // [中文导读] [AllReduce逐行 S291] 将内存登记迭代器向较小起始地址方向移动一项，以检查该项是否覆盖请求范围。
        --it;
        // [中文导读] [AllReduce逐行 S292] 调用iterBuf；保持声明的局部对象用于后续处理。
        Buffer iterBuf(it->second.addr, it->second.size);
        // [中文导读] [AllReduce逐行 S293] 仅当`(iterBuf.Contains(rmtBuf.GetAddr(), rmtBuf.GetSize()))`成立时进入此分支；判断请求地址和完整字节范围是否包含在此已注册区内；读取缓冲区起始地址；读取缓冲区字节长度。
        if (iterBuf.Contains(rmtBuf.GetAddr(), rmtBuf.GetSize())) {
            // [中文导读] [AllReduce逐行 S294] 直接返回`RmtRmaBufSliceLite(`；调用RmtRmaBufSliceLite。
            return RmtRmaBufSliceLite(
                // [中文导读] [AllReduce逐行 S295] 为读取缓冲区起始地址；读取缓冲区字节长度补入`rmtBuf.GetAddr(), rmtBuf.GetSize(), 0, it->second.tokenId, it->second.tokenValue, UINT32_MAX)`；本行是参数/结构化初始化续行。
                rmtBuf.GetAddr(), rmtBuf.GetSize(), 0, it->second.tokenId, it->second.tokenValue, UINT32_MAX);
        // [中文导读] [AllReduce逐行 S296] 结束`if (iterBuf.Contains(rmtBuf.GetAddr(), rmtBuf.GetSize()))`分支/循环；控制流返回外层。
        }
    // [中文导读] [AllReduce逐行 S297] 结束`while (it != rmtBufferMap.begin())`（远端交换得到的注册范围映射的begin字段）分支/循环；控制流返回外层。
    }
    // [中文导读] [AllReduce逐行 S298] 抛出当前内部/驱动异常并中止此操作；上层EXCEPTION捕获边界负责转为返回状态。
    MACRO_THROW(InternalException, StringFormat("%s is not in current transport", rmtBuf.Describe().c_str()));
// [中文导读] [AllReduce逐行 S299] 结束UbTransportLiteImpl::GetRmtRmaBufSliceLite函数体；控制流返回外层。
}

RmtRmaBufSliceLite UbTransportLiteImpl::GetRmtRmaBufSliceLite(const RmaBufferLite& lite) const
{
    return RmtRmaBufSliceLite(lite.GetAddr(), lite.GetSize(), 0, lite.GetTokenId(), lite.GetTokenValue(), UINT32_MAX);
}

// [中文导读] [AllReduce逐行 S306] UbTransportLiteImpl::BuildLocRmaBufferLite的接口声明：从本端注册范围查找 RMA token；不包含请求时警告并按源码继续使用查到的 token；这些参数属于本函数调用边界。
HcclResult
// [中文导读] [AllReduce逐行 S307] UbTransportLiteImpl::BuildLocRmaBufferLite的接口声明：待访问的本地内存地址、字节容量或单片字节数、本端RMA描述出参；这些参数属于本函数调用边界。
UbTransportLiteImpl::BuildLocRmaBufferLite(const uintptr_t addr, const size_t size, RmaBufferLite& rmaBufferLite)
// [中文导读] [AllReduce逐行 S308] 进入UbTransportLiteImpl::BuildLocRmaBufferLite函数体：从本端注册范围查找 RMA token；不包含请求时警告并按源码继续使用查到的 token。
{
    // [中文导读] [AllReduce逐行 S309] 记录UbTransportLiteImpl::BuildLocRmaBufferLite的状态/性能诊断；日志本身不执行传输。
    HCCL_INFO(
        // [中文导读] [AllReduce逐行 S310] 为当前UbTransportLiteImpl::BuildLocRmaBufferLite诊断/异常表达式提供格式文本；这一物理行没有数据搬运副作用。
        "[UbTransportLiteImpl::%s] start to find addr[0x%llx], size[0x%llx] in locBufferMap, whose size is %zu. ",
        // [中文导读] [AllReduce逐行 S311] 为读取容器登记项数补入`__func__, addr, size, locBufferMap.size())`（待访问的本地内存地址、字节容量或单片字节数、本端已注册范围映射的size字段）；本行是参数/结构化初始化续行。
        __func__, addr, size, locBufferMap.size());
    // [中文导读] [AllReduce逐行 S312] 仅当`(locBufferMap.empty())`（本端已注册范围映射的empty字段）成立时进入此分支；检查容器是否没有登记项。
    if (locBufferMap.empty()) {
        // [中文导读] [AllReduce逐行 S313] 记录UbTransportLiteImpl::BuildLocRmaBufferLite的错误诊断；日志本身不执行传输。
        HCCL_ERROR("[UbTransportLiteImpl::%s] locBufferMap is empty.", __func__);
        // [中文导读] [AllReduce逐行 S314] 返回HCCL_E_INTERNAL，表示内部处理失败；此路径停止本函数的后续处理。
        return HCCL_E_INTERNAL;
    // [中文导读] [AllReduce逐行 S315] 结束`if (locBufferMap.empty())`（本端已注册范围映射的empty字段）分支/循环；控制流返回外层。
    }

    // [中文导读] [AllReduce逐行 S317] 设置bool isAddrInRange为/按`false`。
    bool isAddrInRange = false;
    // [中文导读] [AllReduce逐行 S318] 设置auto it为/按`locBufferMap.upper_bound(addr)`（本端已注册范围映射的upper_bound字段、待访问的本地内存地址）；定位第一个起始地址严格大于请求地址的内存登记项。
    auto it = locBufferMap.upper_bound(addr);

    // [中文导读] [AllReduce逐行 S320] 在`(it != locBufferMap.begin())`（本端已注册范围映射的begin字段）条件下重复执行后续等待或分片处理。
    while (it != locBufferMap.begin()) {
        // [中文导读] [AllReduce逐行 S321] 将内存登记迭代器向较小起始地址方向移动一项，以检查该项是否覆盖请求范围。
        --it;
        // [中文导读] [AllReduce逐行 S322] 调用iterBuf；保持声明的局部对象用于后续处理。
        Buffer iterBuf(it->second.addr, it->second.size);
        // [中文导读] [AllReduce逐行 S323] 仅当`(iterBuf.Contains(addr, size))`（待访问的本地内存地址、字节容量或单片字节数）成立时进入此分支；判断请求地址和完整字节范围是否包含在此已注册区内。
        if (iterBuf.Contains(addr, size)) {
            // [中文导读] [AllReduce逐行 S324] 设置本端RMA描述出参为/按`RmaBufferLite(addr, size, it->second.tokenId, it->second.tokenValue)`（待访问的本地内存地址、字节容量或单片字节数）；调用RmaBufferLite，使用待访问的本地内存地址、字节容量或单片字节数。
            rmaBufferLite = RmaBufferLite(addr, size, it->second.tokenId, it->second.tokenValue);
            // [中文导读] [AllReduce逐行 S325] 设置isAddrInRange为/按`true`。
            isAddrInRange = true;
            // [中文导读] [AllReduce逐行 S326] 退出当前循环或switch路径，继续其后处理；不代表其他执行流已经完成。
            break;
        // [中文导读] [AllReduce逐行 S327] 结束`if (iterBuf.Contains(addr, size))`（待访问的本地内存地址、字节容量或单片字节数）分支/循环；控制流返回外层。
        }
    // [中文导读] [AllReduce逐行 S328] 结束`while (it != locBufferMap.begin())`（本端已注册范围映射的begin字段）分支/循环；控制流返回外层。
    }

    // [中文导读] [AllReduce逐行 S330] 仅当`(!isAddrInRange)`成立时进入此分支。
    if (!isAddrInRange) {
        // [中文导读] [AllReduce逐行 S331] 记录UbTransportLiteImpl::BuildLocRmaBufferLite的警告诊断；日志本身不执行传输。
        HCCL_WARNING(
            // [中文导读] [AllReduce逐行 S332] 为当前UbTransportLiteImpl::BuildLocRmaBufferLite诊断/异常表达式提供格式文本；这一物理行没有数据搬运副作用。
            "[UbTransportLiteImpl::%s] addr[0x%llx], size[0x%llx] not in any range of locBufferMap, use the first in "
            // [中文导读] [AllReduce逐行 S333] 为当前UbTransportLiteImpl::BuildLocRmaBufferLite诊断/异常表达式提供格式文本；这一物理行没有数据搬运副作用。
            "map addr[0x%llx] size[0x%llx]",
            // [中文导读] [AllReduce逐行 S334] 为前述多行表达式补入`__func__, addr, size, it->second.addr, it->second.size)`（待访问的本地内存地址、字节容量或单片字节数）；本行是参数/结构化初始化续行。
            __func__, addr, size, it->second.addr, it->second.size);
        // [中文导读] [AllReduce逐行 S335] 未找到包含请求范围的登记项时，仍使用迭代器当前项token构造请求地址/长度的RMA描述；此分支只告警，没有范围失败返回。
        rmaBufferLite = RmaBufferLite(addr, size, it->second.tokenId, it->second.tokenValue);
    // [中文导读] [AllReduce逐行 S336] 结束`if (!isAddrInRange)`分支/循环；控制流返回外层。
    }

    // [中文导读] [AllReduce逐行 S338] 当前路径返回成功状态；仅说明本函数处理/任务组织成功，完成语义由其具体调用职责决定。
    return HCCL_SUCCESS;
// [中文导读] [AllReduce逐行 S339] 结束UbTransportLiteImpl::BuildLocRmaBufferLite函数体；控制流返回外层。
}

void UbTransportLiteImpl::ClearConnOut()
{
    wqeData.clear();
    wqeData.resize(UB_WQE_MAX_SIZE);
    connOut.data = (u8*)wqeData.data();
    connOut.dataSize = sizeof(wqeData);
}

// 检查connection不能为空
void UbTransportLiteImpl::CheckConnVec(const std::string& desc)
{
    if (UNLIKELY(connVec.size() == 0)) {
        THROW<InternalException>(StringFormat("connVec size is 0 %s", desc.c_str()));
    }

    u32 idx = 0;
    for (auto& it : connVec) {
        if (UNLIKELY(it == nullptr)) {
            THROW<InternalException>(StringFormat("connVec[%u] is null %s", idx, desc.c_str()));
        }
        idx++;
    }
}

// [中文导读] [AllReduce逐行 S365] UbTransportLiteImpl::GetRmaBufSlicelite的接口声明：将本地 RMA 描述转为 UB slice，rkey 按 UB 约定为0；这些参数属于本函数调用边界。
RmaBufSliceLite UbTransportLiteImpl::GetRmaBufSlicelite(const RmaBufferLite& lite) const
// [中文导读] [AllReduce逐行 S366] 进入UbTransportLiteImpl::GetRmaBufSlicelite函数体：将本地 RMA 描述转为 UB slice，rkey 按 UB 约定为0。
{
    // ub conn lite 不关心rkey , rkey 设定为0
    // [中文导读] [AllReduce逐行 S368] 直接返回`RmaBufSliceLite(lite.GetAddr(), lite.GetSize(), 0, lite.GetTokenId())`；读取缓冲区起始地址；读取缓冲区字节长度；读取当前内存注册token ID。
    return RmaBufSliceLite(lite.GetAddr(), lite.GetSize(), 0, lite.GetTokenId());
// [中文导读] [AllReduce逐行 S369] 结束UbTransportLiteImpl::GetRmaBufSlicelite函数体；控制流返回外层。
}

// [中文导读] [AllReduce逐行 S371] UbTransportLiteImpl::Post的接口声明：条目下标、承载任务的执行流；这些参数属于本函数调用边界。
void UbTransportLiteImpl::Post(u32 index, const StreamLite& stream)
// [中文导读] [AllReduce逐行 S372] 进入UbTransportLiteImpl::Post函数体：对端通知区写内联值1，PostFin 索引1加强完成/保序。
{
    // [中文导读] [AllReduce逐行 S373] 准备UB WQE保序/完成配置的局部存储/结构描述，初始化方式以本行声明为准。
    SqeConfigLite cfg;
    // [中文导读] [AllReduce逐行 S374] 仅当`(index == 1)`（条目下标）成立时进入此分支。
    if (index == 1) { // PostFin场景
        // [中文导读] [AllReduce逐行 S375] 设置UB WQE保序/完成配置的cqeEn字段为/按`true`。
        cfg.cqeEn = true;
        // [中文导读] [AllReduce逐行 S376] 设置UB WQE保序/完成配置的placeOdr字段为/按`UB_STRONG_ORDER`。
        cfg.placeOdr = UB_STRONG_ORDER;
        // [中文导读] [AllReduce逐行 S377] 设置UB WQE保序/完成配置的compOrder字段为/按`UB_COMPLETION`。
        cfg.compOrder = UB_COMPLETION;
        // [中文导读] [AllReduce逐行 S378] 设置UB WQE保序/完成配置的userConfig字段为/按`true`。
        cfg.userConfig = true;
    // [中文导读] [AllReduce逐行 S379] 结束`if (index == 1)`（条目下标）分支/循环；控制流返回外层。
    }
    // [中文导读] [AllReduce逐行 S380] 设置内联通知值1为/按`1`。
    u32 inlineData = 1;

    // [中文导读] [AllReduce逐行 S382] 设置当前任务编号为/按`stream.GetRtsq()->GetTaskId()`（承载任务的执行流的GetRtsq字段）；返回当前StreamLite持有的具体执行队列；读取当前队列taskId，用于该操作与观测信息关联。
    auto taskId = stream.GetRtsq()->GetTaskId();

    // 当前使用1个connection，下标为0 构建sqe
    // [中文导读] [AllReduce逐行 S385] 设置所选RMA连接对象为/按`connVec[0]`（当前transport的RMA连接数组）。
    RmaConnLite* conn = connVec[0];

    // 展开下发WQE前, 按需设置cache context
    // [中文导读] [AllReduce逐行 S388] 设置用于WQE缓存/调试的UB连接对象为/按`nullptr`。
    UbConnLite* ubConnLitePtr = nullptr;
    // [中文导读] [AllReduce逐行 S389] 设置是否记录本轮任务缓存为/按`false`。
    bool needCacheTask = false;
    // [中文导读] [AllReduce逐行 S390] 按缓存/日志配置开启WQE跟踪并取得具体UB连接；传入/处理用于WQE缓存/调试的UB连接对象、是否记录本轮任务缓存、所选RMA连接对象。
    PreLaunchWqe(ubConnLitePtr, needCacheTask, conn);
    // 下发DbSqe前, 备份相关信息
    // [中文导读] [AllReduce逐行 S392] 设置本地待提交SQE条数为/按`needCacheTask ? stream.GetRtsq()->GetPendingSqeCnt() : 0`（是否记录本轮任务缓存、承载任务的执行流的GetRtsq字段）；返回当前StreamLite持有的具体执行队列；读取Doorbell生成之前当前执行队列的待提交SQE数。
    const uint32_t pendingSqeCnt = needCacheTask ? stream.GetRtsq()->GetPendingSqeCnt() : 0;

    // 展开下发WQE
    // [中文导读] [AllReduce逐行 S395] 设置auto rmtBuffSliceLite为/按`GetRmtNotifySliceLite(index)`（条目下标）；按对端通知槽取远端通知地址/token/notifyId。
    auto rmtBuffSliceLite = GetRmtNotifySliceLite(index);
    // [中文导读] [AllReduce逐行 S396] 生成向远端通知区内联写值1的WQE；传入/处理所选RMA连接对象的InlineWrite字段、内联通知值1、UB WQE保序/完成配置、承载任务的执行流、WQE构造后的UB PI等输出。
    conn->InlineWrite(ReinterpretAs<u8*>(&inlineData), UB_INLINE_WRITE_SIZE, rmtBuffSliceLite, cfg, stream, connOut);

    // 展开下发WQE后, 展开下发DbSqe前, 按需缓存wqe及DbSqeIdx
    // 注意: pendingSqeCnt在下发DbSqe前已备份
    // 注意: 一定要在展开下发DbSqe前调用PostLaunchWqe, 否则如果下发DbSqe触发LaunchTask,
    // 而尚未调用PostLaunchWqe插入当前WQE数组，
    //     会导致AicpuTaskCache找不到当前WQE数组, 无法正确更新对应的DbSqeLocation
    // [中文导读] [AllReduce逐行 S403] 设置是否生成任务观测记录为/按`IsReportTask()`；检查是否需要任务异常/性能观测。
    const bool isReportTask = IsReportTask();
    // [中文导读] [AllReduce逐行 S404] 准备Doorbell任务的缓存观测描述的局部存储/结构描述，初始化方式以本行声明为准。
    DbSqeProfInfo dbSqeProfInfo;
    // [中文导读] [AllReduce逐行 S405] 仅当`(needCacheTask && isReportTask)`（是否记录本轮任务缓存、是否生成任务观测记录）成立时进入此分支。
    if (needCacheTask && isReportTask) { // 构造DbSqeProfInfo
        // [中文导读] [AllReduce逐行 S406] 设置Doorbell任务的缓存观测描述的isValid字段为/按`true`。
        dbSqeProfInfo.isValid = true;
        // [中文导读] [AllReduce逐行 S407] 设置Doorbell任务的缓存观测描述的taskParamType字段为/按`TaskParamType::TASK_UB_INLINE_WRITE`。
        dbSqeProfInfo.taskParamType = TaskParamType::TASK_UB_INLINE_WRITE;
        // [中文导读] [AllReduce逐行 S408] 调用FillDbSqeProfInfoDmaPub。
        FillDbSqeProfInfoDmaPub(
            // [中文导读] [AllReduce逐行 S409] 为读取缓冲区起始地址；读取缓冲区字节长度补入`ReinterpretAs<void*>(rmtBuffSliceLite.GetAddr()), rmtBuffSliceLite.GetSize(), DmaOp::HCCL_DMA_WRITE,`；本行是参数/结构化初始化续行。
            ReinterpretAs<void*>(rmtBuffSliceLite.GetAddr()), rmtBuffSliceLite.GetSize(), DmaOp::HCCL_DMA_WRITE,
            // [中文导读] [AllReduce逐行 S410] 为读取缓冲区起始地址；读取缓冲区字节长度补入`dbSqeProfInfo)`（Doorbell任务的缓存观测描述）；本行是参数/结构化初始化续行。
            dbSqeProfInfo);
        // [中文导读] [AllReduce逐行 S411] 设置Doorbell任务的缓存观测描述的notifyId字段为/按`rmtBuffSliceLite.GetNotifyId()`；调用GetNotifyId。
        dbSqeProfInfo.notifyId = rmtBuffSliceLite.GetNotifyId();
    // [中文导读] [AllReduce逐行 S412] 结束`if (needCacheTask && isReportTask)`（是否记录本轮任务缓存、是否生成任务观测记录）分支/循环；控制流返回外层。
    }
    // [中文导读] [AllReduce逐行 S413] 在Doorbell之前保存/打印已展开WQE及DbSqe位置，然后关闭跟踪；传入/处理承载任务的执行流、用于WQE缓存/调试的UB连接对象、是否记录本轮任务缓存、本地待提交SQE条数、是否生成任务观测记录、Doorbell任务的缓存观测描述。
    PostLaunchWqe(stream, ubConnLitePtr, needCacheTask, pendingSqeCnt, isReportTask, dbSqeProfInfo);

    // 构建rts 的 sqe
    // [中文导读] [AllReduce逐行 S416] 给当前执行流生成携带UB PI的jetty Doorbell SQE；取得UB jetty的die/function/jetty标识用于Doorbell；传入/处理承载任务的执行流、所选RMA连接对象的GetUbJettyLiteId字段、WQE构造后的UB PI等输出的pi字段。
    BuildUbDbSendTask(stream, conn->GetUbJettyLiteId(), connOut.pi);

    // [中文导读] [AllReduce逐行 S418] 记录UbTransportLiteImpl::Post的状态/性能诊断；日志本身不执行传输。
    HCCL_INFO(
        // [中文导读] [AllReduce逐行 S419] 为当前UbTransportLiteImpl::Post诊断/异常表达式提供格式文本；这一物理行没有数据搬运副作用。
        "UbTransportLiteImpl::Post notifyId[%u], pi=[%u], locEid[%s], rmtEid[%s]", rmtBuffSliceLite.GetNotifyId(),
        // [中文导读] [AllReduce逐行 S420] 为取得本端UB网络地址标识；取得对象诊断文本用于日志；取得对端UB网络地址标识补入`connOut.pi, GetLocEid().Describe().c_str(), GetRmtEid().Describe().c_str())`（WQE构造后的UB PI等输出的pi字段）；本行是参数/结构化初始化续行。
        connOut.pi, GetLocEid().Describe().c_str(), GetRmtEid().Describe().c_str());

    // [中文导读] [AllReduce逐行 S422] 登记对端通知写入的观测信息。
    NotifyRecordProfilingProcess(
        // [中文导读] [AllReduce逐行 S423] 为登记对端通知写入的观测信息；读取缓冲区起始地址；读取缓冲区字节长度补入`ReinterpretAs<void*>(rmtBuffSliceLite.GetAddr()), rmtBuffSliceLite.GetSize(), stream, taskId,`（承载任务的执行流、当前任务编号）；本行是参数/结构化初始化续行。
        ReinterpretAs<void*>(rmtBuffSliceLite.GetAddr()), rmtBuffSliceLite.GetSize(), stream, taskId,
        // [中文导读] [AllReduce逐行 S424] 为登记对端通知写入的观测信息；读取缓冲区起始地址；读取缓冲区字节长度补入`rmtBuffSliceLite.GetNotifyId())`；本行是参数/结构化初始化续行。
        rmtBuffSliceLite.GetNotifyId());
// [中文导读] [AllReduce逐行 S425] 结束UbTransportLiteImpl::Post函数体；控制流返回外层。
}

void UbTransportLiteImpl::Wait(u32 index, const StreamLite& stream)
{
    WaitWithTimeout(index, stream, CommunicatorImplLiteMgr::GetInstance().GetEnvConfig().hcclExecTimeout);
}

// [中文导读] [AllReduce逐行 S432] UbTransportLiteImpl::WaitWithTimeout的接口声明：条目下标、承载任务的执行流、超时秒数；这些参数属于本函数调用边界。
void UbTransportLiteImpl::WaitWithTimeout(u32 index, const StreamLite& stream, u32 timeout)
// [中文导读] [AllReduce逐行 S433] 进入UbTransportLiteImpl::WaitWithTimeout函数体：本地通知索引转为硬件 ID，并向 RTSQ 排入等待 SQE 与观测。
{
    // [中文导读] [AllReduce逐行 S434] 设置当前任务编号为/按`stream.GetRtsq()->GetTaskId()`（承载任务的执行流的GetRtsq字段）；返回当前StreamLite持有的具体执行队列；读取当前队列taskId，用于该操作与观测信息关联。
    auto taskId = stream.GetRtsq()->GetTaskId();
    // [中文导读] [AllReduce逐行 S435] 设置硬件通知ID为/按`locNotifyVec[index]->GetId()`（通道本地通知资源数组、条目下标）；读取执行流/通知资源的实际ID。
    auto notifyId = locNotifyVec[index]->GetId();
    // [中文导读] [AllReduce逐行 S436] 返回当前StreamLite持有的具体执行队列；生成指定硬件通知ID的等待SQE；传入/处理承载任务的执行流的GetRtsq字段、硬件通知ID、超时秒数。
    stream.GetRtsq()->NotifyWait(notifyId, timeout);

    // [中文导读] [AllReduce逐行 S438] 仅当`(!IsReportTask())`成立时进入此分支；检查是否需要任务异常/性能观测。
    if (!IsReportTask()) {
        // [中文导读] [AllReduce逐行 S439] 结束本次void调用，当前路径不再继续下发后续操作。
        return;
    // [中文导读] [AllReduce逐行 S440] 结束`if (!IsReportTask())`分支/循环；控制流返回外层。
    }

    // [中文导读] [AllReduce逐行 S442] 仅当`(callback_)`（任务观测回调）成立时进入此分支。
    if (callback_) {
        // [中文导读] [AllReduce逐行 S443] 准备`TaskParam taskParam{}`的局部存储/结构描述，初始化方式以本行声明为准。
        TaskParam taskParam{};
        // [中文导读] [AllReduce逐行 S444] 设置taskParam.taskType为/按`TaskParamType::TASK_NOTIFY_WAIT`。
        taskParam.taskType = TaskParamType::TASK_NOTIFY_WAIT;
        // [中文导读] [AllReduce逐行 S445] 设置taskParam.beginTime为/按`ProfGetCurCpuTimestamp()`；读取性能观测CPU时间戳。
        taskParam.beginTime = ProfGetCurCpuTimestamp();
        // [中文导读] [AllReduce逐行 S446] 设置taskParam.taskPara.Notify.notifyID为/按`notifyId`（硬件通知ID）。
        taskParam.taskPara.Notify.notifyID = notifyId;
        // [中文导读] [AllReduce逐行 S447] 设置taskParam.taskPara.Notify.value为/按`1`。
        taskParam.taskPara.Notify.value = 1;
        // [中文导读] [AllReduce逐行 S448] 调用任务观测回调登记本次taskId；传入/处理承载任务的执行流、当前任务编号。
        AddTaskCallback(stream, taskId, taskParam);
    // [中文导读] [AllReduce逐行 S449] 结束`if (callback_)`（任务观测回调）分支/循环；控制流返回外层。
    }
    // [中文导读] [AllReduce逐行 S450] 填充通道通知等待的诊断任务信息；传入/处理承载任务的执行流、当前任务编号。
    FillSlotWaitInfo(stream, taskId);
// [中文导读] [AllReduce逐行 S451] 结束UbTransportLiteImpl::WaitWithTimeout函数体；控制流返回外层。
}

void UbTransportLiteImpl::ProfilingProcess(
    void* src, void* dst, u64 size, const StreamLite& stream, DmaOp dmaOp, u32 taskId)
{
    if (!IsReportTask()) {
        return;
    }

    if (callback_) {
        TaskParam taskParam{};
        taskParam.taskType = TaskParamType::TASK_UB;
        taskParam.beginTime = ProfGetCurCpuTimestamp();
        FillTaskParamDmaPub(taskParam, dst, size, dmaOp);
        taskParam.taskPara.DMA.src = src;
        AddTaskCallback(stream, taskId, taskParam);
    }
    FillSlotUbDmaInfo(
        stream, taskId, TaskParamTypeVal::TASK_UB, ReinterpretAs<u64>(src), ReinterpretAs<u64>(dst), size, INVALID_U32);
}

void UbTransportLiteImpl::ReduceProfilingProcess(
    void* src, void* dst, u64 size, const ReduceIn& reduceIn, const StreamLite& stream, u32 taskId)

{
    if (!IsReportTask()) {
        return;
    }

    if (callback_) {
        TaskParam taskParam{};
        taskParam.taskType = TaskParamType::TASK_UB_REDUCE_INLINE;
        taskParam.beginTime = ProfGetCurCpuTimestamp();
        FillTaskParamReducePub(taskParam, src, dst, size, reduceIn);
        taskParam.taskPara.Reduce.notifyID = INVALID_VALUE_NOTIFYID;
        AddTaskCallback(stream, taskId, taskParam);
    }
    FillSlotReduceInfo(
        stream, taskId, TaskParamTypeVal::TASK_UB_REDUCE_INLINE, ReinterpretAs<u64>(src), ReinterpretAs<u64>(dst), size,
        INVALID_U32, static_cast<u8>(ConvertReduceOpToHcclReduceOp(reduceIn.reduceOp)));
}

void UbTransportLiteImpl::WriteWithNotifyProfilingProcess(
    void* src, void* dst, u64 size, const StreamLite& stream, u32 taskId, u64 notifyId)
{
    if (!IsReportTask()) {
        return;
    }

    if (callback_) {
        TaskParam taskParam{};
        taskParam.taskType = TaskParamType::TASK_WRITE_WITH_NOTIFY;
        taskParam.beginTime = ProfGetCurCpuTimestamp();
        FillTaskParamDmaPub(taskParam, dst, size, DmaOp::HCCL_DMA_WRITE);
        taskParam.taskPara.DMA.src = src;
        taskParam.taskPara.DMA.notifyID = notifyId;
        taskParam.taskPara.DMA.notifyValue = 1;
        AddTaskCallback(stream, taskId, taskParam);
    }
    FillSlotUbDmaInfo(
        stream, taskId, TaskParamTypeVal::TASK_WRITE_WITH_NOTIFY, ReinterpretAs<u64>(src), ReinterpretAs<u64>(dst),
        size, static_cast<u32>(notifyId));
}

void UbTransportLiteImpl::WriteReduceWithNotifyProfilingProcess(
    void* src, void* dst, u64 size, const ReduceIn& reduceIn, const StreamLite& stream, u32 taskId, u64 notifyId)
{
    if (!IsReportTask()) {
        return;
    }

    if (callback_) {
        TaskParam taskParam{};
        taskParam.taskType = TaskParamType::TASK_WRITE_REDUCE_WITH_NOTIFY;
        taskParam.beginTime = ProfGetCurCpuTimestamp();
        FillTaskParamReducePub(taskParam, src, dst, size, reduceIn);
        taskParam.taskPara.Reduce.notifyID = notifyId;
        AddTaskCallback(stream, taskId, taskParam);
    }
    FillSlotReduceInfo(
        stream, taskId, TaskParamTypeVal::TASK_WRITE_REDUCE_WITH_NOTIFY, ReinterpretAs<u64>(src),
        ReinterpretAs<u64>(dst), size, static_cast<u32>(notifyId),
        static_cast<u8>(ConvertReduceOpToHcclReduceOp(reduceIn.reduceOp)));
}

void UbTransportLiteImpl::NotifyRecordProfilingProcess(
    void* dst, u64 size, const StreamLite& stream, u32 taskId, u64 notifyId)
{
    if (!IsReportTask()) {
        return;
    }

    if (callback_) {
        TaskParam taskParam{};
        taskParam.taskType = TaskParamType::TASK_UB_INLINE_WRITE;
        taskParam.beginTime = ProfGetCurCpuTimestamp();
        FillTaskParamDmaPub(taskParam, dst, size, DmaOp::HCCL_DMA_WRITE);
        taskParam.taskPara.DMA.notifyID = notifyId;
        taskParam.taskPara.DMA.notifyValue = 1;
        AddTaskCallback(stream, taskId, taskParam);
    }
    FillSlotUbDmaInfo(
        stream, taskId, TaskParamTypeVal::TASK_UB_INLINE_WRITE, 0, ReinterpretAs<u64>(dst), size,
        static_cast<u32>(notifyId));
}

void UbTransportLiteImpl::FillSlotUbDmaInfo(
    const StreamLite& stream, u32 taskId, TaskParamTypeVal taskType, u64 srcAddr, u64 dstAddr, u64 size,
    u32 notifyId) const
{
    DfxTaskInfo* slot = stream.NextTaskSlot();
    slot->taskType = static_cast<u8>(taskType);
    slot->sqId = stream.GetSqId();
    slot->taskId = taskId;
    const void* opInfo = stream.GetLatestDfxOpInfo();
    slot->dfxOpInfo = (opInfo != nullptr) ? ReinterpretAs<u64>(opInfo) : INVALID_U64;
    slot->linkType = (linkType_ == DfxLinkType::UB) ? DfxLinkTypeVal::LINK_UB : DfxLinkTypeVal::LINK_UBoE;
    slot->transportType = static_cast<u8>(DfxTransportType::DFX_TRANSPORT_TYPE_UB);
    slot->channelHandle = ReinterpretAs<u64>(this);
    slot->taskPara.ubDma.sqeAddr = stream.GetRtsq()->GetSqeAddr();
    slot->taskPara.ubDma.srcAddr = srcAddr;
    slot->taskPara.ubDma.dstAddr = dstAddr;
    slot->taskPara.ubDma.size = size;
    slot->taskPara.ubDma.notifyId = notifyId;
    slot->taskPara.ubDma.jettyHandle = GetJettyHandle();
    slot->taskPara.ubDma.jettyId = GetJettyId();
    slot->taskPara.ubDma.tpn = GetTpn();
    PLF_CONFIG_INFO(Hccl::PLF_TASK, "[%s] %s", __func__, slot->Describe().c_str());
}

void UbTransportLiteImpl::FillSlotReduceInfo(
    const StreamLite& stream, u32 taskId, TaskParamTypeVal taskType, u64 srcAddr, u64 dstAddr, u64 size, u32 notifyId,
    u8 reduceOp) const
{
    DfxTaskInfo* slot = stream.NextTaskSlot();
    slot->taskType = static_cast<u8>(taskType);
    slot->sqId = stream.GetSqId();
    slot->taskId = taskId;
    const void* opInfo = stream.GetLatestDfxOpInfo();
    slot->dfxOpInfo = (opInfo != nullptr) ? ReinterpretAs<u64>(opInfo) : INVALID_U64;
    slot->linkType = (linkType_ == DfxLinkType::UB) ? DfxLinkTypeVal::LINK_UB : DfxLinkTypeVal::LINK_UBoE;
    slot->transportType = static_cast<u8>(DfxTransportType::DFX_TRANSPORT_TYPE_UB);
    slot->channelHandle = ReinterpretAs<u64>(this);
    slot->taskPara.Reduce.sqeAddr = stream.GetRtsq()->GetSqeAddr();
    slot->taskPara.Reduce.srcAddr = srcAddr;
    slot->taskPara.Reduce.dstAddr = dstAddr;
    slot->taskPara.Reduce.size = size;
    slot->taskPara.Reduce.notifyId = notifyId;
    slot->taskPara.Reduce.reduceOp = reduceOp;
    slot->taskPara.Reduce.jettyHandle = GetJettyHandle();
    slot->taskPara.Reduce.jettyId = GetJettyId();
    slot->taskPara.Reduce.tpn = GetTpn();
    PLF_CONFIG_INFO(Hccl::PLF_TASK, "[%s] %s", __func__, slot->Describe().c_str());
}

void UbTransportLiteImpl::FillSlotWaitInfo(const StreamLite& stream, u32 taskId) const
{
    DfxTaskInfo* slot = stream.NextTaskSlot();
    slot->taskType = static_cast<u8>(TaskParamTypeVal::TASK_NOTIFY_WAIT);
    slot->sqId = stream.GetSqId();
    slot->taskId = taskId;
    const void* opInfo = stream.GetLatestDfxOpInfo();
    slot->dfxOpInfo = (opInfo != nullptr) ? ReinterpretAs<u64>(opInfo) : INVALID_U64;
    slot->linkType = (linkType_ == DfxLinkType::UB) ? DfxLinkTypeVal::LINK_UB : DfxLinkTypeVal::LINK_UBoE;
    slot->transportType = static_cast<u8>(DfxTransportType::DFX_TRANSPORT_TYPE_UB);
    slot->channelHandle = ReinterpretAs<u64>(this);
    slot->taskPara.Notify.sqeAddr = stream.GetRtsq()->GetSqeAddr();
    PLF_CONFIG_INFO(Hccl::PLF_TASK, "[%s] %s", __func__, slot->Describe().c_str());
}

void UbTransportLiteImpl::Read(const RmaBufferLite& loc, const Buffer& rmt, const StreamLite& stream)
{
    SqeConfigLite cfg;
    SetFenceConfig(cfg);

    auto taskId = stream.GetRtsq()->GetTaskId();

    // 当前使用1个connection,下标为0
    RmaConnLite* conn = connVec[0];

    // 展开下发WQE前, 按需设置cache context
    UbConnLite* ubConnLitePtr = nullptr;
    bool needCacheTask = false;
    PreLaunchWqe(ubConnLitePtr, needCacheTask, conn);
    // 下发DbSqe前, 备份相关信息
    const uint32_t pendingSqeCnt = needCacheTask ? stream.GetRtsq()->GetPendingSqeCnt() : 0;

    // 展开下发WQE
    auto locRmaBufSlicelite = GetRmaBufSlicelite(loc);
    auto rmtRmaBufSlicelite = GetRmtRmaBufSliceLite(rmt);
    conn->Read(locRmaBufSlicelite, rmtRmaBufSlicelite, cfg, stream, connOut);

    // 展开下发WQE后, 展开下发DbSqe前, 按需缓存wqe及DbSqeIdx
    // 注意: pendingSqeCnt在下发DbSqe前已备份
    // 注意: 一定要在展开下发DbSqe前调用PostLaunchWqe, 否则如果下发DbSqe触发LaunchTask,
    // 而尚未调用PostLaunchWqe插入当前WQE数组，
    //     会导致AicpuTaskCache找不到当前WQE数组, 无法正确更新对应的DbSqeLocation
    const bool isReportTask = IsReportTask();
    DbSqeProfInfo dbSqeProfInfo;
    if (needCacheTask && isReportTask) {
        BuildDbSqeProfInfoForProfilingProcess(
            ReinterpretAs<void*>(locRmaBufSlicelite.GetAddr()), ReinterpretAs<void*>(rmtRmaBufSlicelite.GetAddr()),
            locRmaBufSlicelite.GetSize(), DmaOp::HCCL_DMA_READ, dbSqeProfInfo);
    }
    PostLaunchWqe(stream, ubConnLitePtr, needCacheTask, pendingSqeCnt, isReportTask, dbSqeProfInfo);

    BuildUbDbSendTask(stream, conn->GetUbJettyLiteId(), connOut.pi);

    ProfilingProcess(
        ReinterpretAs<void*>(locRmaBufSlicelite.GetAddr()), ReinterpretAs<void*>(rmtRmaBufSlicelite.GetAddr()),
        locRmaBufSlicelite.GetSize(), stream, DmaOp::HCCL_DMA_READ, taskId);
}

// [中文导读] [AllReduce逐行 S664] UbTransportLiteImpl::Write的接口声明：本端缓冲区/地址、远端缓冲区/地址、承载任务的执行流；这些参数属于本函数调用边界。
void UbTransportLiteImpl::Write(const RmaBufferLite& loc, const Buffer& rmt, const StreamLite& stream)
// [中文导读] [AllReduce逐行 S665] 进入UbTransportLiteImpl::Write函数体：构造本地/远端 slice 后虚派发 UB WQE，再排入 Doorbell SQE 和观测。
{
    // [中文导读] [AllReduce逐行 S666] 准备UB WQE保序/完成配置的局部存储/结构描述，初始化方式以本行声明为准。
    SqeConfigLite cfg;
    // [中文导读] [AllReduce逐行 S667] 把当前传输Fence状态应用到本次WQE配置；传入/处理UB WQE保序/完成配置。
    SetFenceConfig(cfg);

    // [中文导读] [AllReduce逐行 S669] 设置当前任务编号为/按`stream.GetRtsq()->GetTaskId()`（承载任务的执行流的GetRtsq字段）；返回当前StreamLite持有的具体执行队列；读取当前队列taskId，用于该操作与观测信息关联。
    auto taskId = stream.GetRtsq()->GetTaskId();

    // 当前使用1个connection，下标为0
    // [中文导读] [AllReduce逐行 S672] 设置所选RMA连接对象为/按`connVec[0]`（当前transport的RMA连接数组）。
    RmaConnLite* conn = connVec[0];

    // 展开下发WQE前, 按需设置cache context
    // [中文导读] [AllReduce逐行 S675] 设置用于WQE缓存/调试的UB连接对象为/按`nullptr`。
    UbConnLite* ubConnLitePtr = nullptr;
    // [中文导读] [AllReduce逐行 S676] 设置是否记录本轮任务缓存为/按`false`。
    bool needCacheTask = false;
    // [中文导读] [AllReduce逐行 S677] 按缓存/日志配置开启WQE跟踪并取得具体UB连接；传入/处理用于WQE缓存/调试的UB连接对象、是否记录本轮任务缓存、所选RMA连接对象。
    PreLaunchWqe(ubConnLitePtr, needCacheTask, conn);
    // 下发DbSqe前, 备份相关信息
    // [中文导读] [AllReduce逐行 S679] 设置本地待提交SQE条数为/按`needCacheTask ? stream.GetRtsq()->GetPendingSqeCnt() : 0`（是否记录本轮任务缓存、承载任务的执行流的GetRtsq字段）；返回当前StreamLite持有的具体执行队列；读取Doorbell生成之前当前执行队列的待提交SQE数。
    const uint32_t pendingSqeCnt = needCacheTask ? stream.GetRtsq()->GetPendingSqeCnt() : 0;

    // 展开下发WQE
    // [中文导读] [AllReduce逐行 S682] 设置本地RMA切片为/按`GetRmaBufSlicelite(loc)`（本端缓冲区/地址）；将本端RMA描述改写为UB本地slice，rkey置0。
    auto locRmaBufSlicelite = GetRmaBufSlicelite(loc);
    // [中文导读] [AllReduce逐行 S683] 设置远端RMA切片为/按`GetRmtRmaBufSliceLite(rmt)`（远端缓冲区/地址）；从远端已交换注册区解析目标地址/长度/token。
    auto rmtRmaBufSlicelite = GetRmtRmaBufSliceLite(rmt);
    // [中文导读] [AllReduce逐行 S684] 向连接远端目标生成WRITE WQE，本端为源；传入/处理所选RMA连接对象的Write字段、本地RMA切片、远端RMA切片、UB WQE保序/完成配置、承载任务的执行流、WQE构造后的UB PI等输出。
    conn->Write(locRmaBufSlicelite, rmtRmaBufSlicelite, cfg, stream, connOut);

    // 展开下发WQE后, 展开下发DbSqe前, 按需缓存wqe及DbSqeIdx
    // 注意: pendingSqeCnt在下发DbSqe前已备份
    // 注意: 一定要在展开下发DbSqe前调用PostLaunchWqe, 否则如果下发DbSqe触发LaunchTask,
    // 而尚未调用PostLaunchWqe插入当前WQE数组，
    //     会导致AicpuTaskCache找不到当前WQE数组, 无法正确更新对应的DbSqeLocation
    // [中文导读] [AllReduce逐行 S691] 设置是否生成任务观测记录为/按`IsReportTask()`；检查是否需要任务异常/性能观测。
    const bool isReportTask = IsReportTask();
    // [中文导读] [AllReduce逐行 S692] 准备Doorbell任务的缓存观测描述的局部存储/结构描述，初始化方式以本行声明为准。
    DbSqeProfInfo dbSqeProfInfo;
    // [中文导读] [AllReduce逐行 S693] 仅当`(needCacheTask && isReportTask)`（是否记录本轮任务缓存、是否生成任务观测记录）成立时进入此分支。
    if (needCacheTask && isReportTask) {
        // [中文导读] [AllReduce逐行 S694] 保存Doorbell关联数据搬运信息供缓存重放观测。
        BuildDbSqeProfInfoForProfilingProcess(
            // [中文导读] [AllReduce逐行 S695] 为保存Doorbell关联数据搬运信息供缓存重放观测；读取缓冲区起始地址补入`ReinterpretAs<void*>(locRmaBufSlicelite.GetAddr()), ReinterpretAs<void*>(rmtRmaBufSlicelite.GetAddr()),`（本地RMA切片的GetAddr字段、远端RMA切片的GetAddr字段）；本行是参数/结构化初始化续行。
            ReinterpretAs<void*>(locRmaBufSlicelite.GetAddr()), ReinterpretAs<void*>(rmtRmaBufSlicelite.GetAddr()),
            // [中文导读] [AllReduce逐行 S696] 为保存Doorbell关联数据搬运信息供缓存重放观测；读取缓冲区起始地址；读取缓冲区字节长度补入`locRmaBufSlicelite.GetSize(), DmaOp::HCCL_DMA_WRITE, dbSqeProfInfo)`（本地RMA切片的GetSize字段、Doorbell任务的缓存观测描述）；本行是参数/结构化初始化续行。
            locRmaBufSlicelite.GetSize(), DmaOp::HCCL_DMA_WRITE, dbSqeProfInfo);
    // [中文导读] [AllReduce逐行 S697] 结束`if (needCacheTask && isReportTask)`（是否记录本轮任务缓存、是否生成任务观测记录）分支/循环；控制流返回外层。
    }
    // [中文导读] [AllReduce逐行 S698] 在Doorbell之前保存/打印已展开WQE及DbSqe位置，然后关闭跟踪；传入/处理承载任务的执行流、用于WQE缓存/调试的UB连接对象、是否记录本轮任务缓存、本地待提交SQE条数、是否生成任务观测记录、Doorbell任务的缓存观测描述。
    PostLaunchWqe(stream, ubConnLitePtr, needCacheTask, pendingSqeCnt, isReportTask, dbSqeProfInfo);

    // [中文导读] [AllReduce逐行 S700] 给当前执行流生成携带UB PI的jetty Doorbell SQE；取得UB jetty的die/function/jetty标识用于Doorbell；传入/处理承载任务的执行流、所选RMA连接对象的GetUbJettyLiteId字段、WQE构造后的UB PI等输出的pi字段。
    BuildUbDbSendTask(stream, conn->GetUbJettyLiteId(), connOut.pi);

    // [中文导读] [AllReduce逐行 S702] 登记UB数据传输方向/地址/长度的观测信息。
    ProfilingProcess(
        // [中文导读] [AllReduce逐行 S703] 为登记UB数据传输方向/地址/长度的观测信息；读取缓冲区起始地址补入`ReinterpretAs<void*>(locRmaBufSlicelite.GetAddr()), ReinterpretAs<void*>(rmtRmaBufSlicelite.GetAddr()),`（本地RMA切片的GetAddr字段、远端RMA切片的GetAddr字段）；本行是参数/结构化初始化续行。
        ReinterpretAs<void*>(locRmaBufSlicelite.GetAddr()), ReinterpretAs<void*>(rmtRmaBufSlicelite.GetAddr()),
        // [中文导读] [AllReduce逐行 S704] 为登记UB数据传输方向/地址/长度的观测信息；读取缓冲区起始地址；读取缓冲区字节长度补入`locRmaBufSlicelite.GetSize(), stream, DmaOp::HCCL_DMA_WRITE, taskId)`（本地RMA切片的GetSize字段、承载任务的执行流、当前任务编号）；本行是参数/结构化初始化续行。
        locRmaBufSlicelite.GetSize(), stream, DmaOp::HCCL_DMA_WRITE, taskId);
// [中文导读] [AllReduce逐行 S705] 结束UbTransportLiteImpl::Write函数体；控制流返回外层。
}

// [中文导读] [AllReduce逐行 S707] UbTransportLiteImpl::ReadReduce的接口声明：构造读取归约 WQE，再组织 Doorbell SQE 与归约观测；这些参数属于本函数调用边界。
void UbTransportLiteImpl::ReadReduce(
    // [中文导读] [AllReduce逐行 S708] UbTransportLiteImpl::ReadReduce的接口声明：本端缓冲区/地址、远端缓冲区/地址、底层归约类型/操作描述、承载任务的执行流；这些参数属于本函数调用边界。
    const RmaBufferLite& loc, const Buffer& rmt, const ReduceIn& reduceIn, const StreamLite& stream)
// [中文导读] [AllReduce逐行 S709] 进入UbTransportLiteImpl::ReadReduce函数体：构造读取归约 WQE，再组织 Doorbell SQE 与归约观测。
{
    // [中文导读] [AllReduce逐行 S710] 准备UB WQE保序/完成配置的局部存储/结构描述，初始化方式以本行声明为准。
    SqeConfigLite cfg;
    // [中文导读] [AllReduce逐行 S711] 把当前传输Fence状态应用到本次WQE配置；传入/处理UB WQE保序/完成配置。
    SetFenceConfig(cfg);

    // [中文导读] [AllReduce逐行 S713] 设置当前任务编号为/按`stream.GetRtsq()->GetTaskId()`（承载任务的执行流的GetRtsq字段）；返回当前StreamLite持有的具体执行队列；读取当前队列taskId，用于该操作与观测信息关联。
    auto taskId = stream.GetRtsq()->GetTaskId();

    // 当前使用1个connection，下标为0
    // [中文导读] [AllReduce逐行 S716] 设置所选RMA连接对象为/按`connVec[0]`（当前transport的RMA连接数组）。
    RmaConnLite* conn = connVec[0];

    // 展开下发WQE前, 按需设置cache context
    // [中文导读] [AllReduce逐行 S719] 设置用于WQE缓存/调试的UB连接对象为/按`nullptr`。
    UbConnLite* ubConnLitePtr = nullptr;
    // [中文导读] [AllReduce逐行 S720] 设置是否记录本轮任务缓存为/按`false`。
    bool needCacheTask = false;
    // [中文导读] [AllReduce逐行 S721] 按缓存/日志配置开启WQE跟踪并取得具体UB连接；传入/处理用于WQE缓存/调试的UB连接对象、是否记录本轮任务缓存、所选RMA连接对象。
    PreLaunchWqe(ubConnLitePtr, needCacheTask, conn);
    // 下发DbSqe前, 备份相关信息
    // [中文导读] [AllReduce逐行 S723] 设置本地待提交SQE条数为/按`needCacheTask ? stream.GetRtsq()->GetPendingSqeCnt() : 0`（是否记录本轮任务缓存、承载任务的执行流的GetRtsq字段）；返回当前StreamLite持有的具体执行队列；读取Doorbell生成之前当前执行队列的待提交SQE数。
    const uint32_t pendingSqeCnt = needCacheTask ? stream.GetRtsq()->GetPendingSqeCnt() : 0;

    // 展开下发WQE
    // [中文导读] [AllReduce逐行 S726] 设置本地RMA切片为/按`GetRmaBufSlicelite(loc)`（本端缓冲区/地址）；将本端RMA描述改写为UB本地slice，rkey置0。
    auto locRmaBufSlicelite = GetRmaBufSlicelite(loc);
    // [中文导读] [AllReduce逐行 S727] 设置远端RMA切片为/按`GetRmtRmaBufSliceLite(rmt)`（远端缓冲区/地址）；从远端已交换注册区解析目标地址/长度/token。
    auto rmtRmaBufSlicelite = GetRmtRmaBufSliceLite(rmt);
    // [中文导读] [AllReduce逐行 S728] 生成读取远端源并归约到本地目标的READ WQE；传入/处理所选RMA连接对象的ReadReduce字段、底层归约类型/操作描述、本地RMA切片、远端RMA切片、承载任务的执行流、UB WQE保序/完成配置、WQE构造后的UB PI等输出。
    conn->ReadReduce(reduceIn, locRmaBufSlicelite, rmtRmaBufSlicelite, stream, cfg, connOut);

    // 展开下发WQE后, 展开下发DbSqe前, 按需缓存wqe及DbSqeIdx
    // 注意: pendingSqeCnt在下发DbSqe前已备份
    // 注意: 一定要在展开下发DbSqe前调用PostLaunchWqe, 否则如果下发DbSqe触发LaunchTask,
    // 而尚未调用PostLaunchWqe插入当前WQE数组，
    //     会导致AicpuTaskCache找不到当前WQE数组, 无法正确更新对应的DbSqeLocation
    // [中文导读] [AllReduce逐行 S735] 设置是否生成任务观测记录为/按`IsReportTask()`；检查是否需要任务异常/性能观测。
    const bool isReportTask = IsReportTask();
    // [中文导读] [AllReduce逐行 S736] 准备Doorbell任务的缓存观测描述的局部存储/结构描述，初始化方式以本行声明为准。
    DbSqeProfInfo dbSqeProfInfo;
    // [中文导读] [AllReduce逐行 S737] 仅当`(needCacheTask && isReportTask)`（是否记录本轮任务缓存、是否生成任务观测记录）成立时进入此分支。
    if (needCacheTask && isReportTask) {
        // [中文导读] [AllReduce逐行 S738] 保存Doorbell关联归约信息供缓存重放观测。
        BuildDbSqeProfInfoForReduceProfilingProcess(
            // [中文导读] [AllReduce逐行 S739] 为保存Doorbell关联归约信息供缓存重放观测；读取缓冲区起始地址补入`ReinterpretAs<void*>(locRmaBufSlicelite.GetAddr()), ReinterpretAs<void*>(rmtRmaBufSlicelite.GetAddr()),`（本地RMA切片的GetAddr字段、远端RMA切片的GetAddr字段）；本行是参数/结构化初始化续行。
            ReinterpretAs<void*>(locRmaBufSlicelite.GetAddr()), ReinterpretAs<void*>(rmtRmaBufSlicelite.GetAddr()),
            // [中文导读] [AllReduce逐行 S740] 为保存Doorbell关联归约信息供缓存重放观测；读取缓冲区起始地址；读取缓冲区字节长度补入`locRmaBufSlicelite.GetSize(), reduceIn, dbSqeProfInfo)`（本地RMA切片的GetSize字段、底层归约类型/操作描述、Doorbell任务的缓存观测描述）；本行是参数/结构化初始化续行。
            locRmaBufSlicelite.GetSize(), reduceIn, dbSqeProfInfo);
    // [中文导读] [AllReduce逐行 S741] 结束`if (needCacheTask && isReportTask)`（是否记录本轮任务缓存、是否生成任务观测记录）分支/循环；控制流返回外层。
    }
    // [中文导读] [AllReduce逐行 S742] 在Doorbell之前保存/打印已展开WQE及DbSqe位置，然后关闭跟踪；传入/处理承载任务的执行流、用于WQE缓存/调试的UB连接对象、是否记录本轮任务缓存、本地待提交SQE条数、是否生成任务观测记录、Doorbell任务的缓存观测描述。
    PostLaunchWqe(stream, ubConnLitePtr, needCacheTask, pendingSqeCnt, isReportTask, dbSqeProfInfo);

    // [中文导读] [AllReduce逐行 S744] 给当前执行流生成携带UB PI的jetty Doorbell SQE；取得UB jetty的die/function/jetty标识用于Doorbell；传入/处理承载任务的执行流、所选RMA连接对象的GetUbJettyLiteId字段、WQE构造后的UB PI等输出的pi字段。
    BuildUbDbSendTask(stream, conn->GetUbJettyLiteId(), connOut.pi);

    // [中文导读] [AllReduce逐行 S746] 登记UB归约数据传输的类型/操作/地址/长度。
    ReduceProfilingProcess(
        // [中文导读] [AllReduce逐行 S747] 为登记UB归约数据传输的类型/操作/地址/长度；读取缓冲区起始地址补入`ReinterpretAs<void*>(locRmaBufSlicelite.GetAddr()), ReinterpretAs<void*>(rmtRmaBufSlicelite.GetAddr()),`（本地RMA切片的GetAddr字段、远端RMA切片的GetAddr字段）；本行是参数/结构化初始化续行。
        ReinterpretAs<void*>(locRmaBufSlicelite.GetAddr()), ReinterpretAs<void*>(rmtRmaBufSlicelite.GetAddr()),
        // [中文导读] [AllReduce逐行 S748] 为登记UB归约数据传输的类型/操作/地址/长度；读取缓冲区起始地址；读取缓冲区字节长度补入`locRmaBufSlicelite.GetSize(), reduceIn, stream, taskId)`（本地RMA切片的GetSize字段、底层归约类型/操作描述、承载任务的执行流、当前任务编号）；本行是参数/结构化初始化续行。
        locRmaBufSlicelite.GetSize(), reduceIn, stream, taskId);
// [中文导读] [AllReduce逐行 S749] 结束UbTransportLiteImpl::ReadReduce函数体；控制流返回外层。
}

void UbTransportLiteImpl::WriteReduce(
    const RmaBufferLite& loc, const Buffer& rmt, const ReduceIn& reduceIn, const StreamLite& stream)
{
    SqeConfigLite cfg;
    SetFenceConfig(cfg);

    auto taskId = stream.GetRtsq()->GetTaskId();

    // 当前使用1个connection，下标为0
    RmaConnLite* conn = connVec[0];

    // 展开下发WQE前, 按需设置cache context
    UbConnLite* ubConnLitePtr = nullptr;
    bool needCacheTask = false;
    PreLaunchWqe(ubConnLitePtr, needCacheTask, conn);
    // 下发DbSqe前, 备份相关信息
    const uint32_t pendingSqeCnt = needCacheTask ? stream.GetRtsq()->GetPendingSqeCnt() : 0;

    // 展开下发WQE
    auto locRmaBufSlicelite = GetRmaBufSlicelite(loc);
    auto rmtRmaBufSlicelite = GetRmtRmaBufSliceLite(rmt);
    conn->WriteReduce(
        reduceIn.dataType, reduceIn.reduceOp, locRmaBufSlicelite, stream, rmtRmaBufSlicelite, cfg, connOut);

    // 展开下发WQE后, 展开下发DbSqe前, 按需缓存wqe及DbSqeIdx
    // 注意: pendingSqeCnt在下发DbSqe前已备份
    // 注意: 一定要在展开下发DbSqe前调用PostLaunchWqe, 否则如果下发DbSqe触发LaunchTask,
    // 而尚未调用PostLaunchWqe插入当前WQE数组，
    //     会导致AicpuTaskCache找不到当前WQE数组, 无法正确更新对应的DbSqeLocation
    const bool isReportTask = IsReportTask();
    DbSqeProfInfo dbSqeProfInfo;
    if (needCacheTask && isReportTask) {
        BuildDbSqeProfInfoForReduceProfilingProcess(
            ReinterpretAs<void*>(locRmaBufSlicelite.GetAddr()), ReinterpretAs<void*>(rmtRmaBufSlicelite.GetAddr()),
            locRmaBufSlicelite.GetSize(), reduceIn, dbSqeProfInfo);
    }
    PostLaunchWqe(stream, ubConnLitePtr, needCacheTask, pendingSqeCnt, isReportTask, dbSqeProfInfo);

    BuildUbDbSendTask(stream, conn->GetUbJettyLiteId(), connOut.pi);

    ReduceProfilingProcess(
        ReinterpretAs<void*>(locRmaBufSlicelite.GetAddr()), ReinterpretAs<void*>(rmtRmaBufSlicelite.GetAddr()),
        locRmaBufSlicelite.GetSize(), reduceIn, stream, taskId);
}

void UbTransportLiteImpl::ExecProfiling(
    const RmaBufferLite& loc, const Buffer& rmt, const u64 totalSize,
    const BaseTransportLiteImpl::TransferOp& transferOp, const StreamLite& stream, u32 taskId)
{
    if (transferOp.reduceIn.reduceOp == ReduceOp::INVALID) {
        DmaOp dmaOp = DmaOp::HCCL_DMA_WRITE;
        if (transferOp.transType == TransferType::READ) {
            dmaOp = DmaOp::HCCL_DMA_READ;
        }
        ProfilingProcess(
            ReinterpretAs<void*>(GetRmaBufSlicelite(loc).GetAddr()),
            ReinterpretAs<void*>(GetRmtRmaBufSliceLite(rmt).GetAddr()), totalSize, stream, dmaOp, taskId);
    } else {
        ReduceProfilingProcess(
            ReinterpretAs<void*>(GetRmaBufSlicelite(loc).GetAddr()),
            ReinterpretAs<void*>(GetRmtRmaBufSliceLite(rmt).GetAddr()), totalSize, transferOp.reduceIn, stream, taskId);
    }
}

void UbTransportLiteImpl::ExecProfilingAll(
    const RmaBufferLite& loc, const Buffer& rmt, const u64 totalSize,
    const BaseTransportLiteImpl::TransferOp& transferOp, const StreamLite& stream, u32 taskId, const uint32_t notifyIdx)
{
    if (transferOp.transType == TransferType::READ) {
        ProfilingProcess(
            ReinterpretAs<void*>(GetRmaBufSlicelite(loc).GetAddr()),
            ReinterpretAs<void*>(GetRmtRmaBufSliceLite(rmt).GetAddr()), totalSize, stream, DmaOp::HCCL_DMA_READ,
            taskId);
    } else if (transferOp.transType == TransferType::WRITE) {
        ProfilingProcess(
            ReinterpretAs<void*>(GetRmaBufSlicelite(loc).GetAddr()),
            ReinterpretAs<void*>(GetRmtRmaBufSliceLite(rmt).GetAddr()), totalSize, stream, DmaOp::HCCL_DMA_WRITE,
            taskId);
    } else if (transferOp.transType == TransferType::READ_REDUCE) {
        ReduceProfilingProcess(
            ReinterpretAs<void*>(GetRmaBufSlicelite(loc).GetAddr()),
            ReinterpretAs<void*>(GetRmtRmaBufSliceLite(rmt).GetAddr()), totalSize, transferOp.reduceIn, stream, taskId);
    } else if (transferOp.transType == TransferType::WRITE_REDUCE) {
        ReduceProfilingProcess(
            ReinterpretAs<void*>(GetRmaBufSlicelite(loc).GetAddr()),
            ReinterpretAs<void*>(GetRmtRmaBufSliceLite(rmt).GetAddr()), totalSize, transferOp.reduceIn, stream, taskId);
    } else if (transferOp.transType == TransferType::WRITE_WITH_NOTIFY) {
        WriteWithNotifyProfilingProcess(
            ReinterpretAs<void*>(GetRmaBufSlicelite(loc).GetAddr()),
            ReinterpretAs<void*>(GetRmtRmaBufSliceLite(rmt).GetAddr()), totalSize, stream, taskId,
            GetRmtNotifySliceLite(notifyIdx).GetNotifyId());
    } else if (transferOp.transType == TransferType::WRITE_REDUCE_WITH_NOTIFY) {
        WriteReduceWithNotifyProfilingProcess(
            ReinterpretAs<void*>(GetRmaBufSlicelite(loc).GetAddr()),
            ReinterpretAs<void*>(GetRmtRmaBufSliceLite(rmt).GetAddr()), totalSize, transferOp.reduceIn, stream, taskId,
            GetRmtNotifySliceLite(notifyIdx).GetNotifyId());
    } else if (transferOp.transType == TransferType::NOTIFY_RECORD) {
        NotifyRecordProfilingProcess(
            ReinterpretAs<void*>(GetRmtNotifySliceLite(notifyIdx).GetAddr()),
            GetRmtNotifySliceLite(notifyIdx).GetSize(), stream, taskId, GetRmtNotifySliceLite(notifyIdx).GetNotifyId());
    }
}

// [中文导读] [AllReduce逐行 S854] UbTransportLiteImpl::BatchTransfer的接口声明：传统批数据接口按 WRITE/WRITE_REDUCE/READ/READ_REDUCE 构造 WQE，仅末项 CQE；这些参数属于本函数调用边界。
void UbTransportLiteImpl::BatchTransfer(
    // [中文导读] [AllReduce逐行 S855] UbTransportLiteImpl::BatchTransfer的接口声明：本端缓冲区/地址、远端缓冲区/地址；这些参数属于本函数调用边界。
    const std::vector<RmaBufferLite>& loc, const std::vector<Buffer>& rmt,
    // [中文导读] [AllReduce逐行 S856] UbTransportLiteImpl::BatchTransfer的接口声明：批操作种类/归约描述数组、承载任务的执行流；这些参数属于本函数调用边界。
    const std::vector<BaseTransportLiteImpl::TransferOp>& transferOp, const StreamLite& stream)
// [中文导读] [AllReduce逐行 S857] 进入UbTransportLiteImpl::BatchTransfer函数体：传统批数据接口按 WRITE/WRITE_REDUCE/READ/READ_REDUCE 构造 WQE，仅末项 CQE。
{
    // [中文导读] [AllReduce逐行 S858] 仅当`(UNLIKELY(loc.empty()))`（本端缓冲区/地址的empty字段）成立时进入此分支；检查容器是否没有登记项。
    if (UNLIKELY(loc.empty())) {
        // [中文导读] [AllReduce逐行 S859] 结束本次void调用，当前路径不再继续下发后续操作。
        return;
    // [中文导读] [AllReduce逐行 S860] 结束`if (UNLIKELY(loc.empty()))`（本端缓冲区/地址的empty字段）分支/循环；控制流返回外层。
    }
    // [中文导读] [AllReduce逐行 S861] 准备UB WQE保序/完成配置的局部存储/结构描述，初始化方式以本行声明为准。
    SqeConfigLite cfg;
    // [中文导读] [AllReduce逐行 S862] 把当前传输Fence状态应用到本次WQE配置；传入/处理UB WQE保序/完成配置。
    SetFenceConfig(cfg);

    // [中文导读] [AllReduce逐行 S864] 设置当前任务编号为/按`stream.GetRtsq()->GetTaskId()`（承载任务的执行流的GetRtsq字段）；返回当前StreamLite持有的具体执行队列；读取当前队列taskId，用于该操作与观测信息关联。
    auto taskId = stream.GetRtsq()->GetTaskId();

    // 当前使用1个connection，下标为0 (当前只有一个connection，对应一个jetty)
    // [中文导读] [AllReduce逐行 S867] 设置所选RMA连接对象为/按`connVec[0]`（当前transport的RMA连接数组）。
    RmaConnLite* conn = connVec[0];

    // 展开下发WQE前, 按需设置cache context
    // [中文导读] [AllReduce逐行 S870] 设置用于WQE缓存/调试的UB连接对象为/按`nullptr`。
    UbConnLite* ubConnLitePtr = nullptr;
    // [中文导读] [AllReduce逐行 S871] 设置是否记录本轮任务缓存为/按`false`。
    bool needCacheTask = false;
    // [中文导读] [AllReduce逐行 S872] 按缓存/日志配置开启WQE跟踪并取得具体UB连接；传入/处理用于WQE缓存/调试的UB连接对象、是否记录本轮任务缓存、所选RMA连接对象。
    PreLaunchWqe(ubConnLitePtr, needCacheTask, conn);
    // 下发DbSqe前, 备份相关信息
    // [中文导读] [AllReduce逐行 S874] 设置本地待提交SQE条数为/按`needCacheTask ? stream.GetRtsq()->GetPendingSqeCnt() : 0`（是否记录本轮任务缓存、承载任务的执行流的GetRtsq字段）；返回当前StreamLite持有的具体执行队列；读取Doorbell生成之前当前执行队列的待提交SQE数。
    const uint32_t pendingSqeCnt = needCacheTask ? stream.GetRtsq()->GetPendingSqeCnt() : 0;

    // [中文导读] [AllReduce逐行 S876] 设置批项数量为/按`loc.size()`（本端缓冲区/地址的size字段）；读取容器登记项数。
    u32 insNum = loc.size();
    // [中文导读] [AllReduce逐行 S877] 按`(u32 i = 0; i < insNum; i++)`（本批条目下标、批项数量）遍历本批条目/分片；各次处理保持数组对应关系。
    for (u32 i = 0; i < insNum; i++) {
        // [中文导读] [AllReduce逐行 S878] 设置UB WQE保序/完成配置的cqeEn字段为/按`(i == insNum - 1) ? true : false`（本批条目下标、批项数量）。
        cfg.cqeEn = (i == insNum - 1) ? true : false; // 返回最后一个sqe的cqe
        // [中文导读] [AllReduce逐行 S879] 设置UB WQE保序/完成配置的placeOdr字段为/按`UB_RELAX_ORDER`。
        cfg.placeOdr = UB_RELAX_ORDER;
        // [中文导读] [AllReduce逐行 S880] 设置UB WQE保序/完成配置的compOrder字段为/按`UB_NO_COMPLETION`。
        cfg.compOrder = UB_NO_COMPLETION;
        // [中文导读] [AllReduce逐行 S881] 设置UB WQE保序/完成配置的userConfig字段为/按`true`。
        cfg.userConfig = true;

        // [中文导读] [AllReduce逐行 S883] 设置当前批项本端切片为/按`GetRmaBufSlicelite(loc[i])`（本端缓冲区/地址、本批条目下标）；将本端RMA描述改写为UB本地slice，rkey置0。
        auto localBuffer = GetRmaBufSlicelite(loc[i]);
        // [中文导读] [AllReduce逐行 S884] 设置当前批项远端切片为/按`GetRmtRmaBufSliceLite(rmt[i])`（远端缓冲区/地址、本批条目下标）；从远端已交换注册区解析目标地址/长度/token。
        auto remoteBuffer = GetRmtRmaBufSliceLite(rmt[i]);
        // [中文导读] [AllReduce逐行 S885] 仅当`(transferOp[i].transType == TransferType::WRITE)`（批操作种类/归约描述数组、本批条目下标）成立时进入此分支。
        if (transferOp[i].transType == TransferType::WRITE) {
            // [中文导读] [AllReduce逐行 S886] 向连接远端目标生成WRITE WQE，本端为源；传入/处理所选RMA连接对象的Write字段、当前批项本端切片、当前批项远端切片、UB WQE保序/完成配置、承载任务的执行流、WQE构造后的UB PI等输出。
            conn->Write(localBuffer, remoteBuffer, cfg, stream, connOut); // 当前只有一个connection，对应一个jetty
        // [中文导读] [AllReduce逐行 S887] 仅当`(transferOp[i].transType == TransferType::WRITE_REDUCE)`（批操作种类/归约描述数组、本批条目下标）成立时进入此分支。
        } else if (transferOp[i].transType == TransferType::WRITE_REDUCE) { // write reduce
            // [中文导读] [AllReduce逐行 S888] 构造向远端目标归约的WRITE WQE；传入/处理所选RMA连接对象的WriteReduce字段。
            conn->WriteReduce(
                // [中文导读] [AllReduce逐行 S889] 为构造向远端目标归约的WRITE WQE补入`transferOp[i].reduceIn.dataType, transferOp[i].reduceIn.reduceOp, localBuffer, stream, remoteBuffer,`（批操作种类/归约描述数组、本批条目下标、当前批项本端切片、承载任务的执行流、当前批项远端切片）；本行是参数/结构化初始化续行。
                transferOp[i].reduceIn.dataType, transferOp[i].reduceIn.reduceOp, localBuffer, stream, remoteBuffer,
                // [中文导读] [AllReduce逐行 S890] 为构造向远端目标归约的WRITE WQE补入`cfg, connOut)`（UB WQE保序/完成配置、WQE构造后的UB PI等输出）；本行是参数/结构化初始化续行。
                cfg, connOut);
        // [中文导读] [AllReduce逐行 S891] 仅当`(transferOp[i].transType == TransferType::READ)`（批操作种类/归约描述数组、本批条目下标）成立时进入此分支。
        } else if (transferOp[i].transType == TransferType::READ) {
            // [中文导读] [AllReduce逐行 S892] 生成从远端源读取到本端目标的READ WQE；传入/处理所选RMA连接对象的Read字段、当前批项本端切片、当前批项远端切片、UB WQE保序/完成配置、承载任务的执行流、WQE构造后的UB PI等输出。
            conn->Read(localBuffer, remoteBuffer, cfg, stream, connOut); // 当前只有一个connection，对应一个jetty
        // [中文导读] [AllReduce逐行 S893] 仅当`(transferOp[i].transType == TransferType::READ_REDUCE)`（批操作种类/归约描述数组、本批条目下标）成立时进入此分支。
        } else if (transferOp[i].transType == TransferType::READ_REDUCE) { // read reduce
            // [中文导读] [AllReduce逐行 S894] 生成读取远端源并归约到本地目标的READ WQE；传入/处理所选RMA连接对象的ReadReduce字段、批操作种类/归约描述数组、本批条目下标、当前批项本端切片、当前批项远端切片、承载任务的执行流、UB WQE保序/完成配置、WQE构造后的UB PI等输出。
            conn->ReadReduce(transferOp[i].reduceIn, localBuffer, remoteBuffer, stream, cfg, connOut);
        // [中文导读] [AllReduce逐行 S895] 结束`if (transferOp[i].transType == TransferType::WRITE)`（批操作种类/归约描述数组、本批条目下标）分支/循环；控制流返回外层。
        }
    // [中文导读] [AllReduce逐行 S896] 结束`for (u32 i = 0; i < insNum; i++)`（本批条目下标、批项数量）分支/循环；控制流返回外层。
    }

    // 按需计算totalSize
    // [中文导读] [AllReduce逐行 S899] 设置是否生成任务观测记录为/按`IsReportTask()`；检查是否需要任务异常/性能观测。
    const bool isReportTask = IsReportTask();
    // [中文导读] [AllReduce逐行 S900] 设置累计传输字节数为/按`0`。
    u64 totalSize = 0;
    // [中文导读] [AllReduce逐行 S901] 仅当`(isReportTask)`（是否生成任务观测记录）成立时进入此分支。
    if (isReportTask) {
        // [中文导读] [AllReduce逐行 S902] 按`(u32 i = 0; i < insNum; i++)`（本批条目下标、批项数量）遍历本批条目/分片；各次处理保持数组对应关系。
        for (u32 i = 0; i < insNum; i++) {
            // [中文导读] [AllReduce逐行 S903] 增加累计传输字节数为/按`GetRmaBufSlicelite(loc[i]).GetSize()`（本端缓冲区/地址、本批条目下标）；将本端RMA描述改写为UB本地slice，rkey置0；读取缓冲区字节长度。
            totalSize += GetRmaBufSlicelite(loc[i]).GetSize();
        // [中文导读] [AllReduce逐行 S904] 结束`for (u32 i = 0; i < insNum; i++)`（本批条目下标、批项数量）分支/循环；控制流返回外层。
        }
    // [中文导读] [AllReduce逐行 S905] 结束`if (isReportTask)`（是否生成任务观测记录）分支/循环；控制流返回外层。
    }

    // 展开下发WQE后, 展开下发DbSqe前, 按需缓存wqe及DbSqeIdx
    // 注意: pendingSqeCnt在下发DbSqe前已备份
    // 注意: 一定要在展开下发DbSqe前调用PostLaunchWqe, 否则如果下发DbSqe触发LaunchTask,
    // 而尚未调用PostLaunchWqe插入当前WQE数组，
    //     会导致AicpuTaskCache找不到当前WQE数组, 无法正确更新对应的DbSqeLocation
    // [中文导读] [AllReduce逐行 S912] 准备Doorbell任务的缓存观测描述的局部存储/结构描述，初始化方式以本行声明为准。
    DbSqeProfInfo dbSqeProfInfo;
    // [中文导读] [AllReduce逐行 S913] 仅当`(needCacheTask && isReportTask)`（是否记录本轮任务缓存、是否生成任务观测记录）成立时进入此分支。
    if (needCacheTask && isReportTask) {
        // [中文导读] [AllReduce逐行 S914] 形成传统批数据传输Doorbell缓存观测描述。
        BuildDbSqeProfInfoForExecProfiling(
            // [中文导读] [AllReduce逐行 S915] 为形成传统批数据传输Doorbell缓存观测描述补入`loc[insNum - 1], rmt[insNum - 1], totalSize, transferOp[insNum - 1], dbSqeProfInfo)`（本端缓冲区/地址、批项数量、远端缓冲区/地址、累计传输字节数、批操作种类/归约描述数组、Doorbell任务的缓存观测描述）；本行是参数/结构化初始化续行。
            loc[insNum - 1], rmt[insNum - 1], totalSize, transferOp[insNum - 1], dbSqeProfInfo);
    // [中文导读] [AllReduce逐行 S916] 结束`if (needCacheTask && isReportTask)`（是否记录本轮任务缓存、是否生成任务观测记录）分支/循环；控制流返回外层。
    }
    // [中文导读] [AllReduce逐行 S917] 在Doorbell之前保存/打印已展开WQE及DbSqe位置，然后关闭跟踪；传入/处理承载任务的执行流、用于WQE缓存/调试的UB连接对象、是否记录本轮任务缓存、本地待提交SQE条数、是否生成任务观测记录、Doorbell任务的缓存观测描述。
    PostLaunchWqe(stream, ubConnLitePtr, needCacheTask, pendingSqeCnt, isReportTask, dbSqeProfInfo);

    // [中文导读] [AllReduce逐行 S919] 给当前执行流生成携带UB PI的jetty Doorbell SQE；取得UB jetty的die/function/jetty标识用于Doorbell；传入/处理承载任务的执行流、所选RMA连接对象的GetUbJettyLiteId字段、WQE构造后的UB PI等输出的pi字段。
    BuildUbDbSendTask(stream, conn->GetUbJettyLiteId(), connOut.pi);

    // [中文导读] [AllReduce逐行 S921] 登记传统批数据传输末项及累计字节；传入/处理本端缓冲区/地址、批项数量、远端缓冲区/地址、累计传输字节数、批操作种类/归约描述数组、承载任务的执行流、当前任务编号。
    ExecProfiling(loc[insNum - 1], rmt[insNum - 1], totalSize, transferOp[insNum - 1], stream, taskId);
// [中文导读] [AllReduce逐行 S922] 结束UbTransportLiteImpl::BatchTransfer函数体；控制流返回外层。
}

// Convert hccl::HcommDataType => Hccl::DataType, hccl::HcommReduceOp => Hccl::ReduceOp
static const std::unordered_map<HcommReduceOp, Hccl::ReduceOp> mapHcommReduceOpA5
    = {{HcommReduceOp::HCOMM_REDUCE_SUM, Hccl::ReduceOp::SUM},
       {HcommReduceOp::HCOMM_REDUCE_PROD, Hccl::ReduceOp::PROD},
       {HcommReduceOp::HCOMM_REDUCE_MAX, Hccl::ReduceOp::MAX},
       {HcommReduceOp::HCOMM_REDUCE_MIN, Hccl::ReduceOp::MIN},
       {HcommReduceOp::HCOMM_REDUCE_RESERVED, Hccl::ReduceOp::INVALID}};

static const std::unordered_map<HcommDataType, Hccl::DataType> mapHcommDataTypeA5 = {
#ifndef OPEN_BUILD_PROJECT
    {HcommDataType::HCOMM_DATA_TYPE_HIF8, Hccl::DataType::HIF8},
    {HcommDataType::HCOMM_DATA_TYPE_FP8E4M3, Hccl::DataType::FP8E4M3},
    {HcommDataType::HCOMM_DATA_TYPE_FP8E5M2, Hccl::DataType::FP8E5M2},
    {HcommDataType::HCOMM_DATA_TYPE_FP8E8M0, Hccl::DataType::FP8E8M0},
#endif
    {HcommDataType::HCOMM_DATA_TYPE_INT8, Hccl::DataType::INT8},
    {HcommDataType::HCOMM_DATA_TYPE_INT16, Hccl::DataType::INT16},
    {HcommDataType::HCOMM_DATA_TYPE_INT32, Hccl::DataType::INT32},
    {HcommDataType::HCOMM_DATA_TYPE_INT64, Hccl::DataType::INT64},
    {HcommDataType::HCOMM_DATA_TYPE_INT128, Hccl::DataType::INT128},
    {HcommDataType::HCOMM_DATA_TYPE_UINT8, Hccl::DataType::UINT8},
    {HcommDataType::HCOMM_DATA_TYPE_UINT16, Hccl::DataType::UINT16},
    {HcommDataType::HCOMM_DATA_TYPE_UINT32, Hccl::DataType::UINT32},
    {HcommDataType::HCOMM_DATA_TYPE_UINT64, Hccl::DataType::UINT64},
    {HcommDataType::HCOMM_DATA_TYPE_FP16, Hccl::DataType::FP16},
    {HcommDataType::HCOMM_DATA_TYPE_FP32, Hccl::DataType::FP32},
    {HcommDataType::HCOMM_DATA_TYPE_FP64, Hccl::DataType::FP64},
    {HcommDataType::HCOMM_DATA_TYPE_BFP16, Hccl::DataType::BFP16},
    {HcommDataType::HCOMM_DATA_TYPE_RESERVED, Hccl::DataType::INVALID}};

static HcclResult CheckReduceHcommDataTypeAndHcommReduceOp(HcommDataType dataType, HcommReduceOp reduceOp)
{
    auto dataTypeIt = mapHcommDataTypeA5.find(dataType); // reduce类型，dataType不能是RESERVED
    if (dataTypeIt == mapHcommDataTypeA5.end() || dataTypeIt->first == HcommDataType::HCOMM_DATA_TYPE_RESERVED) {
        HCCL_ERROR("[%s] type[%u] is not supported.", __func__, dataType);
        return HCCL_E_PARA;
    }

    auto reduceOpIt = mapHcommReduceOpA5.find(reduceOp); // reduce类型，reduceOp不能是RESERVED
    if (reduceOpIt == mapHcommReduceOpA5.end() || reduceOpIt->first == HcommReduceOp::HCOMM_REDUCE_RESERVED) {
        HCCL_ERROR("[%s] op[%u] is not supported.", __func__, reduceOp);
        return HCCL_E_PARA;
    }

    return HCCL_SUCCESS;
}

constexpr u32 SIZE_TABLE[HCCL_DATA_TYPE_RESERVED]
    = {sizeof(s8),
       sizeof(s16),
       sizeof(s32),
       2,
       sizeof(float),
       sizeof(s64),
       sizeof(u64),
       sizeof(u8),
       sizeof(u16),
       sizeof(u32),
       8,
       2,
       16,
       2,
       1,
       1,
       1,
       1};

// [中文导读] [AllReduce逐行 S991] ParasReduceData的接口声明：取得批归约元素数、类型和操作并检查支持映射；这些参数属于本函数调用边界。
static HcclResult ParasReduceData(
    // [中文导读] [AllReduce逐行 S992] ParasReduceData的接口声明：当前公开批传输描述、归约元素数出参（之后由ParseData换算字节）、元素数据类型、归约操作；这些参数属于本函数调用边界。
    const HcommBatchTransferDesc& transferDesc, uint64_t& len, HcommDataType& dataType, HcommReduceOp& reduceOp)
// [中文导读] [AllReduce逐行 S993] 进入ParasReduceData函数体：取得批归约元素数、类型和操作并检查支持映射。
{
    // [中文导读] [AllReduce逐行 S994] 设置归约元素数出参（之后由ParseData换算字节）为/按`transferDesc.transferInfo.reduce.count`（当前公开批传输描述的transferInfo.reduce.count字段）。
    len = transferDesc.transferInfo.reduce.count;
    // [中文导读] [AllReduce逐行 S995] 设置元素数据类型为/按`transferDesc.transferInfo.reduce.dataType`（当前公开批传输描述的transferInfo.reduce.dataType字段）。
    dataType = transferDesc.transferInfo.reduce.dataType;
    // [中文导读] [AllReduce逐行 S996] 设置归约操作为/按`transferDesc.transferInfo.reduce.reduceOp`（当前公开批传输描述的transferInfo.reduce.reduceOp字段）。
    reduceOp = transferDesc.transferInfo.reduce.reduceOp;
    // [中文导读] [AllReduce逐行 S997] 设置当前调用状态为/按`CheckReduceHcommDataTypeAndHcommReduceOp(dataType, reduceOp)`（元素数据类型、归约操作）；批归约拒绝缺失映射与RESERVED类型/操作。
    auto ret = CheckReduceHcommDataTypeAndHcommReduceOp(dataType, reduceOp);
    // [中文导读] [AllReduce逐行 S998] 开始带日志的条件错误处理：后续实参提供触发条件、诊断与返回状态；命中条件才提前返回。
    CHK_PRT_RET(
        // [中文导读] [AllReduce逐行 S999] 向条件错误检查提供`ret != HCCL_SUCCESS,`（当前调用状态），用于确定触发条件或形成对应诊断。
        ret != HCCL_SUCCESS,
        // [中文导读] [AllReduce逐行 S1000] 记录ParasReduceData的错误诊断，字段包含元素数据类型、归约操作；日志本身不执行传输。
        HCCL_ERROR("FAIL at CheckReduceHcommDataTypeAndHcommReduceOp dataType[%d], reduceOp[%d].", dataType, reduceOp),
        // [中文导读] [AllReduce逐行 S1001] 指定条件命中时要返回的错误状态`ret)`（当前调用状态），未命中则继续原处理路径。
        ret);
    // [中文导读] [AllReduce逐行 S1002] 当前路径返回成功状态；仅说明本函数处理/任务组织成功，完成语义由其具体调用职责决定。
    return HCCL_SUCCESS;
// [中文导读] [AllReduce逐行 S1003] 结束ParasReduceData函数体；控制流返回外层。
}

// [中文导读] [AllReduce逐行 S1005] ParseData的接口声明：按批传输种类解析本端/远端地址、字节数或归约元素数与通知槽，归约最终换算字节；这些参数属于本函数调用边界。
static HcclResult ParseData(
    // [中文导读] [AllReduce逐行 S1006] ParseData的接口声明：当前公开批传输描述、远端缓冲区/地址、本端缓冲区/地址、本次字节长度、底层传输种类；这些参数属于本函数调用边界。
    const HcommBatchTransferDesc& transferDesc, void*& rmt, void*& loc, uint64_t& len, Hccl::TransferType& tfType,
    // [中文导读] [AllReduce逐行 S1007] ParseData的接口声明：元素数据类型、归约操作、对端通知槽索引；这些参数属于本函数调用边界。
    HcommDataType& dataType, HcommReduceOp& reduceOp, uint32_t& notifyIdx)
// [中文导读] [AllReduce逐行 S1008] 进入ParseData函数体：按批传输种类解析本端/远端地址、字节数或归约元素数与通知槽，归约最终换算字节。
{
    // [中文导读] [AllReduce逐行 S1009] 仅当`(transferDesc.transType == HCOMM_TRANSFER_TYPE_WRITE)`（当前公开批传输描述的transType字段）成立时进入此分支。
    if (transferDesc.transType == HCOMM_TRANSFER_TYPE_WRITE) {
        // [中文导读] [AllReduce逐行 S1010] 设置远端缓冲区/地址为/按`transferDesc.transferInfo.write.dst`（当前公开批传输描述的transferInfo.write.dst字段）。
        rmt = transferDesc.transferInfo.write.dst; // write操作，dst是远端地址
        // [中文导读] [AllReduce逐行 S1011] 设置本端缓冲区/地址为/按`transferDesc.transferInfo.write.src`（当前公开批传输描述的transferInfo.write.src字段）。
        loc = transferDesc.transferInfo.write.src; // src是本端地址
        // [中文导读] [AllReduce逐行 S1012] 设置本次字节长度为/按`transferDesc.transferInfo.write.len`（当前公开批传输描述的transferInfo.write.len字段）。
        len = transferDesc.transferInfo.write.len;
        // [中文导读] [AllReduce逐行 S1013] 设置底层传输种类为/按`Hccl::TransferType::WRITE`。
        tfType = Hccl::TransferType::WRITE;
    // [中文导读] [AllReduce逐行 S1014] 仅当`(transferDesc.transType == HCOMM_TRANSFER_TYPE_READ)`（当前公开批传输描述的transType字段）成立时进入此分支。
    } else if (transferDesc.transType == HCOMM_TRANSFER_TYPE_READ) {
        // [中文导读] [AllReduce逐行 S1015] 设置远端缓冲区/地址为/按`transferDesc.transferInfo.read.src`（当前公开批传输描述的transferInfo.read.src字段）。
        rmt = transferDesc.transferInfo.read.src; // read操作，src是远端地址
        // [中文导读] [AllReduce逐行 S1016] 设置本端缓冲区/地址为/按`transferDesc.transferInfo.read.dst`（当前公开批传输描述的transferInfo.read.dst字段）。
        loc = transferDesc.transferInfo.read.dst; // dst是本端地址
        // [中文导读] [AllReduce逐行 S1017] 设置本次字节长度为/按`transferDesc.transferInfo.read.len`（当前公开批传输描述的transferInfo.read.len字段）。
        len = transferDesc.transferInfo.read.len;
        // [中文导读] [AllReduce逐行 S1018] 设置底层传输种类为/按`Hccl::TransferType::READ`。
        tfType = Hccl::TransferType::READ;
    // [中文导读] [AllReduce逐行 S1019] 仅当`(transferDesc.transType == HCOMM_TRANSFER_TYPE_WRITE_REDUCE)`（当前公开批传输描述的transType字段）成立时进入此分支。
    } else if (transferDesc.transType == HCOMM_TRANSFER_TYPE_WRITE_REDUCE) {
        // [中文导读] [AllReduce逐行 S1020] 设置远端缓冲区/地址为/按`transferDesc.transferInfo.reduce.dst`（当前公开批传输描述的transferInfo.reduce.dst字段）。
        rmt = transferDesc.transferInfo.reduce.dst;
        // [中文导读] [AllReduce逐行 S1021] 设置本端缓冲区/地址为/按`transferDesc.transferInfo.reduce.src`（当前公开批传输描述的transferInfo.reduce.src字段）。
        loc = transferDesc.transferInfo.reduce.src;
        // [中文导读] [AllReduce逐行 S1022] 设置底层传输种类为/按`Hccl::TransferType::WRITE_REDUCE`。
        tfType = Hccl::TransferType::WRITE_REDUCE;
        // [中文导读] [AllReduce逐行 S1023] 读取批归约count/type/op并检查支持组合；返回非成功时由检查宏立即向上传递。
        CHK_RET(ParasReduceData(transferDesc, len, dataType, reduceOp));
    // [中文导读] [AllReduce逐行 S1024] 仅当`(transferDesc.transType == HCOMM_TRANSFER_TYPE_READ_REDUCE)`（当前公开批传输描述的transType字段）成立时进入此分支。
    } else if (transferDesc.transType == HCOMM_TRANSFER_TYPE_READ_REDUCE) {
        // [中文导读] [AllReduce逐行 S1025] 设置远端缓冲区/地址为/按`transferDesc.transferInfo.reduce.src`（当前公开批传输描述的transferInfo.reduce.src字段）。
        rmt = transferDesc.transferInfo.reduce.src;
        // [中文导读] [AllReduce逐行 S1026] 设置本端缓冲区/地址为/按`transferDesc.transferInfo.reduce.dst`（当前公开批传输描述的transferInfo.reduce.dst字段）。
        loc = transferDesc.transferInfo.reduce.dst;
        // [中文导读] [AllReduce逐行 S1027] 设置底层传输种类为/按`Hccl::TransferType::READ_REDUCE`。
        tfType = Hccl::TransferType::READ_REDUCE;
        // [中文导读] [AllReduce逐行 S1028] 读取批归约count/type/op并检查支持组合；返回非成功时由检查宏立即向上传递。
        CHK_RET(ParasReduceData(transferDesc, len, dataType, reduceOp));
    // [中文导读] [AllReduce逐行 S1029] 仅当`(transferDesc.transType == HCOMM_TRANSFER_TYPE_WRITE_WITH_NOTIFY)`（当前公开批传输描述的transType字段）成立时进入此分支。
    } else if (transferDesc.transType == HCOMM_TRANSFER_TYPE_WRITE_WITH_NOTIFY) {
        // [中文导读] [AllReduce逐行 S1030] 设置远端缓冲区/地址为/按`transferDesc.transferInfo.writeWithNotify.dst`（当前公开批传输描述的transferInfo.writeWithNotify.dst字段）。
        rmt = transferDesc.transferInfo.writeWithNotify.dst; // write操作，dst是远端地址
        // [中文导读] [AllReduce逐行 S1031] 设置本端缓冲区/地址为/按`transferDesc.transferInfo.writeWithNotify.src`（当前公开批传输描述的transferInfo.writeWithNotify.src字段）。
        loc = transferDesc.transferInfo.writeWithNotify.src; // src是本端地址
        // [中文导读] [AllReduce逐行 S1032] 设置本次字节长度为/按`transferDesc.transferInfo.writeWithNotify.len`（当前公开批传输描述的transferInfo.writeWithNotify.len字段）。
        len = transferDesc.transferInfo.writeWithNotify.len;
        // [中文导读] [AllReduce逐行 S1033] 设置对端通知槽索引为/按`transferDesc.transferInfo.writeWithNotify.notifyIdx`（当前公开批传输描述的transferInfo.writeWithNotify.notifyIdx字段）。
        notifyIdx = transferDesc.transferInfo.writeWithNotify.notifyIdx;
        // [中文导读] [AllReduce逐行 S1034] 设置底层传输种类为/按`Hccl::TransferType::WRITE_WITH_NOTIFY`。
        tfType = Hccl::TransferType::WRITE_WITH_NOTIFY;
    // [中文导读] [AllReduce逐行 S1035] 仅当`(transferDesc.transType == HCOMM_TRANSFER_TYPE_WRITE_REDUCE_WITH_NOTIFY)`（当前公开批传输描述的transType字段）成立时进入此分支。
    } else if (transferDesc.transType == HCOMM_TRANSFER_TYPE_WRITE_REDUCE_WITH_NOTIFY) {
        // [中文导读] [AllReduce逐行 S1036] 设置远端缓冲区/地址为/按`transferDesc.transferInfo.writeReduceWithNotify.dst`（当前公开批传输描述的transferInfo.writeReduceWithNotify.dst字段）。
        rmt = transferDesc.transferInfo.writeReduceWithNotify.dst;
        // [中文导读] [AllReduce逐行 S1037] 设置本端缓冲区/地址为/按`transferDesc.transferInfo.writeReduceWithNotify.src`（当前公开批传输描述的transferInfo.writeReduceWithNotify.src字段）。
        loc = transferDesc.transferInfo.writeReduceWithNotify.src;
        // [中文导读] [AllReduce逐行 S1038] 先把writeReduceWithNotify.count写入len；此处仍为归约元素数，后续乘SIZE_TABLE[dataType]才得到字节数。
        len = transferDesc.transferInfo.writeReduceWithNotify.count;
        // [中文导读] [AllReduce逐行 S1039] 设置元素数据类型为/按`transferDesc.transferInfo.writeReduceWithNotify.dataType`（当前公开批传输描述的transferInfo.writeReduceWithNotify.dataType字段）。
        dataType = transferDesc.transferInfo.writeReduceWithNotify.dataType;
        // [中文导读] [AllReduce逐行 S1040] 设置归约操作为/按`transferDesc.transferInfo.writeReduceWithNotify.reduceOp`（当前公开批传输描述的transferInfo.writeReduceWithNotify.reduceOp字段）。
        reduceOp = transferDesc.transferInfo.writeReduceWithNotify.reduceOp;
        // [中文导读] [AllReduce逐行 S1041] 设置对端通知槽索引为/按`transferDesc.transferInfo.writeReduceWithNotify.notifyIdx`（当前公开批传输描述的transferInfo.writeReduceWithNotify.notifyIdx字段）。
        notifyIdx = transferDesc.transferInfo.writeReduceWithNotify.notifyIdx;
        // [中文导读] [AllReduce逐行 S1042] 设置底层传输种类为/按`Hccl::TransferType::WRITE_REDUCE_WITH_NOTIFY`。
        tfType = Hccl::TransferType::WRITE_REDUCE_WITH_NOTIFY;
        // [中文导读] [AllReduce逐行 S1043] 批归约拒绝缺失映射与RESERVED类型/操作；返回非成功时由检查宏立即向上传递。
        CHK_RET(CheckReduceHcommDataTypeAndHcommReduceOp(dataType, reduceOp));
    // [中文导读] [AllReduce逐行 S1044] 仅当`(transferDesc.transType == HCOMM_TRANSFER_TYPE_NOTIFY_RECORD)`（当前公开批传输描述的transType字段）成立时进入此分支。
    } else if (transferDesc.transType == HCOMM_TRANSFER_TYPE_NOTIFY_RECORD) {
        // [中文导读] [AllReduce逐行 S1045] 设置对端通知槽索引为/按`transferDesc.transferInfo.notifyRecord.notifyIdx`（当前公开批传输描述的transferInfo.notifyRecord.notifyIdx字段）。
        notifyIdx = transferDesc.transferInfo.notifyRecord.notifyIdx;
        // [中文导读] [AllReduce逐行 S1046] 设置底层传输种类为/按`Hccl::TransferType::NOTIFY_RECORD`。
        tfType = Hccl::TransferType::NOTIFY_RECORD;
    // [中文导读] [AllReduce逐行 S1047] 进入上一个条件未命中的替代路径，按本函数的资源/设备/协议分流继续处理。
    } else {
        // [中文导读] [AllReduce逐行 S1048] 记录ParseData的错误诊断，字段包含当前公开批传输描述的transType字段；日志本身不执行传输。
        HCCL_ERROR("[%s] unsupported transType[%d]", __func__, transferDesc.transType);
        // [中文导读] [AllReduce逐行 S1049] 返回HCCL_E_NOT_SUPPORT，表示当前引擎/设备/协议不支持此路径；此路径停止本函数的后续处理。
        return HCCL_E_NOT_SUPPORT;
    // [中文导读] [AllReduce逐行 S1050] 结束`if (transferDesc.transType == HCOMM_TRANSFER_TYPE_WRITE)`（当前公开批传输描述的transType字段）分支/循环；控制流返回外层。
    }
    // [中文导读] [AllReduce逐行 S1051] 仅当`(reduceOp != HcommReduceOp::HCOMM_REDUCE_RESERVED)`（归约操作）成立时进入此分支。
    if (reduceOp != HcommReduceOp::HCOMM_REDUCE_RESERVED) { // 对于规约类型, size = count * sizeof(datatype)
        // [中文导读] [AllReduce逐行 S1052] 将归约元素数len乘单元素字节数，得到传给RMA buffer/slice的字节长度；FP32对应4字节。
        len = len * SIZE_TABLE[dataType];
    // [中文导读] [AllReduce逐行 S1053] 结束`if (reduceOp != HcommReduceOp::HCOMM_REDUCE_RESERVED)`（归约操作）分支/循环；控制流返回外层。
    }
    // [中文导读] [AllReduce逐行 S1054] 当前路径返回成功状态；仅说明本函数处理/任务组织成功，完成语义由其具体调用职责决定。
    return HCCL_SUCCESS;
// [中文导读] [AllReduce逐行 S1055] 结束ParseData函数体；控制流返回外层。
}
constexpr uint32_t NOTIFYIDX_INVALID_VALUE = 0xFFFFFFFF; // NOTIFY idex非法值
// [中文导读] [AllReduce逐行 S1057] UbTransportLiteImpl::ExecuteBatchTransfer的接口声明：逐条解析传输/通知描述并组装 slice、操作和通知索引数组；这些参数属于本函数调用边界。
HcclResult UbTransportLiteImpl::ExecuteBatchTransfer(
    // [中文导读] [AllReduce逐行 S1058] UbTransportLiteImpl::ExecuteBatchTransfer的接口声明：设备轻量执行流、公开批传输描述数组、批操作描述条数；这些参数属于本函数调用边界。
    StreamLite* streamLitePtr, const HcommBatchTransferDesc* transferDescs, uint32_t transferDescNum)
// [中文导读] [AllReduce逐行 S1059] 进入UbTransportLiteImpl::ExecuteBatchTransfer函数体：逐条解析传输/通知描述并组装 slice、操作和通知索引数组。
{
    // [中文导读] [AllReduce逐行 S1060] 准备批操作本端RMA描述数组的局部存储/结构描述，初始化方式以本行声明为准。
    std::vector<Hccl::RmaBufferLite> locSlices;
    // [中文导读] [AllReduce逐行 S1061] 准备批操作远端地址范围数组的局部存储/结构描述，初始化方式以本行声明为准。
    std::vector<Hccl::Buffer> rmtSlices;
    // [中文导读] [AllReduce逐行 S1062] 准备批操作种类/归约描述数组的局部存储/结构描述，初始化方式以本行声明为准。
    std::vector<Hccl::BaseTransportLiteImpl::TransferOp> transferOps;
    // [中文导读] [AllReduce逐行 S1063] 准备批操作通知槽索引数组的局部存储/结构描述，初始化方式以本行声明为准。
    std::vector<uint32_t> notifyIdxs;

    // [中文导读] [AllReduce逐行 S1065] 调用reserve，使用批操作本端RMA描述数组的reserve字段、批操作描述条数；传入/处理批操作本端RMA描述数组的reserve字段、批操作描述条数。
    locSlices.reserve(transferDescNum);
    // [中文导读] [AllReduce逐行 S1066] 调用reserve，使用批操作远端地址范围数组的reserve字段、批操作描述条数；传入/处理批操作远端地址范围数组的reserve字段、批操作描述条数。
    rmtSlices.reserve(transferDescNum);
    // [中文导读] [AllReduce逐行 S1067] 调用reserve，使用批操作种类/归约描述数组的reserve字段、批操作描述条数；传入/处理批操作种类/归约描述数组的reserve字段、批操作描述条数。
    transferOps.reserve(transferDescNum);
    // [中文导读] [AllReduce逐行 S1068] 调用reserve，使用批操作通知槽索引数组的reserve字段、批操作描述条数；传入/处理批操作通知槽索引数组的reserve字段、批操作描述条数。
    notifyIdxs.reserve(transferDescNum);

    // [中文导读] [AllReduce逐行 S1070] 按`(uint32_t i = 0; i < transferDescNum; i++)`（本批条目下标、批操作描述条数）遍历本批条目/分片；各次处理保持数组对应关系。
    for (uint32_t i = 0; i < transferDescNum; i++) {
        // [中文导读] [AllReduce逐行 S1071] 准备本端RMA描述的局部存储/结构描述，初始化方式以本行声明为准。
        Hccl::RmaBufferLite locRmaBuf;
        // [中文导读] [AllReduce逐行 S1072] 设置远端缓冲区/地址为/按`nullptr`。
        void* rmt = nullptr;
        // [中文导读] [AllReduce逐行 S1073] 设置本端缓冲区/地址为/按`nullptr`。
        void* loc = nullptr;
        // [中文导读] [AllReduce逐行 S1074] 设置本次字节长度为/按`0`。
        uint64_t len = 0;
        // [中文导读] [AllReduce逐行 S1075] 准备底层传输种类的局部存储/结构描述，初始化方式以本行声明为准。
        Hccl::TransferType tfType;
        // [中文导读] [AllReduce逐行 S1076] 准备元素数据类型的局部存储/结构描述，初始化方式以本行声明为准。
        HcommDataType dataType{HcommDataType::HCOMM_DATA_TYPE_RESERVED};
        // [中文导读] [AllReduce逐行 S1077] 准备归约操作的局部存储/结构描述，初始化方式以本行声明为准。
        HcommReduceOp reduceOp{HcommReduceOp::HCOMM_REDUCE_RESERVED};
        // [中文导读] [AllReduce逐行 S1078] 设置对端通知槽索引为/按`NOTIFYIDX_INVALID_VALUE`。
        uint32_t notifyIdx = NOTIFYIDX_INVALID_VALUE;
        // [中文导读] [AllReduce逐行 S1079] 按批项transType解析本端/远端地址、字节数/元素数和通知索引；返回非成功时由检查宏立即向上传递。
        CHK_RET(ParseData(transferDescs[i], rmt, loc, len, tfType, dataType, reduceOp, notifyIdx));
        // [中文导读] [AllReduce逐行 S1080] 仅当`(tfType != Hccl::TransferType::NOTIFY_RECORD)`（底层传输种类）成立时进入此分支。
        if (tfType != Hccl::TransferType::NOTIFY_RECORD) { // NOTIFY_RECORD时没有地址字段
            // [中文导读] [AllReduce逐行 S1081] 检查`rmt`（远端缓冲区/地址）不是空对象；宏命中失败条件时立即返回对应指针错误。
            CHK_PTR_NULL(rmt);
            // [中文导读] [AllReduce逐行 S1082] 检查`loc`（本端缓冲区/地址）不是空对象；宏命中失败条件时立即返回对应指针错误。
            CHK_PTR_NULL(loc);
            // [中文导读] [AllReduce逐行 S1083] 设置当前调用状态为/按`BuildLocRmaBufferLite(ReinterpretAs<uintptr_t>(loc), len, locRmaBuf)`（本端缓冲区/地址、本次字节长度、本端RMA描述）；为本端地址与字节范围解析已注册RMA token。
            HcclResult ret = BuildLocRmaBufferLite(ReinterpretAs<uintptr_t>(loc), len, locRmaBuf);
            // [中文导读] [AllReduce逐行 S1084] 开始带日志的条件错误处理：后续实参提供触发条件、诊断与返回状态；命中条件才提前返回。
            CHK_PRT_RET(
                // [中文导读] [AllReduce逐行 S1085] 向条件错误检查提供`ret != HCCL_SUCCESS,`（当前调用状态），用于确定触发条件或形成对应诊断。
                ret != HCCL_SUCCESS,
                // [中文导读] [AllReduce逐行 S1086] 记录UbTransportLiteImpl::ExecuteBatchTransfer的错误诊断；日志本身不执行传输。
                HCCL_ERROR(
                    // [中文导读] [AllReduce逐行 S1087] 为当前UbTransportLiteImpl::ExecuteBatchTransfer诊断/异常表达式提供格式文本；这一物理行没有数据搬运副作用。
                    "[%s] FAIL at BuildLocRmaBufferLite for index %u. rmt[%p], loc[%p], len[0x%llx], tfType[%u], "
                    // [中文导读] [AllReduce逐行 S1088] 为当前UbTransportLiteImpl::ExecuteBatchTransfer诊断/异常表达式提供格式文本；这一物理行没有数据搬运副作用。
                    "dataType[%d], reduceOp[%d].",
                    // [中文导读] [AllReduce逐行 S1089] 为前述多行表达式补入`__func__, i, rmt, loc, len, tfType, dataType, reduceOp),`（本批条目下标、远端缓冲区/地址、本端缓冲区/地址、本次字节长度、底层传输种类、元素数据类型、归约操作）；本行是参数/结构化初始化续行。
                    __func__, i, rmt, loc, len, tfType, dataType, reduceOp),
                // [中文导读] [AllReduce逐行 S1090] 指定条件命中时要返回的错误状态`ret)`（当前调用状态），未命中则继续原处理路径。
                ret);
        // [中文导读] [AllReduce逐行 S1091] 结束`if (tfType != Hccl::TransferType::NOTIFY_RECORD)`（底层传输种类）分支/循环；控制流返回外层。
        }
        // [中文导读] [AllReduce逐行 S1092] 仅当`(tfType == Hccl::TransferType::NOTIFY_RECORD || tfType == Hccl::TransferType::WRITE_WITH_NOTIFY`（底层传输种类）成立时进入此分支。
        if (tfType == Hccl::TransferType::NOTIFY_RECORD || tfType == Hccl::TransferType::WRITE_WITH_NOTIFY
            // [中文导读] [AllReduce逐行 S1093] 补全本分支/循环判断的`|| tfType == Hccl::TransferType::WRITE_REDUCE_WITH_NOTIFY)`（底层传输种类），和前面条件共同决定是否进入后续路径。
            || tfType == Hccl::TransferType::WRITE_REDUCE_WITH_NOTIFY) {
            // [中文导读] [AllReduce逐行 S1094] 开始带日志的条件错误处理：后续实参提供触发条件、诊断与返回状态；命中条件才提前返回。
            CHK_PRT_RET(
                // [中文导读] [AllReduce逐行 S1095] 向条件错误检查提供`notifyIdx == NOTIFYIDX_INVALID_VALUE,`（对端通知槽索引），用于确定触发条件或形成对应诊断。
                notifyIdx == NOTIFYIDX_INVALID_VALUE,
                // [中文导读] [AllReduce逐行 S1096] 记录UbTransportLiteImpl::ExecuteBatchTransfer的错误诊断；日志本身不执行传输。
                HCCL_ERROR(
                    // [中文导读] [AllReduce逐行 S1097] 为当前UbTransportLiteImpl::ExecuteBatchTransfer诊断/异常表达式提供格式文本，将报告本批条目下标、底层传输种类、对端通知槽索引；这一物理行没有数据搬运副作用。
                    "[%s] FAIL at ParseData for index %u. tfType[%u], notifyIdx[%u].", __func__, i, tfType, notifyIdx),
                // [中文导读] [AllReduce逐行 S1098] 记录UbTransportLiteImpl::ExecuteBatchTransfer的状态/性能诊断；日志本身不执行传输。
                HCCL_E_PARA);
        // [中文导读] [AllReduce逐行 S1099] 结束当前局部作用域；控制流返回外层。
        }
        // [中文导读] [AllReduce逐行 S1100] 将当前条目追加到对应数组/列表；传入/处理批操作通知槽索引数组的push_back字段、对端通知槽索引。
        notifyIdxs.push_back(notifyIdx);
        // [中文导读] [AllReduce逐行 S1101] 将当前条目追加到对应数组/列表；传入/处理批操作本端RMA描述数组的push_back字段、本端RMA描述。
        locSlices.push_back(locRmaBuf);

        // [中文导读] [AllReduce逐行 S1103] 准备远端缓冲区/地址、本次字节长度的局部存储/结构描述，初始化方式以本行声明为准。
        const Hccl::Buffer rmtBuf{ReinterpretAs<uintptr_t>(rmt), len};
        // [中文导读] [AllReduce逐行 S1104] 将当前条目追加到对应数组/列表；传入/处理批操作远端地址范围数组的push_back字段。
        rmtSlices.push_back(rmtBuf);

        // [中文导读] [AllReduce逐行 S1106] 调用at, at，使用底层归约类型/操作描述、元素数据类型、归约操作；对象涉及底层归约类型/操作描述、元素数据类型、归约操作。
        Hccl::ReduceIn reduceIn{mapHcommDataTypeA5.at(dataType), mapHcommReduceOpA5.at(reduceOp)};

        // [中文导读] [AllReduce逐行 S1108] 将当前条目追加到对应数组/列表；传入/处理批操作种类/归约描述数组的push_back字段、底层传输种类、底层归约类型/操作描述。
        transferOps.push_back(Hccl::BaseTransportLiteImpl::TransferOp{tfType, reduceIn});

        // [中文导读] [AllReduce逐行 S1110] 记录UbTransportLiteImpl::ExecuteBatchTransfer的调试诊断；日志本身不执行传输。
        HCCL_DEBUG(
            // [中文导读] [AllReduce逐行 S1111] 为当前UbTransportLiteImpl::ExecuteBatchTransfer诊断/异常表达式提供格式文本；这一物理行没有数据搬运副作用。
            "[%s] Prepared transfer op for index %u. rmt[%p], loc[%p], len[0x%llx], tfType[%u], dataType[%d], "
            // [中文导读] [AllReduce逐行 S1112] 为当前UbTransportLiteImpl::ExecuteBatchTransfer诊断/异常表达式提供格式文本；这一物理行没有数据搬运副作用。
            "reduceOp[%d].",
            // [中文导读] [AllReduce逐行 S1113] 为前述多行表达式补入`__func__, i, rmt, loc, len, tfType, dataType, reduceOp)`（本批条目下标、远端缓冲区/地址、本端缓冲区/地址、本次字节长度、底层传输种类、元素数据类型、归约操作）；本行是参数/结构化初始化续行。
            __func__, i, rmt, loc, len, tfType, dataType, reduceOp);
    // [中文导读] [AllReduce逐行 S1114] 结束`for (uint32_t i = 0; i < transferDescNum; i++)`（本批条目下标、批操作描述条数）分支/循环；控制流返回外层。
    }
    // [中文导读] [AllReduce逐行 S1115] 在异常捕获边界执行后续表达式；异常按后续处理语句转换成HCCL状态或提前返回。
    EXCEPTION_CATCH(
        // [中文导读] [AllReduce逐行 S1116] 为将一批数据/通知WQE组织好后生成一个Doorbell补入`BatchTransferAll(locSlices, rmtSlices, transferOps, notifyIdxs, *streamLitePtr), return HCCL_E_INTERNAL)`（批操作本端RMA描述数组、批操作远端地址范围数组、批操作种类/归约描述数组、批操作通知槽索引数组、设备轻量执行流）；本行是参数/结构化初始化续行。
        BatchTransferAll(locSlices, rmtSlices, transferOps, notifyIdxs, *streamLitePtr), return HCCL_E_INTERNAL);
    // [中文导读] [AllReduce逐行 S1117] 当前路径返回成功状态；仅说明本函数处理/任务组织成功，完成语义由其具体调用职责决定。
    return HCCL_SUCCESS;
// [中文导读] [AllReduce逐行 S1118] 结束UbTransportLiteImpl::ExecuteBatchTransfer函数体；控制流返回外层。
}

// [中文导读] [AllReduce逐行 S1120] UbTransportLiteImpl::BatchTransferAll的接口声明：完整批接口一次组织全部 WQE，按末项建观测并排入 Doorbell；这些参数属于本函数调用边界。
void UbTransportLiteImpl::BatchTransferAll(
    // [中文导读] [AllReduce逐行 S1121] UbTransportLiteImpl::BatchTransferAll的接口声明：本端缓冲区/地址、远端缓冲区/地址；这些参数属于本函数调用边界。
    const std::vector<RmaBufferLite>& loc, const std::vector<Buffer>& rmt,
    // [中文导读] [AllReduce逐行 S1122] UbTransportLiteImpl::BatchTransferAll的接口声明：批操作种类/归约描述数组、批操作通知槽索引数组；这些参数属于本函数调用边界。
    const std::vector<BaseTransportLiteImpl::TransferOp>& transferOp, const std::vector<uint32_t>& notifyIdxs,
    // [中文导读] [AllReduce逐行 S1123] UbTransportLiteImpl::BatchTransferAll的接口声明：承载任务的执行流；这些参数属于本函数调用边界。
    const StreamLite& stream)
// [中文导读] [AllReduce逐行 S1124] 进入UbTransportLiteImpl::BatchTransferAll函数体：完整批接口一次组织全部 WQE，按末项建观测并排入 Doorbell。
{
    // [中文导读] [AllReduce逐行 S1125] 仅当`(UNLIKELY(loc.empty()))`（本端缓冲区/地址的empty字段）成立时进入此分支；检查容器是否没有登记项。
    if (UNLIKELY(loc.empty())) {
        // [中文导读] [AllReduce逐行 S1126] 结束本次void调用，当前路径不再继续下发后续操作。
        return;
    // [中文导读] [AllReduce逐行 S1127] 结束`if (UNLIKELY(loc.empty()))`（本端缓冲区/地址的empty字段）分支/循环；控制流返回外层。
    }

    // [中文导读] [AllReduce逐行 S1129] 设置当前任务编号为/按`stream.GetRtsq()->GetTaskId()`（承载任务的执行流的GetRtsq字段）；返回当前StreamLite持有的具体执行队列；读取当前队列taskId，用于该操作与观测信息关联。
    auto taskId = stream.GetRtsq()->GetTaskId();

    // 当前使用1个connection，下标为0 (当前只有一个connection，对应一个jetty)
    // [中文导读] [AllReduce逐行 S1132] 设置所选RMA连接对象为/按`connVec[0]`（当前transport的RMA连接数组）。
    RmaConnLite* conn = connVec[0];

    // 展开下发WQE前, 按需设置cache context
    // [中文导读] [AllReduce逐行 S1135] 设置用于WQE缓存/调试的UB连接对象为/按`nullptr`。
    UbConnLite* ubConnLitePtr = nullptr;
    // [中文导读] [AllReduce逐行 S1136] 设置是否记录本轮任务缓存为/按`false`。
    bool needCacheTask = false;
    // [中文导读] [AllReduce逐行 S1137] 按缓存/日志配置开启WQE跟踪并取得具体UB连接；传入/处理用于WQE缓存/调试的UB连接对象、是否记录本轮任务缓存、所选RMA连接对象。
    PreLaunchWqe(ubConnLitePtr, needCacheTask, conn);
    // 下发DbSqe前, 备份相关信息
    // [中文导读] [AllReduce逐行 S1139] 设置本地待提交SQE条数为/按`needCacheTask ? stream.GetRtsq()->GetPendingSqeCnt() : 0`（是否记录本轮任务缓存、承载任务的执行流的GetRtsq字段）；返回当前StreamLite持有的具体执行队列；读取Doorbell生成之前当前执行队列的待提交SQE数。
    const uint32_t pendingSqeCnt = needCacheTask ? stream.GetRtsq()->GetPendingSqeCnt() : 0;

    // 批量展开下发WQE
    // [中文导读] [AllReduce逐行 S1142] 设置批项数量为/按`loc.size()`（本端缓冲区/地址的size字段）；读取容器登记项数。
    u32 insNum = loc.size();
    // [中文导读] [AllReduce逐行 S1143] 设置累计传输字节数为/按`0`。
    u64 totalSize = 0;
    // [中文导读] [AllReduce逐行 S1144] 按各批项操作生成WQE并为最后一项设置完成/保序；传入/处理本端缓冲区/地址、远端缓冲区/地址、批操作种类/归约描述数组、批操作通知槽索引数组、承载任务的执行流、所选RMA连接对象、累计传输字节数。
    BatchTransferAllWqe_(loc, rmt, transferOp, notifyIdxs, stream, conn, totalSize);

    // [中文导读] [AllReduce逐行 S1146] 设置是否生成任务观测记录为/按`IsReportTask()`；检查是否需要任务异常/性能观测。
    const bool isReportTask = IsReportTask();
    // 展开下发WQE后, 展开下发DbSqe前, 按需缓存wqe及DbSqeIdx
    // 注意: pendingSqeCnt在下发DbSqe前已备份
    // 注意: 一定要在展开下发DbSqe前调用PostLaunchWqe, 否则如果下发DbSqe触发LaunchTask,
    // 而尚未调用PostLaunchWqe插入当前WQE数组，
    //     会导致AicpuTaskCache找不到当前WQE数组, 无法正确更新对应的DbSqeLocation
    // [中文导读] [AllReduce逐行 S1152] 准备Doorbell任务的缓存观测描述的局部存储/结构描述，初始化方式以本行声明为准。
    DbSqeProfInfo dbSqeProfInfo;
    // [中文导读] [AllReduce逐行 S1153] 仅当`(needCacheTask && isReportTask)`（是否记录本轮任务缓存、是否生成任务观测记录）成立时进入此分支。
    if (needCacheTask && isReportTask) {
        // [中文导读] [AllReduce逐行 S1154] 形成完整批操作Doorbell缓存观测描述。
        BuildDbSqeProfInfoForExecProfilingAll(
            // [中文导读] [AllReduce逐行 S1155] 为形成完整批操作Doorbell缓存观测描述补入`loc[insNum - 1], rmt[insNum - 1], totalSize, transferOp[insNum - 1], notifyIdxs[insNum - 1], dbSqeProfInfo)`（本端缓冲区/地址、批项数量、远端缓冲区/地址、累计传输字节数、批操作种类/归约描述数组、批操作通知槽索引数组、Doorbell任务的缓存观测描述）；本行是参数/结构化初始化续行。
            loc[insNum - 1], rmt[insNum - 1], totalSize, transferOp[insNum - 1], notifyIdxs[insNum - 1], dbSqeProfInfo);
    // [中文导读] [AllReduce逐行 S1156] 结束`if (needCacheTask && isReportTask)`（是否记录本轮任务缓存、是否生成任务观测记录）分支/循环；控制流返回外层。
    }
    // [中文导读] [AllReduce逐行 S1157] 在Doorbell之前保存/打印已展开WQE及DbSqe位置，然后关闭跟踪；传入/处理承载任务的执行流、用于WQE缓存/调试的UB连接对象、是否记录本轮任务缓存、本地待提交SQE条数、是否生成任务观测记录、Doorbell任务的缓存观测描述。
    PostLaunchWqe(stream, ubConnLitePtr, needCacheTask, pendingSqeCnt, isReportTask, dbSqeProfInfo);

    // [中文导读] [AllReduce逐行 S1159] 给当前执行流生成携带UB PI的jetty Doorbell SQE；取得UB jetty的die/function/jetty标识用于Doorbell；传入/处理承载任务的执行流、所选RMA连接对象的GetUbJettyLiteId字段、WQE构造后的UB PI等输出的pi字段。
    BuildUbDbSendTask(stream, conn->GetUbJettyLiteId(), connOut.pi); // 约束使用一批wqe的个数不会导致反压

    // [中文导读] [AllReduce逐行 S1161] 登记完整批操作末项及累计字节的任务观测。
    ExecProfilingAll(
        // [中文导读] [AllReduce逐行 S1162] 为登记完整批操作末项及累计字节的任务观测补入`loc[insNum - 1], rmt[insNum - 1], totalSize, transferOp[insNum - 1], stream, taskId, notifyIdxs[insNum - 1])`（本端缓冲区/地址、批项数量、远端缓冲区/地址、累计传输字节数、批操作种类/归约描述数组、承载任务的执行流、当前任务编号、批操作通知槽索引数组）；本行是参数/结构化初始化续行。
        loc[insNum - 1], rmt[insNum - 1], totalSize, transferOp[insNum - 1], stream, taskId, notifyIdxs[insNum - 1]);
// [中文导读] [AllReduce逐行 S1163] 结束UbTransportLiteImpl::BatchTransferAll函数体；控制流返回外层。
}

// [中文导读] [AllReduce逐行 S1165] UbTransportLiteImpl::BatchTransferAllWqe_的接口声明：逐批项分流数据/归约/通知操作，末项加强保序/完成；这些参数属于本函数调用边界。
inline void UbTransportLiteImpl::BatchTransferAllWqe_(
    // [中文导读] [AllReduce逐行 S1166] UbTransportLiteImpl::BatchTransferAllWqe_的接口声明：本端缓冲区/地址、远端缓冲区/地址；这些参数属于本函数调用边界。
    const std::vector<RmaBufferLite>& loc, const std::vector<Buffer>& rmt,
    // [中文导读] [AllReduce逐行 S1167] UbTransportLiteImpl::BatchTransferAllWqe_的接口声明：批操作种类/归约描述数组、批操作通知槽索引数组；这些参数属于本函数调用边界。
    const std::vector<BaseTransportLiteImpl::TransferOp>& transferOp, const std::vector<uint32_t>& notifyIdxs,
    // [中文导读] [AllReduce逐行 S1168] UbTransportLiteImpl::BatchTransferAllWqe_的接口声明：承载任务的执行流、所选RMA连接对象、累计传输字节数；这些参数属于本函数调用边界。
    const StreamLite& stream, RmaConnLite* conn, u64& totalSize)
// [中文导读] [AllReduce逐行 S1169] 进入UbTransportLiteImpl::BatchTransferAllWqe_函数体：逐批项分流数据/归约/通知操作，末项加强保序/完成。
{
    // [中文导读] [AllReduce逐行 S1170] 设置写入远端通知的值1为/按`1`。
    u64 notifyData = 1; // 普通notify，固定1，用于writeWithNotify与writeReduceWithNotify
    // [中文导读] [AllReduce逐行 S1171] 准备UB WQE保序/完成配置的局部存储/结构描述，初始化方式以本行声明为准。
    SqeConfigLite cfg;
    // [中文导读] [AllReduce逐行 S1172] 把当前传输Fence状态应用到本次WQE配置；传入/处理UB WQE保序/完成配置。
    SetFenceConfig(cfg);
    // [中文导读] [AllReduce逐行 S1173] 设置批项数量为/按`loc.size()`（本端缓冲区/地址的size字段）；读取容器登记项数。
    u32 insNum = loc.size();
    // [中文导读] [AllReduce逐行 S1174] 设置是否生成任务观测记录为/按`IsReportTask()`；检查是否需要任务异常/性能观测。
    const bool isReportTask = IsReportTask();

    // [中文导读] [AllReduce逐行 S1176] 按`(u32 i = 0; i < insNum; i++)`（本批条目下标、批项数量）遍历本批条目/分片；各次处理保持数组对应关系。
    for (u32 i = 0; i < insNum; i++) {
        // [中文导读] [AllReduce逐行 S1177] 设置UB WQE保序/完成配置的cqeEn字段为/按`(i == insNum - 1) ? true : false`（本批条目下标、批项数量）。
        cfg.cqeEn = (i == insNum - 1) ? true : false;                        // 返回最后一个sqe的cqe
        // [中文导读] [AllReduce逐行 S1178] 设置UB WQE保序/完成配置的placeOdr字段为/按`(i == insNum - 1) ? UB_STRONG_ORDER : UB_RELAX_ORDER`（本批条目下标、批项数量）。
        cfg.placeOdr = (i == insNum - 1) ? UB_STRONG_ORDER : UB_RELAX_ORDER; // 最后一个要求保序
        // [中文导读] [AllReduce逐行 S1179] 设置UB WQE保序/完成配置的compOrder字段为/按`(i == insNum - 1) ? UB_COMPLETION : UB_NO_COMPLETION`（本批条目下标、批项数量）。
        cfg.compOrder = (i == insNum - 1) ? UB_COMPLETION : UB_NO_COMPLETION;
        // [中文导读] [AllReduce逐行 S1180] 设置UB WQE保序/完成配置的userConfig字段为/按`true`。
        cfg.userConfig = true;

        // [中文导读] [AllReduce逐行 S1182] 仅当`(transferOp[i].transType == TransferType::NOTIFY_RECORD)`（批操作种类/归约描述数组、本批条目下标）成立时进入此分支。
        if (transferOp[i].transType == TransferType::NOTIFY_RECORD) { // notifyRecord操作没有loc/rmt，因此单独处理
            // [中文导读] [AllReduce逐行 S1183] 仅当`(notifyIdxs[i] == 1)`（批操作通知槽索引数组、本批条目下标）成立时进入此分支。
            if (notifyIdxs[i] == 1) {                                 // PostFin场景
                // [中文导读] [AllReduce逐行 S1184] 设置UB WQE保序/完成配置的cqeEn字段为/按`true`。
                cfg.cqeEn = true;
                // [中文导读] [AllReduce逐行 S1185] 设置UB WQE保序/完成配置的placeOdr字段为/按`UB_STRONG_ORDER`。
                cfg.placeOdr = UB_STRONG_ORDER;
                // [中文导读] [AllReduce逐行 S1186] 设置UB WQE保序/完成配置的compOrder字段为/按`UB_COMPLETION`。
                cfg.compOrder = UB_COMPLETION;
                // [中文导读] [AllReduce逐行 S1187] 设置UB WQE保序/完成配置的userConfig字段为/按`true`。
                cfg.userConfig = true;
            // [中文导读] [AllReduce逐行 S1188] 结束`if (notifyIdxs[i] == 1)`（批操作通知槽索引数组、本批条目下标）分支/循环；控制流返回外层。
            }
            // [中文导读] [AllReduce逐行 S1189] 设置内联通知值1为/按`1`。
            u32 inlineData = 1;
            // 当前使用1个connection，下标为0 构建sqe
            // [中文导读] [AllReduce逐行 S1191] 生成向远端通知区内联写值1的WQE；传入/处理所选RMA连接对象的InlineWrite字段。
            conn->InlineWrite(
                // [中文导读] [AllReduce逐行 S1192] 为生成向远端通知区内联写值1的WQE；按对端通知槽取远端通知地址/token/notifyId补入`ReinterpretAs<u8*>(&inlineData), UB_INLINE_WRITE_SIZE, GetRmtNotifySliceLite(notifyIdxs[i]), cfg,`（内联通知值1、批操作通知槽索引数组、本批条目下标、UB WQE保序/完成配置）；本行是参数/结构化初始化续行。
                ReinterpretAs<u8*>(&inlineData), UB_INLINE_WRITE_SIZE, GetRmtNotifySliceLite(notifyIdxs[i]), cfg,
                // [中文导读] [AllReduce逐行 S1193] 为生成向远端通知区内联写值1的WQE；按对端通知槽取远端通知地址/token/notifyId补入`stream, connOut)`（承载任务的执行流、WQE构造后的UB PI等输出）；本行是参数/结构化初始化续行。
                stream, connOut);
        // [中文导读] [AllReduce逐行 S1194] 进入上一个条件未命中的替代路径，按本函数的资源/设备/协议分流继续处理。
        } else {
            // [中文导读] [AllReduce逐行 S1195] 设置当前批项本端切片为/按`GetRmaBufSlicelite(loc[i])`（本端缓冲区/地址、本批条目下标）；将本端RMA描述改写为UB本地slice，rkey置0。
            auto localBuffer = GetRmaBufSlicelite(loc[i]);
            // [中文导读] [AllReduce逐行 S1196] 设置当前批项远端切片为/按`GetRmtRmaBufSliceLite(rmt[i])`（远端缓冲区/地址、本批条目下标）；从远端已交换注册区解析目标地址/长度/token。
            auto remoteBuffer = GetRmtRmaBufSliceLite(rmt[i]);
            // [中文导读] [AllReduce逐行 S1197] 仅当`(transferOp[i].transType == TransferType::WRITE)`（批操作种类/归约描述数组、本批条目下标）成立时进入此分支。
            if (transferOp[i].transType == TransferType::WRITE) {
                // [中文导读] [AllReduce逐行 S1198] 向连接远端目标生成WRITE WQE，本端为源；传入/处理所选RMA连接对象的Write字段、当前批项本端切片、当前批项远端切片、UB WQE保序/完成配置、承载任务的执行流、WQE构造后的UB PI等输出。
                conn->Write(localBuffer, remoteBuffer, cfg, stream, connOut);
            // [中文导读] [AllReduce逐行 S1199] 仅当`(transferOp[i].transType == TransferType::WRITE_REDUCE)`（批操作种类/归约描述数组、本批条目下标）成立时进入此分支。
            } else if (transferOp[i].transType == TransferType::WRITE_REDUCE) {
                // [中文导读] [AllReduce逐行 S1200] 构造向远端目标归约的WRITE WQE；传入/处理所选RMA连接对象的WriteReduce字段。
                conn->WriteReduce(
                    // [中文导读] [AllReduce逐行 S1201] 为构造向远端目标归约的WRITE WQE补入`transferOp[i].reduceIn.dataType, transferOp[i].reduceIn.reduceOp, localBuffer, stream, remoteBuffer,`（批操作种类/归约描述数组、本批条目下标、当前批项本端切片、承载任务的执行流、当前批项远端切片）；本行是参数/结构化初始化续行。
                    transferOp[i].reduceIn.dataType, transferOp[i].reduceIn.reduceOp, localBuffer, stream, remoteBuffer,
                    // [中文导读] [AllReduce逐行 S1202] 为构造向远端目标归约的WRITE WQE补入`cfg, connOut)`（UB WQE保序/完成配置、WQE构造后的UB PI等输出）；本行是参数/结构化初始化续行。
                    cfg, connOut);
            // [中文导读] [AllReduce逐行 S1203] 仅当`(transferOp[i].transType == TransferType::READ)`（批操作种类/归约描述数组、本批条目下标）成立时进入此分支。
            } else if (transferOp[i].transType == TransferType::READ) {
                // [中文导读] [AllReduce逐行 S1204] 生成从远端源读取到本端目标的READ WQE；传入/处理所选RMA连接对象的Read字段、当前批项本端切片、当前批项远端切片、UB WQE保序/完成配置、承载任务的执行流、WQE构造后的UB PI等输出。
                conn->Read(localBuffer, remoteBuffer, cfg, stream, connOut);
            // [中文导读] [AllReduce逐行 S1205] 仅当`(transferOp[i].transType == TransferType::READ_REDUCE)`（批操作种类/归约描述数组、本批条目下标）成立时进入此分支。
            } else if (transferOp[i].transType == TransferType::READ_REDUCE) {
                // [中文导读] [AllReduce逐行 S1206] 生成读取远端源并归约到本地目标的READ WQE；传入/处理所选RMA连接对象的ReadReduce字段、批操作种类/归约描述数组、本批条目下标、当前批项本端切片、当前批项远端切片、承载任务的执行流、UB WQE保序/完成配置、WQE构造后的UB PI等输出。
                conn->ReadReduce(transferOp[i].reduceIn, localBuffer, remoteBuffer, stream, cfg, connOut);
            // [中文导读] [AllReduce逐行 S1207] 仅当`(transferOp[i].transType == TransferType::WRITE_WITH_NOTIFY)`（批操作种类/归约描述数组、本批条目下标）成立时进入此分支。
            } else if (transferOp[i].transType == TransferType::WRITE_WITH_NOTIFY) {
                // [中文导读] [AllReduce逐行 S1208] 构造数据写入与对端通知组合WQE；传入/处理所选RMA连接对象的WriteWithNotify字段。
                conn->WriteWithNotify(
                    // [中文导读] [AllReduce逐行 S1209] 为构造数据写入与对端通知组合WQE；按对端通知槽取远端通知地址/token/notifyId补入`localBuffer, remoteBuffer, cfg, connOut, GetRmtNotifySliceLite(notifyIdxs[i]), stream,`（当前批项本端切片、当前批项远端切片、UB WQE保序/完成配置、WQE构造后的UB PI等输出、批操作通知槽索引数组、本批条目下标、承载任务的执行流）；本行是参数/结构化初始化续行。
                    localBuffer, remoteBuffer, cfg, connOut, GetRmtNotifySliceLite(notifyIdxs[i]), stream,
                    // [中文导读] [AllReduce逐行 S1210] 为构造数据写入与对端通知组合WQE；按对端通知槽取远端通知地址/token/notifyId补入`notifyData)`（写入远端通知的值1）；本行是参数/结构化初始化续行。
                    notifyData); // 当前使用1个connection，下标为0
            // [中文导读] [AllReduce逐行 S1211] 仅当`(transferOp[i].transType == TransferType::WRITE_REDUCE_WITH_NOTIFY)`（批操作种类/归约描述数组、本批条目下标）成立时进入此分支。
            } else if (transferOp[i].transType == TransferType::WRITE_REDUCE_WITH_NOTIFY) {
                // [中文导读] [AllReduce逐行 S1212] 构造远端归约与通知组合WQE；传入/处理所选RMA连接对象的WriteReduceWithNotify字段。
                conn->WriteReduceWithNotify(
                    // [中文导读] [AllReduce逐行 S1213] 为构造远端归约与通知组合WQE补入`transferOp[i].reduceIn.dataType, transferOp[i].reduceIn.reduceOp, localBuffer, remoteBuffer, cfg,`（批操作种类/归约描述数组、本批条目下标、当前批项本端切片、当前批项远端切片、UB WQE保序/完成配置）；本行是参数/结构化初始化续行。
                    transferOp[i].reduceIn.dataType, transferOp[i].reduceIn.reduceOp, localBuffer, remoteBuffer, cfg,
                    // [中文导读] [AllReduce逐行 S1214] 为构造远端归约与通知组合WQE；按对端通知槽取远端通知地址/token/notifyId补入`stream, connOut, GetRmtNotifySliceLite(notifyIdxs[i]),`（承载任务的执行流、WQE构造后的UB PI等输出、批操作通知槽索引数组、本批条目下标）；本行是参数/结构化初始化续行。
                    stream, connOut, GetRmtNotifySliceLite(notifyIdxs[i]),
                    // [中文导读] [AllReduce逐行 S1215] 为构造远端归约与通知组合WQE；按对端通知槽取远端通知地址/token/notifyId补入`notifyData)`（写入远端通知的值1）；本行是参数/结构化初始化续行。
                    notifyData); // 当前使用1个connection，下标为0
            // [中文导读] [AllReduce逐行 S1216] 结束`if (transferOp[i].transType == TransferType::WRITE)`（批操作种类/归约描述数组、本批条目下标）分支/循环；控制流返回外层。
            }
        // [中文导读] [AllReduce逐行 S1217] 结束`if (transferOp[i].transType == TransferType::NOTIFY_RECORD)`（批操作种类/归约描述数组、本批条目下标）分支/循环；控制流返回外层。
        }
        // [中文导读] [AllReduce逐行 S1218] 仅当`(isReportTask)`（是否生成任务观测记录）成立时进入此分支。
        if (isReportTask) {
            // [中文导读] [AllReduce逐行 S1219] 增加累计传输字节数为/按`GetRmaBufSlicelite(loc[i]).GetSize()`（本端缓冲区/地址、本批条目下标）；将本端RMA描述改写为UB本地slice，rkey置0；读取缓冲区字节长度。
            totalSize += GetRmaBufSlicelite(loc[i]).GetSize();
        // [中文导读] [AllReduce逐行 S1220] 结束`if (isReportTask)`（是否生成任务观测记录）分支/循环；控制流返回外层。
        }
    // [中文导读] [AllReduce逐行 S1221] 结束`for (u32 i = 0; i < insNum; i++)`（本批条目下标、批项数量）分支/循环；控制流返回外层。
    }
// [中文导读] [AllReduce逐行 S1222] 结束UbTransportLiteImpl::BatchTransferAllWqe_函数体；控制流返回外层。
}

void UbTransportLiteImpl::Drain(const StreamLite& stream)
{
    std::lock_guard<std::mutex> lock(drainMtx_);
    if (drainNotify_.size == 0 || rmtDrainBuffer_.size == 0) {
        HCCL_WARNING("[UbTransportLiteImpl::%s] drain resource is null skip", __func__);
        return;
    }

    SqeConfigLite cfg;
    Fence();
    SetFenceConfig(cfg);

    auto taskId = stream.GetRtsq()->GetTaskId();

    // 当前使用1个connection，下标为0 (当前只有一个connection，对应一个jetty)
    RmaConnLite* conn = connVec[0];

    // 展开下发WQE前, 按需设置cache context
    UbConnLite* ubConnLitePtr = nullptr;
    bool needCacheTask = false;
    PreLaunchWqe(ubConnLitePtr, needCacheTask, conn);
    // 下发DbSqe前, 备份相关信息
    const uint32_t pendingSqeCnt = needCacheTask ? stream.GetRtsq()->GetPendingSqeCnt() : 0;

    // 展开下发WQE
    auto drainNotifyBufSlice = RmaBufSliceLite(drainNotify_.addr, drainNotify_.size, 0, drainNotify_.tokenId);
    auto drainConstBufSlice = RmtRmaBufSliceLite(
        rmtDrainBuffer_.addr, rmtDrainBuffer_.size, 0, rmtDrainBuffer_.tokenId, rmtDrainBuffer_.tokenValue, UINT32_MAX);
    conn->Read(drainNotifyBufSlice, drainConstBufSlice, cfg, stream, connOut);

    const bool isReportTask = IsReportTask();
    // 展开下发WQE后, 展开下发DbSqe前, 按需缓存wqe及DbSqeIdx
    // 注意: pendingSqeCnt在下发DbSqe前已备份
    // 注意: 一定要在展开下发DbSqe前调用PostLaunchWqe, 否则如果下发DbSqe触发LaunchTask,
    // 而尚未调用PostLaunchWqe插入当前WQE数组，
    //     会导致AicpuTaskCache找不到当前WQE数组, 无法正确更新对应的DbSqeLocation
    DbSqeProfInfo dbSqeProfInfo;
    if (needCacheTask && isReportTask) {
        BuildDbSqeProfInfoForProfilingProcess(
            ReinterpretAs<void*>(drainNotifyBufSlice.GetAddr()), ReinterpretAs<void*>(drainConstBufSlice.GetAddr()),
            drainNotifyBufSlice.GetSize(), DmaOp::HCCL_DMA_READ, dbSqeProfInfo);
    }
    PostLaunchWqe(stream, ubConnLitePtr, needCacheTask, pendingSqeCnt, isReportTask, dbSqeProfInfo);

    BuildUbDbSendTask(stream, conn->GetUbJettyLiteId(), connOut.pi);
    ProfilingProcess(
        ReinterpretAs<void*>(drainNotifyBufSlice.GetAddr()), ReinterpretAs<void*>(drainConstBufSlice.GetAddr()),
        drainNotifyBufSlice.GetSize(), stream, DmaOp::HCCL_DMA_READ, taskId);

    auto waitTaskId = stream.GetRtsq()->GetTaskId();
    BuildNotifyWaitTask(stream, drainNotify_.notifyId);
    if (IsReportTask()) {
        if (callback_) {
            TaskParam taskParam{};
            taskParam.taskType = TaskParamType::TASK_NOTIFY_WAIT;
            taskParam.beginTime = ProfGetCurCpuTimestamp();
            taskParam.taskPara.Notify.notifyID = drainNotify_.notifyId;
            taskParam.taskPara.Notify.value = 1;
            AddTaskCallback(stream, waitTaskId, taskParam);
        }
        FillSlotWaitInfo(stream, waitTaskId);
    }
}

void UbTransportLiteImpl::WriteWithNotify(
    const RmaBufferLite& loc, const Buffer& rmt, const WithNotifyIn& withNotify, const StreamLite& stream)
{
    SqeConfigLite cfg;
    SetFenceConfig(cfg);
    u64 notifyData = 1; // 普通notify，固定1

    auto taskId = stream.GetRtsq()->GetTaskId();

    // 当前使用1个connection，下标为0
    RmaConnLite* conn = connVec[0];

    // 展开下发WQE前, 按需设置cache context
    UbConnLite* ubConnLitePtr = nullptr;
    bool needCacheTask = false;
    PreLaunchWqe(ubConnLitePtr, needCacheTask, conn);
    // 下发DbSqe前, 备份相关信息
    const uint32_t pendingSqeCnt = needCacheTask ? stream.GetRtsq()->GetPendingSqeCnt() : 0;

    // 展开下发WQE
    auto locRmaBufSlicelite = GetRmaBufSlicelite(loc);
    auto rmtRmaBufSlicelite = GetRmtRmaBufSliceLite(rmt);
    auto rmtNotifySliceLite = GetRmtNotifySliceLite(withNotify.index_);
    conn->WriteWithNotify(locRmaBufSlicelite, rmtRmaBufSlicelite, cfg, connOut, rmtNotifySliceLite, stream, notifyData);

    // 展开下发WQE后, 展开下发DbSqe前, 按需缓存wqe及DbSqeIdx
    // 注意: pendingSqeCnt在下发DbSqe前已备份
    // 注意: 一定要在展开下发DbSqe前调用PostLaunchWqe, 否则如果下发DbSqe触发LaunchTask,
    // 而尚未调用PostLaunchWqe插入当前WQE数组，
    //     会导致AicpuTaskCache找不到当前WQE数组, 无法正确更新对应的DbSqeLocation
    const bool isReportTask = IsReportTask();
    DbSqeProfInfo dbSqeProfInfo;
    if (needCacheTask && isReportTask) {
        BuildDbSqeProfInfoForWriteWithNotify(
            ReinterpretAs<void*>(locRmaBufSlicelite.GetAddr()), ReinterpretAs<void*>(rmtRmaBufSlicelite.GetAddr()),
            locRmaBufSlicelite.GetSize(), rmtNotifySliceLite.GetNotifyId(), dbSqeProfInfo);
    }
    PostLaunchWqe(stream, ubConnLitePtr, needCacheTask, pendingSqeCnt, isReportTask, dbSqeProfInfo);

    BuildUbDbSendTask(stream, conn->GetUbJettyLiteId(), connOut.pi);

    WriteWithNotifyProfilingProcess(
        ReinterpretAs<void*>(locRmaBufSlicelite.GetAddr()), ReinterpretAs<void*>(rmtRmaBufSlicelite.GetAddr()),
        locRmaBufSlicelite.GetSize(), stream, taskId, rmtNotifySliceLite.GetNotifyId());
}

void UbTransportLiteImpl::WriteReduceWithNotify(
    const RmaBufferLite& loc, const Buffer& rmt, const ReduceIn& reduceIn, const WithNotifyIn& withNotify,
    const StreamLite& stream)
{
    SqeConfigLite cfg;
    SetFenceConfig(cfg);
    u64 notifyData = 1; // 普通notify，固定1

    auto taskId = stream.GetRtsq()->GetTaskId();

    // 当前使用1个connection，下标为0
    RmaConnLite* conn = connVec[0];

    // 展开下发WQE前, 按需设置cache context
    UbConnLite* ubConnLitePtr = nullptr;
    bool needCacheTask = false;
    PreLaunchWqe(ubConnLitePtr, needCacheTask, conn);
    // 下发DbSqe前, 备份相关信息
    const uint32_t pendingSqeCnt = needCacheTask ? stream.GetRtsq()->GetPendingSqeCnt() : 0;

    // 展开下发WQE
    auto locRmaBufSlicelite = GetRmaBufSlicelite(loc);
    auto rmtRmaBufSlicelite = GetRmtRmaBufSliceLite(rmt);
    auto rmtNotifySliceLite = GetRmtNotifySliceLite(withNotify.index_);
    conn->WriteReduceWithNotify(
        reduceIn.dataType, reduceIn.reduceOp, locRmaBufSlicelite, rmtRmaBufSlicelite, cfg, stream, connOut,
        rmtNotifySliceLite, notifyData);

    // 展开下发WQE后, 展开下发DbSqe前, 按需缓存wqe及DbSqeIdx
    // 注意: pendingSqeCnt在下发DbSqe前已备份
    // 注意: 一定要在展开下发DbSqe前调用PostLaunchWqe, 否则如果下发DbSqe触发LaunchTask,
    // 而尚未调用PostLaunchWqe插入当前WQE数组，
    //     会导致AicpuTaskCache找不到当前WQE数组, 无法正确更新对应的DbSqeLocation
    const bool isReportTask = IsReportTask();
    DbSqeProfInfo dbSqeProfInfo;
    if (needCacheTask && isReportTask) {
        BuildDbSqeProfInfoForWriteReduceWithNotify(
            ReinterpretAs<void*>(locRmaBufSlicelite.GetAddr()), ReinterpretAs<void*>(rmtRmaBufSlicelite.GetAddr()),
            locRmaBufSlicelite.GetSize(), reduceIn, rmtNotifySliceLite.GetNotifyId(), dbSqeProfInfo);
    }
    PostLaunchWqe(stream, ubConnLitePtr, needCacheTask, pendingSqeCnt, isReportTask, dbSqeProfInfo);

    BuildUbDbSendTask(stream, conn->GetUbJettyLiteId(), connOut.pi);

    WriteReduceWithNotifyProfilingProcess(
        ReinterpretAs<void*>(locRmaBufSlicelite.GetAddr()), ReinterpretAs<void*>(rmtRmaBufSlicelite.GetAddr()),
        locRmaBufSlicelite.GetSize(), reduceIn, stream, taskId, rmtNotifySliceLite.GetNotifyId());
}

void UbTransportLiteImpl::BatchOneSidedRead(
    const vector<RmaBufSliceLite>& loc, const vector<RmtRmaBufSliceLite>& rmt, const StreamLite& stream)
{
    SqeConfigLite cfg;
    SetFenceConfig(cfg);

    // 当前使用1个connection，下标为0
    RmaConnLite* conn = connVec[0];

    // 展开下发WQE前, 按需设置cache context
    UbConnLite* ubConnLitePtr = nullptr;
    bool needCacheTask = false;
    PreLaunchWqe(ubConnLitePtr, needCacheTask, conn);
    // 下发DbSqe前, 备份相关信息
    const uint32_t pendingSqeCnt = needCacheTask ? stream.GetRtsq()->GetPendingSqeCnt() : 0;

    // 展开下发WQE
    conn->BatchOneSidedRead(loc, rmt, cfg, stream, connOut);

    // 展开下发WQE后, 展开下发DbSqe前, 按需缓存wqe及DbSqeIdx
    // 注意: pendingSqeCnt在下发DbSqe前已备份
    // 注意: 一定要在展开下发DbSqe前调用PostLaunchWqe, 否则如果下发DbSqe触发LaunchTask,
    // 而尚未调用PostLaunchWqe插入当前WQE数组，
    //     会导致AicpuTaskCache找不到当前WQE数组, 无法正确更新对应的DbSqeLocation
    PostLaunchWqe(stream, ubConnLitePtr, needCacheTask, pendingSqeCnt, false, DbSqeProfInfo());

    BuildUbDbSendTask(stream, conn->GetUbJettyLiteId(), connOut.pi);
}

void UbTransportLiteImpl::BatchOneSidedWrite(
    const vector<RmaBufSliceLite>& loc, const vector<RmtRmaBufSliceLite>& rmt, const StreamLite& stream)
{
    SqeConfigLite cfg;
    SetFenceConfig(cfg);

    // 当前使用1个connection，下标为0
    RmaConnLite* conn = connVec[0];

    // 展开下发WQE前, 按需设置cache context
    UbConnLite* ubConnLitePtr = nullptr;
    bool needCacheTask = false;
    PreLaunchWqe(ubConnLitePtr, needCacheTask, conn);
    // 下发DbSqe前, 备份相关信息
    const uint32_t pendingSqeCnt = needCacheTask ? stream.GetRtsq()->GetPendingSqeCnt() : 0;

    // 展开下发WQE
    conn->BatchOneSidedWrite(loc, rmt, cfg, stream, connOut);

    // 展开下发WQE后, 展开下发DbSqe前, 按需缓存wqe及DbSqeIdx
    // 注意: pendingSqeCnt在下发DbSqe前已备份
    // 注意: 一定要在展开下发DbSqe前调用PostLaunchWqe, 否则如果下发DbSqe触发LaunchTask,
    // 而尚未调用PostLaunchWqe插入当前WQE数组，
    //     会导致AicpuTaskCache找不到当前WQE数组, 无法正确更新对应的DbSqeLocation
    PostLaunchWqe(stream, ubConnLitePtr, needCacheTask, pendingSqeCnt, false, DbSqeProfInfo());

    BuildUbDbSendTask(stream, conn->GetUbJettyLiteId(), connOut.pi);
}

Eid UbTransportLiteImpl::GetLocEid() const { return connVec[0]->GetLocEid(); }

Eid UbTransportLiteImpl::GetRmtEid() const { return connVec[0]->GetRmtEid(); }

uint64_t UbTransportLiteImpl::GetJettyHandle() const { return connVec[0]->GetJettyHandle(); }

uint32_t UbTransportLiteImpl::GetJettyId() const { return connVec[0]->GetJettyId(); }

uint32_t UbTransportLiteImpl::GetTpn() const { return connVec[0]->GetTpn(); }

HcclResult UbTransportLiteImpl::Clean()
{
    locNotifyVec.clear();
    rmtNotifyVec.clear();
    locBufferMap.clear();
    rmtBufferVec.clear();
    rmtBufferMap.clear();

    // 清理connVec，connLite由UbConnLiteMgr管理
    for (auto& it : connUniqueIdVec) {
        DECTOR_TRY_CATCH("UbTransportLiteImpl", UbConnLiteMgr::GetInstance().Clear(it));
    }
    connUniqueIdVec.clear();
    connVec.clear();

    return HCCL_SUCCESS;
}

HcclResult UbTransportLiteImpl::Resume(std::vector<char>& uniqueId)
{
    Init(uniqueId);
    return HCCL_SUCCESS;
}

HcclResult UbTransportLiteImpl::Fence()
{
    fence_ = true;
    HCCL_INFO("[%s] SUCCESS. fence[%d]", __func__, fence_);
    return HCCL_SUCCESS;
}

void UbTransportLiteImpl::SetFenceConfig(SqeConfigLite& cfg)
{
    if (fence_) {
        cfg.fence = UB_FENCE_ENABLED;
        cfg.placeOdr = UB_STRONG_ORDER;
        cfg.compOrder = UB_COMPLETION;
        cfg.userConfig = true;
    }
    fence_ = false;
}

bool UbTransportLiteImpl::IsReportTask() const
{
    return taskExceptionEnable_ || DfxProfilingHandlerLite::GetInstance().GetProfL1State();
}
} // namespace Hccl
