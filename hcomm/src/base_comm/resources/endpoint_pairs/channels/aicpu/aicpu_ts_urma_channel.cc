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
#include "aicpu_ts_urma_channel.h"
#include "endpoint.h"
#include "../../sockets/socket_mgr.h"
#include "orion_adpt_utils.h"
#include "hcomm_c_adpt.h"
#include "config_log.h"
#include "config_plf_log_v2.h"
#include "hcomm_res_mgr.h"
#include "endpoint.h"

// Orion
#include "adapter_rts_common.h"
#include "topo_common_types.h"
#include "virtual_topo.h"
#include "aicpu_res_package_helper.h"
#include "makebufs_helper.h"

namespace hcomm {
using Hccl::PLF_CHANNEL;
constexpr uint16_t DEFAULT_LISTENING_PORT = 60001;

AicpuTsUrmaChannel::AicpuTsUrmaChannel(EndpointHandle endpointHandle, const HcommChannelDesc& channelDesc)
    : endpointHandle_(endpointHandle),
      channelDesc_(channelDesc)
{}

AicpuTsUrmaChannel::~AicpuTsUrmaChannel()
{
    if (channelDesc_.socket == nullptr && socket_ != nullptr) {
        SocketMgr::GetInstance(devicePhyId_).PutSocket(socketConfig_, socket_);
        socket_ = nullptr;
    }
}

HcclResult AicpuTsUrmaChannel::ParseInputParam()
{
    // 1. 从 endpointHandle_，获得 localEp_ 和 rdmaHandle_
    // TODO: 使用 HcommEndpointGet
    Endpoint* localEpPtr = static_cast<Endpoint*>(endpointHandle_);
    CHK_PTR_NULL(localEpPtr);
    localEp_ = localEpPtr->GetEndpointDesc();
    rdmaHandle_ = localEpPtr->GetRdmaHandle();

    HCCL_INFO("[AicpuTsUrmaChannel][%s] localProtocol[%d]", __func__, localEp_.protocol);

    // 2. 从 channelDesc_，获得 remoteEp_, socket_ 和 notifyNum
    remoteEp_ = channelDesc_.remoteEndpoint;
    socket_ = static_cast<Hccl::Socket*>(channelDesc_.socket);
    notifyNum_ = channelDesc_.notifyNum;
    commonRes_.bufferVec.clear();

    if (channelDesc_.exchangeAllMems) {
        // 3. Get memHandles from endpoint
        HCCL_INFO("[AicpuTsUrmaChannel][%s] exchangeAllMems == True. Get memHandles from endpoint.", __func__);
        std::shared_ptr<Hccl::LocalUbRmaBuffer>* memHandles = nullptr;
        uint32_t memHandleNum = 0;
        CHK_RET(static_cast<HcclResult>(
            HcommMemGetAllMemHandles(endpointHandle_, ReinterpretAs<void**>(&memHandles), &memHandleNum)));
        HCCL_INFO("[AicpuTsUrmaChannel][%s] Got memHandleNum[%u].", __func__, memHandleNum);
        for (uint32_t i = 0; i < memHandleNum; ++i) {
            std::shared_ptr<Hccl::LocalUbRmaBuffer>& localUbRmaBuffer = memHandles[i];
            CHK_SMART_PTR_NULL(localUbRmaBuffer);
            Hccl::Buffer* buf = localUbRmaBuffer->GetBuf();
            CHK_PTR_NULL(buf);
            HCCL_INFO(
                "[AicpuTsUrmaChannel][%s] Got memHandle No.%u: addr[0x%llx], size[0x%llx], type[%d], memInfo[%s].",
                __func__, i, static_cast<unsigned long long>(localUbRmaBuffer->GetAddr()),
                static_cast<unsigned long long>(localUbRmaBuffer->GetSize()), static_cast<int>(buf->GetMemType()),
                buf->GetMemInfo().c_str());
            commonRes_.bufferVec.push_back(localUbRmaBuffer.get());
        }
    } else {
        // 3. 从 channelDesc 的 memHandle，获得 bufs_
        HCCL_INFO("[AicpuTsUrmaChannel][%s] exchangeAllMems == false. Get memHandles from channelDesc.", __func__);
        CHK_RET(MakeRmaBufferVecFromMemHandles(
            channelDesc_.memHandles, channelDesc_.memHandleNum, commonRes_.bufferVec, "AicpuTsUrmaChannel"));
    }

    return HCCL_SUCCESS;
}

HcclResult AicpuTsUrmaChannel::BuildAttr()
{
    attr_.devicePhyId = localEp_.loc.device.devPhyId;
    attr_.opMode = Hccl::OpMode::OPBASE;
    return HCCL_SUCCESS;
}

HcclResult AicpuTsUrmaChannel::BuildConnection()
{
    UbConnBuildContext ctx;
    CHK_RET(PrepareUbConnBuildContext(localEp_, remoteEp_, channelDesc_, ctx));

    Hccl::OpMode opMode = Hccl::OpMode::OPBASE;
    bool devUsed = true; // aicpu 为 true
    // UB_CTP → HCOMM_TA_CTP_UB_TIMEOUT，UB_TP → HCOMM_TA_RTP_UB_TIMEOUT
    u8 taTimeOut = 0;
    uint32_t taTimeOutValue = 0;
    if (ctx.protocol == Hccl::LinkProtocol::UB_CTP) {
        CHK_RET(hcomm::HcommResMgr::GetInstance().GetConfigMgr().GetRdmaConfig().GetTaCtpUbTimeOut(taTimeOutValue));
    } else {
        CHK_RET(hcomm::HcommResMgr::GetInstance().GetConfigMgr().GetRdmaConfig().GetTaRtpUbTimeOut(taTimeOutValue));
    }
    taTimeOut = static_cast<u8>(taTimeOutValue);
    std::unique_ptr<Hccl::DevUbConnection> ubConn = nullptr;
    switch (ctx.protocol) {
        case Hccl::LinkProtocol::UB_TP:
            EXCEPTION_CATCH(
                ubConn = std::make_unique<Hccl::DevUbTpConnection>(
                    rdmaHandle_, ctx.locAddr, ctx.rmtAddr, opMode, devUsed, Hccl::HrtUbJfcMode::STARS_POLL,
                    Hccl::IpAddress(), Hccl::IpAddress(), ctx.qosPre, taTimeOut, COMM_ENGINE_AICPU_TS, ctx.sqDepth),
                return HCCL_E_PTR);
            break;
        case Hccl::LinkProtocol::UB_CTP:
            EXCEPTION_CATCH(
                ubConn = std::make_unique<Hccl::DevUbCtpConnection>(
                    rdmaHandle_, ctx.locAddr, ctx.rmtAddr, opMode, devUsed, Hccl::HrtUbJfcMode::STARS_POLL,
                    Hccl::IpAddress(), Hccl::IpAddress(), ctx.qosPre, taTimeOut, COMM_ENGINE_AICPU_TS, ctx.sqDepth),
                return HCCL_E_PTR);
            break;
        default:
            HCCL_ERROR("%s No LinkProtocol to match", __func__);
            break;
    }
    CHK_SMART_PTR_NULL(ubConn);

    if (devBaseAttr_.maxReadSize == 0 || devBaseAttr_.maxWriteSize == 0) {
        HCCL_ERROR(
            "%s maxReadSize[%u] or maxWriteSize[%u] must not be zero", __func__, devBaseAttr_.maxReadSize,
            devBaseAttr_.maxWriteSize);
        return HCCL_E_PARA;
    }
    ubConn->SetMaxReadSize(devBaseAttr_.maxReadSize);
    ubConn->SetMaxWriteSize(devBaseAttr_.maxWriteSize);
    HCCL_INFO("%s maxReadSize[%u], maxWriteSize[%u]", __func__, devBaseAttr_.maxReadSize, devBaseAttr_.maxWriteSize);

    commonRes_.connVec.clear();
    commonRes_.connVec.emplace_back(ubConn.get());
    connections_.clear();
    connections_.push_back(std::move(ubConn));
    PLF_CONFIG_INFO(
        PLF_CHANNEL, "[AicpuTsUrmaChannel] build DevUbConnection, protocol[%s] sqDepth[%u].",
        ctx.protocol.Describe().c_str(), ctx.sqDepth);
    return HCCL_SUCCESS;
}

HcclResult AicpuTsUrmaChannel::BuildNotify()
{
    localNotifies_.clear();
    commonRes_.notifyVec.clear();
    bool devUsed = true;
    for (uint32_t i = 0; i < notifyNum_; ++i) {
        std::unique_ptr<Hccl::UbLocalNotify> notifyPtr = nullptr;
        EXCEPTION_CATCH(notifyPtr = std::make_unique<Hccl::UbLocalNotify>(rdmaHandle_, devUsed), return HCCL_E_PTR);
        commonRes_.notifyVec.push_back(notifyPtr.get());
        localNotifies_.push_back(std::move(notifyPtr));
    }
    PLF_CONFIG_INFO(PLF_CHANNEL, "[AicpuTsUrmaChannel] create notify, notifyNum[%u].", notifyNum_);
    return HCCL_SUCCESS;
}

// [中文导读] [AllReduce逐行 S173] AicpuTsUrmaChannel::BuildUbMemTransport的接口声明：按已建立Socket与端点对描述构造Host UbMemTransport；这些参数属于本函数调用边界。
HcclResult AicpuTsUrmaChannel::BuildUbMemTransport()
// [中文导读] [AllReduce逐行 S174] 进入AicpuTsUrmaChannel::BuildUbMemTransport函数体：按已建立Socket与端点对描述构造Host UbMemTransport。
{
    // [中文导读] [AllReduce逐行 S175] 准备`Hccl::BaseMemTransport::LocCntNotifyRes locCntNotifyRes{}`的局部存储/结构描述，初始化方式以本行声明为准。
    Hccl::BaseMemTransport::LocCntNotifyRes locCntNotifyRes{};
    // [中文导读] [AllReduce逐行 S176] 调用clear。
    locCntNotifyRes.vec.clear();
    // [中文导读] [AllReduce逐行 S177] 调用clear。
    locCntNotifyRes.desc.clear();
    // [中文导读] [AllReduce逐行 S178] 设置已连接的Socket对象为/按`*socket_`。
    const Hccl::Socket& socket = *socket_;

    // [中文导读] [AllReduce逐行 S180] 设置按端点对构造的链路描述为/按`BuildDefaultLinkData()`；调用BuildDefaultLinkData。
    Hccl::LinkData linkData = BuildDefaultLinkData();
    // [中文导读] [AllReduce逐行 S181] 把本端与远端端点描述转换为链路属性；返回非成功时由检查宏立即向上传递。
    CHK_RET(EndpointDescPairToLinkData(localEp_, remoteEp_, linkData));

    // [中文导读] [AllReduce逐行 S183] 设置按Socket角色决定的描述交换先后顺序为/按`socket.GetRole() == Hccl::SocketRole::CLIENT ? true : false`（已连接的Socket对象的GetRole字段）；调用GetRole，使用已连接的Socket对象的GetRole字段。
    bool isRecvFirst = socket.GetRole() == Hccl::SocketRole::CLIENT ? true : false;

    // make_unique / make_shared / release 包一层抛异常的宏
    // [中文导读] [AllReduce逐行 S186] 在异常捕获边界执行后续表达式；异常按后续处理语句转换成HCCL状态或提前返回。
    EXCEPTION_CATCH(
        // [中文导读] [AllReduce逐行 S187] 为前述多行表达式补入`memTransport_ = std::make_unique<Hccl::UbMemTransport>(`（Host侧UB内存传输对象）；本行是参数/结构化初始化续行。
        memTransport_ = std::make_unique<Hccl::UbMemTransport>(
            // [中文导读] [AllReduce逐行 S188] 为前述多行表达式补入`commonRes_, attr_, linkData, socket, rdmaHandle_, locCntNotifyRes, isRecvFirst),`（通道公共本地资源、UB传输属性、按端点对构造的链路描述、已连接的Socket对象、底层网络资源句柄、按Socket角色决定的描述交换先后顺序）；本行是参数/结构化初始化续行。
            commonRes_, attr_, linkData, socket, rdmaHandle_, locCntNotifyRes, isRecvFirst),
        // [中文导读] [AllReduce逐行 S189] 直接返回`HCCL_E_PTR)`；将当前查询结果/句柄交给调用者。
        return HCCL_E_PTR);
    // [中文导读] [AllReduce逐行 S190] 记录AicpuTsUrmaChannel::BuildUbMemTransport的状态/性能诊断；日志本身不执行传输。
    PLF_CONFIG_INFO(
        // [中文导读] [AllReduce逐行 S191] 为前述多行表达式补入`PLF_CHANNEL, "[AicpuTsUrmaChannel] create UbMemTransport, socket[%s], linkData[%s].",`；本行是参数/结构化初始化续行。
        PLF_CHANNEL, "[AicpuTsUrmaChannel] create UbMemTransport, socket[%s], linkData[%s].",
        // [中文导读] [AllReduce逐行 S192] 为取得对象诊断文本用于日志补入`socket_->Describe().c_str(), linkData.Describe().c_str())`（按端点对构造的链路描述的Describe字段）；本行是参数/结构化初始化续行。
        socket_->Describe().c_str(), linkData.Describe().c_str());
    // [中文导读] [AllReduce逐行 S193] 当前路径返回成功状态；仅说明本函数处理/任务组织成功，完成语义由其具体调用职责决定。
    return HCCL_SUCCESS;
// [中文导读] [AllReduce逐行 S194] 结束AicpuTsUrmaChannel::BuildUbMemTransport函数体；控制流返回外层。
}

HcclResult AicpuTsUrmaChannel::BuildSocket()
{
    if (socket_ != nullptr) {
        return HCCL_SUCCESS;
    }
    HCCL_INFO("[AicpuTsUrmaChannel][%s] socket ptr is NULL, rebuildSocket", __func__);
    Hccl::IpAddress ipaddr{};
    CHK_RET(CommAddrToIpAddress(localEp_.commAddr, ipaddr));
    Hccl::DevNetPortType type = Hccl::DevNetPortType(Hccl::ConnectProtoType::UB);
    Hccl::PortData localPort = Hccl::PortData(static_cast<Hccl::RankId>(localEp_.loc.device.devPhyId), type, 0, ipaddr);
    if (channelDesc_.role == HCOMM_SOCKET_ROLE_RESERVED) {
        Hccl::SocketHandle socketHandle
            = Hccl::SocketHandleManager::GetInstance().Create(localEp_.loc.device.devPhyId, localPort);
        EXCEPTION_CATCH(
            serverSocket_ = std::make_unique<Hccl::Socket>(
                socketHandle, ipaddr, DEFAULT_LISTENING_PORT, ipaddr, "server", Hccl::SocketRole::SERVER,
                Hccl::NicType::DEVICE_NIC_TYPE),
            return HCCL_E_PARA);
        HCCL_INFO("[AicpuTsUrmaChannel][%s] listen_socket_info[%s]", __func__, serverSocket_->Describe().c_str());
        EXCEPTION_CATCH(serverSocket_->Listen(), return HCCL_E_INTERNAL);
        Hccl::LinkData linkData = BuildDefaultLinkData();
        CHK_RET(EndpointDescPairToLinkData(localEp_, remoteEp_, linkData));
        HCCL_INFO("[AicpuTsUrmaChannel][%s] built linkData: %s", __func__, linkData.Describe().c_str());
        std::string socketTag
            = (channelDesc_.channelName != nullptr) ? std::string(channelDesc_.channelName) : "AUTOMATIC_SOCKET_TAG";
        bool noRankId = true;
        Hccl::SocketConfig socketConfig = Hccl::SocketConfig(linkData, socketTag, noRankId);
        CHK_RET(SocketMgr::GetInstance(devicePhyId_).GetSocket(socketConfig, socket_));
        socketConfigHolder_ = std::make_unique<Hccl::SocketConfig>(socketConfig);
        socketConfig_ = socketConfigHolder_.get();
    } else {
        uint16_t port = channelDesc_.port;
        if (port == 0) {
            port = DEFAULT_LISTENING_PORT;
            HCCL_INFO("[AicpuTsUrmaChannel::%s] channelDesc port is 0, use default port [%u]", __func__, port);
        }
        Hccl::LinkData linkData = BuildDefaultLinkData();
        CHK_RET(EndpointDescPairToLinkData(localEp_, remoteEp_, linkData));
        HCCL_INFO("[AicpuTsUrmaChannel][%s] built linkData: %s", __func__, linkData.Describe().c_str());
        std::string socketTag
            = (channelDesc_.channelName != nullptr) ? std::string(channelDesc_.channelName) : "AUTOMATIC_SOCKET_TAG";
        bool isServer = (channelDesc_.role == HCOMM_SOCKET_ROLE_SERVER);
        Hccl::SocketConfig socketConfig = Hccl::SocketConfig(linkData, port, socketTag, isServer);
        CHK_RET(SocketMgr::GetInstance(devicePhyId_).GetSocket(socketConfig, socket_));
        socketConfigHolder_ = std::make_unique<Hccl::SocketConfig>(socketConfig);
        socketConfig_ = socketConfigHolder_.get();
    }
    return HCCL_SUCCESS;
}

// [中文导读] [AllReduce逐行 S246] AicpuTsUrmaChannel::Init的接口声明：UB_CTP具体通道初始化：端点/监听/Socket/属性/连接/通知与Host UB传输对象；这些参数属于本函数调用边界。
HcclResult AicpuTsUrmaChannel::Init()
// [中文导读] [AllReduce逐行 S247] 进入AicpuTsUrmaChannel::Init函数体：UB_CTP具体通道初始化：端点/监听/Socket/属性/连接/通知与Host UB传输对象。
{
    /*
        Argue result: make_unique 配合一场捕获的宏 EXCEPTION CATCH
        Attention: const 和引用
    */
    // TODO: 处理抛异常
    // [中文导读] [AllReduce逐行 S253] 准备逻辑设备编号的局部存储/结构描述，初始化方式以本行声明为准。
    s32 devLogicId;
    // [中文导读] [AllReduce逐行 S254] 解析Endpoint与Hcomm描述，取得内存和驱动资源；返回非成功时由检查宏立即向上传递。
    CHK_RET(ParseInputParam());
    // [中文导读] [AllReduce逐行 S255] 读取当前运行时逻辑设备编号；返回非成功时由检查宏立即向上传递。
    CHK_RET(hrtGetDevice(&devLogicId));
    // [中文导读] [AllReduce逐行 S256] 把逻辑设备编号转换为物理设备编号；返回非成功时由检查宏立即向上传递。
    CHK_RET(hrtGetDevicePhyIdByIndex(static_cast<u32>(devLogicId), devicePhyId_));
    // [中文导读] [AllReduce逐行 S257] 在具体端点上准备监听；返回非成功时由检查宏立即向上传递。
    CHK_RET(StartListen());
    // [中文导读] [AllReduce逐行 S258] 准备具体通道的Socket对象；返回非成功时由检查宏立即向上传递。
    CHK_RET(BuildSocket());
    // [中文导读] [AllReduce逐行 S259] 形成具体通道的UB资源属性；返回非成功时由检查宏立即向上传递。
    CHK_RET(BuildAttr());
    /*
        HccpRaGetDevBaseAttr
        获取urma read/write 单个wr的最大传输数据大小
        调用前,rdmaHandle_要在ParseInputParam中被赋值好,之后BuildConnection会使用获取的属性
    */
    // [中文导读] [AllReduce逐行 S265] 外部网络适配边界：取得UB单WR最大传输等设备能力；返回非成功时由检查宏立即向上传递。
    CHK_RET(HccpRaGetDevBaseAttr(rdmaHandle_, &devBaseAttr_));
    // [中文导读] [AllReduce逐行 S266] 按协议与驱动资源创建Host UB连接；返回非成功时由检查宏立即向上传递。
    CHK_RET(BuildConnection());
    // [中文导读] [AllReduce逐行 S267] 准备通道本地/远端通知资源描述；返回非成功时由检查宏立即向上传递。
    CHK_RET(BuildNotify());
    // [中文导读] [AllReduce逐行 S268] 构造Host UbMemTransport保存连接/通知/内存交换资源；返回非成功时由检查宏立即向上传递。
    CHK_RET(BuildUbMemTransport());

    // [中文导读] [AllReduce逐行 S270] 当前路径返回成功状态；仅说明本函数处理/任务组织成功，完成语义由其具体调用职责决定。
    return HCCL_SUCCESS;
// [中文导读] [AllReduce逐行 S271] 结束AicpuTsUrmaChannel::Init函数体；控制流返回外层。
}

HcclResult AicpuTsUrmaChannel::GetNotifyNum(uint32_t* notifyNum) const
{
    *notifyNum = this->notifyNum_;
    return HCCL_SUCCESS;
}

HcclResult AicpuTsUrmaChannel::GetRemoteMems(uint32_t* memNum, CommMem** remoteMem, char*** memInfos)
{
    return memTransport_->GetRemoteMems(memNum, remoteMem, memInfos);
}

ChannelStatus AicpuTsUrmaChannel::GetStatus()
{
    ChannelStatus out = Channel::TransportStatusToChannelStatus(memTransport_->GetStatus(), localEp_, GetChannelDesc());

    if (isFirstPrintChannelInfo_ && out == ChannelStatus::READY) {
        std::string channelInfo = "create channel info:channel handle[";
        channelInfo.append(std::to_string(ReinterpretAs<uint64_t>(this)));
        channelInfo.append("] ");
        HcclResult ret = memTransport_->Describe(channelInfo);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("[AicpuTsUrmaChannel][%s] Describe channel info failed, ret=%d", __func__, ret);
            out = ChannelStatus::FAILED;
        } else {
            channelInfo.append(" TA[RM]"); // 目前TA只支持RM
            HCCL_CONFIG_DEBUG(hccl::HCCL_RES, "%s", channelInfo.c_str());
        }
        isFirstPrintChannelInfo_ = false;
    }
    return out;
}

HcclResult SetModuleDataName(Hccl::ModuleData& module, const std::string& name)
{
    int ret = strcpy_s(module.name, sizeof(module.name), name.c_str());
    if (ret != 0) {
        HCCL_ERROR("[SetModuleDataName] strcpy_s name %s failed", name.c_str());
        return HCCL_E_INTERNAL;
    }

    return HCCL_SUCCESS;
}

// [中文导读] [AllReduce逐行 S316] AicpuTsUrmaChannel::PackOpData的接口声明：将Host UB transport唯一标识打包供设备恢复传输对象；这些参数属于本函数调用边界。
HcclResult AicpuTsUrmaChannel::PackOpData(std::vector<char>& data)
// [中文导读] [AllReduce逐行 S317] 进入AicpuTsUrmaChannel::PackOpData函数体：将Host UB transport唯一标识打包供设备恢复传输对象。
{
    // [中文导读] [AllReduce逐行 S318] 准备分模块的设备资源包数组的局部存储/结构描述，初始化方式以本行声明为准。
    std::vector<Hccl::ModuleData> dataVec;
    // [中文导读] [AllReduce逐行 S319] 调用resize，使用分模块的设备资源包数组的resize字段；传入/处理分模块的设备资源包数组的resize字段。
    dataVec.resize(Hccl::AicpuResMgrType::__COUNT__);

    // [中文导读] [AllReduce逐行 S321] 设置Hccl::AicpuResMgrType resType为/按`Hccl::AicpuResMgrType::STREAM`。
    Hccl::AicpuResMgrType resType = Hccl::AicpuResMgrType::STREAM;
    // [中文导读] [AllReduce逐行 S322] 为资源模块设置供设备恢复识别的名称；返回非成功时由检查宏立即向上传递。
    CHK_RET(SetModuleDataName(dataVec[resType], "UbMemTransport"));

    // [中文导读] [AllReduce逐行 S324] 准备`std::vector<char> result`的局部存储/结构描述，初始化方式以本行声明为准。
    std::vector<char> result;
    // [中文导读] [AllReduce逐行 S325] 准备资源标识序列化/反序列化流的局部存储/结构描述，初始化方式以本行声明为准。
    Hccl::BinaryStream binaryStream;
    // [中文导读] [AllReduce逐行 S326] 序列化Host transport的设备恢复描述；传入/处理资源标识序列化/反序列化流、Host侧UB内存传输对象的GetUniqueIdV2字段。
    binaryStream << memTransport_->GetUniqueIdV2();

    // [中文导读] [AllReduce逐行 S328] 把序列化流内容输出到字节数组；传入/处理资源标识序列化/反序列化流的Dump字段。
    binaryStream.Dump(result);

    // [中文导读] [AllReduce逐行 S330] 设置分模块的设备资源包数组为/按`result`。
    dataVec[resType].data = result;

    // [中文导读] [AllReduce逐行 S332] 准备`Hccl::AicpuResPackageHelper helper`的局部存储/结构描述，初始化方式以本行声明为准。
    Hccl::AicpuResPackageHelper helper;
    // [中文导读] [AllReduce逐行 S333] 设置data为/按`helper.GetPackedData(dataVec)`（分模块的设备资源包数组）；将分模块资源内容打包为设备恢复数据。
    data = helper.GetPackedData(dataVec);

    // [中文导读] [AllReduce逐行 S335] 当前路径返回成功状态；仅说明本函数处理/任务组织成功，完成语义由其具体调用职责决定。
    return HCCL_SUCCESS;
// [中文导读] [AllReduce逐行 S336] 结束AicpuTsUrmaChannel::PackOpData函数体；控制流返回外层。
}

HcclResult AicpuTsUrmaChannel::H2DResPack(std::vector<char>& buffer)
{
    CHK_RET(PackOpData(buffer));
    HCCL_INFO(
        "[AicpuTsUrmaChannel][%s] Pack Buffer data[%p], Pack Buffer size[%zu].", __func__, buffer.data(),
        buffer.size());
    return HCCL_SUCCESS;
}

HcclResult AicpuTsUrmaChannel::Clean()
{
    PLF_CONFIG_INFO(PLF_CHANNEL, "[AicpuTsUrmaChannel] clean channel resource.");
    memTransport_.reset();
    return HCCL_SUCCESS;
}

HcclResult AicpuTsUrmaChannel::ResetLocalNotifies()
{
    for (size_t i = 0; i < localNotifies_.size(); ++i) {
        if (localNotifies_[i] == nullptr) {
            continue;
        }
        Hccl::RtsNotify* rtsNotify = localNotifies_[i]->GetNotify();
        if (rtsNotify == nullptr) {
            continue;
        }
        HcclRtNotify rtNotify = ReinterpretAs<HcclRtNotify>(rtsNotify->GetHandleAddr());
        if (rtNotify == nullptr) {
            continue;
        }
        HcclResult ret = hrtNotifyReset(rtNotify);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[AicpuTsUrmaChannel][ResetLocalNotifies] hrtNotifyReset failed, channel[%p], notifyIdx[%zu], "
                "notifyId[%u], ret[0x%016llx]",
                this, i, rtsNotify->GetId(), HCCL_ERROR_CODE(ret)),
            ret);
        HCCL_INFO(
            "[AicpuTsUrmaChannel][ResetLocalNotifies] reset notify success, channel[%p], notifyIdx[%zu], "
            "notifyId[%u], notifyPtr[%p]",
            this, i, rtsNotify->GetId(), rtNotify);
    }
    return HCCL_SUCCESS;
}

HcclResult AicpuTsUrmaChannel::Resume()
{
    CHK_RET(BuildSocket());
    CHK_RET(BuildConnection());
    CHK_RET(ResetLocalNotifies());
    CHK_RET(BuildUbMemTransport());
    return HCCL_SUCCESS;
}

HcclResult AicpuTsUrmaChannel::UpdateMemInfo(HcommMemHandle* memHandles, uint32_t memHandleNum)
{
    std::vector<Hccl::LocalRmaBuffer*> bufferVecTemp;
    CHK_RET(MakeRmaBufferVecFromMemHandles(memHandles, memHandleNum, bufferVecTemp, "AicpuTsUrmaChannel"));
    CHK_RET(memTransport_->UpdateMemInfo(bufferVecTemp));
    commonRes_.bufferVec.insert(commonRes_.bufferVec.end(), bufferVecTemp.begin(), bufferVecTemp.end());
    return HCCL_SUCCESS;
}

// 返回当前 channel 类型，供上层区分不同 channel 的能力和行为
HcommChannelKind AicpuTsUrmaChannel::GetChannelKind() const { return HcommChannelKind::AICPU_TS_URMA; }

HcclResult AicpuTsUrmaChannel::NotifyRecord([[maybe_unused]] const uint32_t remoteNotifyIdx)
{
    HCCL_INFO("[AicpuTsUrmaChannel::%s] not supported yet.", __func__);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult
AicpuTsUrmaChannel::NotifyWait([[maybe_unused]] const uint32_t localNotifyIdx, [[maybe_unused]] const uint32_t timeout)
{
    HCCL_INFO("[AicpuTsUrmaChannel::%s] not supported yet.", __func__);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult AicpuTsUrmaChannel::WriteWithNotify(
    [[maybe_unused]] void* dst, [[maybe_unused]] const void* src, [[maybe_unused]] const uint64_t len,
    [[maybe_unused]] uint32_t remoteNotifyIdx)
{
    HCCL_INFO("[AicpuTsUrmaChannel::%s] not supported yet.", __func__);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult
AicpuTsUrmaChannel::Write([[maybe_unused]] void* dst, [[maybe_unused]] const void* src, [[maybe_unused]] uint64_t len)
{
    HCCL_INFO("[AicpuTsUrmaChannel::%s] not supported yet.", __func__);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult
AicpuTsUrmaChannel::Read([[maybe_unused]] void* dst, [[maybe_unused]] const void* src, [[maybe_unused]] uint64_t len)
{
    HCCL_INFO("[AicpuTsUrmaChannel::%s] not supported yet.", __func__);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult AicpuTsUrmaChannel::ChannelFence()
{
    HCCL_INFO("[AicpuTsUrmaChannel::%s] not supported yet.", __func__);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult AicpuTsUrmaChannel::StartListen()
{
    if (channelDesc_.role != HCOMM_SOCKET_ROLE_SERVER) {
        return HCCL_SUCCESS;
    }

    uint16_t port = channelDesc_.port;
    HCCL_INFO(
        "[AicpuTsUrmaChannel::%s] Start. EndpointHandle[0x%llx], port[%u]", __func__,
        ReinterpretAs<uint64_t>(endpointHandle_), port);
    if (port == 0) {
        port = DEFAULT_LISTENING_PORT;
        HCCL_INFO("[AicpuTsUrmaChannel::%s] channelDesc port is 0, use default port [%u]", __func__, port);
    }
    CHK_RET(static_cast<HcclResult>(HcommEndpointStartListen(endpointHandle_, port, nullptr)));
    HCCL_INFO("[AicpuTsUrmaChannel::%s] SUCCESS. port[%u].", __func__, port);
    return HCCL_SUCCESS;
}

} // namespace hcomm
