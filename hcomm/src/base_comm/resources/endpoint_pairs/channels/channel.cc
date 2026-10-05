/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <string>
#include <unordered_map>

#include "log.h"
#include "channel.h"
#include "config_plf_log_v2.h"
#include "orion_adpt_utils.h"
#include "./aicpu/aicpu_ts_urma_channel.h"
#include "comm_engine_utils.h"
#include "protocol_utils.h"

#include "./aicpu/aicpu_ts_p2p_channel.h"
#include "./aicpu/aicpu_ts_uboe_channel.h"
#include "./aicpu/aicpu_ts_ub_rtp_channel.h"
#include "./aicpu/aicpu_ts_roce_channel.h"
#include "./host/host_cpu_roce_channel.h"
#include "./host/host_cpu_urma_channel.h"
#include "./ccu/ccu_urma_channel.h"
#include "./aiv/aiv_ub_mem_channel.h"
#include "./aiv/aiv_urma_channel.h"
#include "./aicpu/aicpu_ts_hccs_channel.h"
#include "./aicpu/aicpu_ts_roce_channel_v2.h"

namespace hcomm {
using Hccl::PLF_CHANNEL;
std::unordered_map<ChannelHandle, ChannelHandle> channelD2HHandleMap_;

// [中文导读] [AllReduce逐行 S38] Channel::CreateChannel的接口声明：按 CommEngine 与 remoteEndpoint.protocol 选择具体通道类并调用初始化；保留所有引擎/协议分支；这些参数属于本函数调用边界。
HcclResult Channel::CreateChannel(
    // [中文导读] [AllReduce逐行 S39] Channel::CreateChannel的接口声明：本地端点句柄、请求的通信引擎、当前通道描述、具体对象/PI出参；这些参数属于本函数调用边界。
    EndpointHandle endpointHandle, CommEngine engine, HcommChannelDesc channelDesc, std::shared_ptr<Channel>& out,
    // [中文导读] [AllReduce逐行 S40] Channel::CreateChannel的接口声明：是否启用共享队列；这些参数属于本函数调用边界。
    bool isSharedQueue)
// [中文导读] [AllReduce逐行 S41] 进入Channel::CreateChannel函数体：按 CommEngine 与 remoteEndpoint.protocol 选择具体通道类并调用初始化；保留所有引擎/协议分支。
{
    // [中文导读] [AllReduce逐行 S42] 设置设备型号为/按`DevType::DEV_TYPE_COUNT`。
    DevType deviceType = DevType::DEV_TYPE_COUNT;
    // [中文导读] [AllReduce逐行 S43] 读取设备型号用于新旧/协议分支选择；返回非成功时由检查宏立即向上传递。
    CHK_RET(hrtGetDeviceType(deviceType));
    // [中文导读] [AllReduce逐行 S44] 准备已选择的具体Channel对象的局部存储/结构描述，初始化方式以本行声明为准。
    std::shared_ptr<Channel> uniqueChannelPtr;
    // [中文导读] [AllReduce逐行 S45] 以`engine`（请求的通信引擎）选择后续互斥处理路径。
    switch (engine) {
        // [中文导读] [AllReduce逐行 S46] 匹配`COMM_ENGINE_CPU`的枚举路径；继续执行本case中的操作。
        case COMM_ENGINE_CPU:
            // [中文导读] [AllReduce逐行 S47] 仅当`(channelDesc.remoteEndpoint.protocol == COMM_PROTOCOL_ROCE)`（当前通道描述的remoteEndpoint.protocol字段）成立时进入此分支。
            if (channelDesc.remoteEndpoint.protocol == COMM_PROTOCOL_ROCE) {
                // [中文导读] [AllReduce逐行 S48] 在异常捕获边界执行后续表达式；异常按后续处理语句转换成HCCL状态或提前返回。
                EXCEPTION_CATCH(
                    // [中文导读] [AllReduce逐行 S49] 为前述多行表达式补入`uniqueChannelPtr = std::make_unique<HostCpuRoceChannel>(endpointHandle, channelDesc),`（已选择的具体Channel对象、本地端点句柄、当前通道描述）；本行是参数/结构化初始化续行。
                    uniqueChannelPtr = std::make_unique<HostCpuRoceChannel>(endpointHandle, channelDesc),
                    // [中文导读] [AllReduce逐行 S50] 直接返回`HCCL_E_PARA)`；将当前查询结果/句柄交给调用者。
                    return HCCL_E_PARA);
                // [中文导读] [AllReduce逐行 S51] 退出当前循环或switch路径，继续其后处理；不代表其他执行流已经完成。
                break;
            // [中文导读] [AllReduce逐行 S52] 结束`if (channelDesc.remoteEndpoint.protocol == COMM_PROTOCOL_ROCE)`（当前通道描述的remoteEndpoint.protocol字段）分支/循环；控制流返回外层。
            }
            // [中文导读] [AllReduce逐行 S53] 仅当`(channelDesc.remoteEndpoint.protocol == COMM_PROTOCOL_UB_CTP`（当前通道描述的remoteEndpoint.protocol字段）成立时进入此分支。
            if (channelDesc.remoteEndpoint.protocol == COMM_PROTOCOL_UB_CTP
                // [中文导读] [AllReduce逐行 S54] 补全本分支/循环判断的`|| channelDesc.remoteEndpoint.protocol == COMM_PROTOCOL_UBC_TP)`（当前通道描述的remoteEndpoint.protocol字段），和前面条件共同决定是否进入后续路径。
                || channelDesc.remoteEndpoint.protocol == COMM_PROTOCOL_UBC_TP) {
                // [中文导读] [AllReduce逐行 S55] 在异常捕获边界执行后续表达式；异常按后续处理语句转换成HCCL状态或提前返回。
                EXCEPTION_CATCH(
                    // [中文导读] [AllReduce逐行 S56] 为前述多行表达式补入`uniqueChannelPtr = std::make_unique<HostCpuUrmaChannel>(endpointHandle, channelDesc),`（已选择的具体Channel对象、本地端点句柄、当前通道描述）；本行是参数/结构化初始化续行。
                    uniqueChannelPtr = std::make_unique<HostCpuUrmaChannel>(endpointHandle, channelDesc),
                    // [中文导读] [AllReduce逐行 S57] 直接返回`HCCL_E_PARA)`；将当前查询结果/句柄交给调用者。
                    return HCCL_E_PARA);
                // [中文导读] [AllReduce逐行 S58] 退出当前循环或switch路径，继续其后处理；不代表其他执行流已经完成。
                break;
            // [中文导读] [AllReduce逐行 S59] 结束当前局部作用域；控制流返回外层。
            }
            // [中文导读] [AllReduce逐行 S60] 记录Channel::CreateChannel的错误诊断；日志本身不执行传输。
            HCCL_ERROR(
                // [中文导读] [AllReduce逐行 S61] 为当前Channel::CreateChannel诊断/异常表达式提供格式文本；这一物理行没有数据搬运副作用。
                "[Channel][%s] Engine[COMM_ENGINE_CPU] not support Protocol[%d]", __func__,
                // [中文导读] [AllReduce逐行 S62] 为前述多行表达式补入`channelDesc.remoteEndpoint.protocol)`（当前通道描述的remoteEndpoint.protocol字段）；本行是参数/结构化初始化续行。
                channelDesc.remoteEndpoint.protocol);
            // [中文导读] [AllReduce逐行 S63] 返回HCCL_E_NOT_SUPPORT，表示当前引擎/设备/协议不支持此路径；此路径停止本函数的后续处理。
            return HCCL_E_NOT_SUPPORT;
        // [中文导读] [AllReduce逐行 S64] 匹配`COMM_ENGINE_CPU_TS`的枚举路径；继续执行本case中的操作。
        case COMM_ENGINE_CPU_TS:
            // [中文导读] [AllReduce逐行 S65] 记录Channel::CreateChannel的错误诊断；日志本身不执行传输。
            HCCL_ERROR("[Channel][%s] CommEngine[COMM_ENGINE_CPU_TS] not support", __func__);
            // [中文导读] [AllReduce逐行 S66] 返回HCCL_E_NOT_SUPPORT，表示当前引擎/设备/协议不支持此路径；此路径停止本函数的后续处理。
            return HCCL_E_NOT_SUPPORT;
        // [中文导读] [AllReduce逐行 S67] 匹配`COMM_ENGINE_AICPU`的枚举路径；继续执行本case中的操作。
        case COMM_ENGINE_AICPU:
        // [中文导读] [AllReduce逐行 S68] 匹配`COMM_ENGINE_AICPU_TS`的枚举路径；继续执行本case中的操作。
        case COMM_ENGINE_AICPU_TS:
            // [中文导读] [AllReduce逐行 S69] 仅当`(channelDesc.remoteEndpoint.protocol == COMM_PROTOCOL_UBOE)`（当前通道描述的remoteEndpoint.protocol字段）成立时进入此分支。
            if (channelDesc.remoteEndpoint.protocol == COMM_PROTOCOL_UBOE) {
                // [中文导读] [AllReduce逐行 S70] 调用reset, new, AicpuTsUboeChannel，使用已选择的具体Channel对象的reset字段、本地端点句柄、当前通道描述；传入/处理已选择的具体Channel对象的reset字段、本地端点句柄、当前通道描述。
                uniqueChannelPtr.reset(new (std::nothrow) AicpuTsUboeChannel(endpointHandle, channelDesc));
            // [中文导读] [AllReduce逐行 S71] 仅当`(channelDesc.remoteEndpoint.protocol == COMM_PROTOCOL_UB_RTP)`（当前通道描述的remoteEndpoint.protocol字段）成立时进入此分支。
            } else if (channelDesc.remoteEndpoint.protocol == COMM_PROTOCOL_UB_RTP) {
                // [中文导读] [AllReduce逐行 S72] 仅当`(deviceType != DevType::DEV_TYPE_950 && deviceType != DevType::DEV_TYPE_960)`（设备型号）成立时进入此分支。
                if (deviceType != DevType::DEV_TYPE_950 && deviceType != DevType::DEV_TYPE_960) {
                    // [中文导读] [AllReduce逐行 S73] 记录Channel::CreateChannel的错误诊断；日志本身不执行传输。
                    HCCL_ERROR(
                        // [中文导读] [AllReduce逐行 S74] 为当前Channel::CreateChannel诊断/异常表达式提供格式文本；这一物理行没有数据搬运副作用。
                        "[Channel][%s] UB_RTP protocol only support DEV_TYPE_950/960, current deviceType=%d", __func__,
                        // [中文导读] [AllReduce逐行 S75] 为前述多行表达式补入`static_cast<int>(deviceType))`（设备型号）；本行是参数/结构化初始化续行。
                        static_cast<int>(deviceType));
                    // [中文导读] [AllReduce逐行 S76] 返回HCCL_E_NOT_SUPPORT，表示当前引擎/设备/协议不支持此路径；此路径停止本函数的后续处理。
                    return HCCL_E_NOT_SUPPORT;
                // [中文导读] [AllReduce逐行 S77] 结束`if (deviceType != DevType::DEV_TYPE_950 && deviceType != DevType::DEV_TYPE_960)`（设备型号）分支/循环；控制流返回外层。
                }
                // [中文导读] [AllReduce逐行 S78] 调用reset, new, AicpuTsUbRtpChannel，使用已选择的具体Channel对象的reset字段、本地端点句柄、当前通道描述；传入/处理已选择的具体Channel对象的reset字段、本地端点句柄、当前通道描述。
                uniqueChannelPtr.reset(new (std::nothrow) AicpuTsUbRtpChannel(endpointHandle, channelDesc));
            // [中文导读] [AllReduce逐行 S79] 仅当`(channelDesc.remoteEndpoint.protocol == COMM_PROTOCOL_PCIE)`（当前通道描述的remoteEndpoint.protocol字段）成立时进入此分支。
            } else if (channelDesc.remoteEndpoint.protocol == COMM_PROTOCOL_PCIE) {
                // [中文导读] [AllReduce逐行 S80] 调用reset, new, AicpuTsP2pChannel，使用已选择的具体Channel对象的reset字段、本地端点句柄、当前通道描述；传入/处理已选择的具体Channel对象的reset字段、本地端点句柄、当前通道描述。
                uniqueChannelPtr.reset(new (std::nothrow) AicpuTsP2pChannel(endpointHandle, channelDesc));
            // [中文导读] [AllReduce逐行 S81] 仅当`(channelDesc.remoteEndpoint.protocol == COMM_PROTOCOL_ROCE)`（当前通道描述的remoteEndpoint.protocol字段）成立时进入此分支。
            } else if (channelDesc.remoteEndpoint.protocol == COMM_PROTOCOL_ROCE) {
                // [中文导读] [AllReduce逐行 S82] 仅当`(deviceType == DevType::DEV_TYPE_950 || deviceType == DevType::DEV_TYPE_960)`（设备型号）成立时进入此分支。
                if (deviceType == DevType::DEV_TYPE_950 || deviceType == DevType::DEV_TYPE_960) {
                    // [中文导读] [AllReduce逐行 S83] 设置已选择的具体Channel对象为/按`std::make_unique<AicpuTsRoceChannelV2>(endpointHandle, channelDesc, engine)`（本地端点句柄、当前通道描述、请求的通信引擎）。
                    uniqueChannelPtr = std::make_unique<AicpuTsRoceChannelV2>(endpointHandle, channelDesc, engine);
                // [中文导读] [AllReduce逐行 S84] 进入上一个条件未命中的替代路径，按本函数的资源/设备/协议分流继续处理。
                } else {
                    // [中文导读] [AllReduce逐行 S85] 设置已选择的具体Channel对象为/按`std::make_unique<AicpuTsRoceChannel>(endpointHandle, channelDesc)`（本地端点句柄、当前通道描述）。
                    uniqueChannelPtr = std::make_unique<AicpuTsRoceChannel>(endpointHandle, channelDesc);
                // [中文导读] [AllReduce逐行 S86] 结束`if (deviceType == DevType::DEV_TYPE_950 || deviceType == DevType::DEV_TYPE_960)`（设备型号）分支/循环；控制流返回外层。
                }
            // [中文导读] [AllReduce逐行 S87] 仅当`(`成立时进入此分支。
            } else if (
                // [中文导读] [AllReduce逐行 S88] 补全本分支/循环判断的`channelDesc.remoteEndpoint.protocol == COMM_PROTOCOL_UB_CTP`（当前通道描述的remoteEndpoint.protocol字段），和前面条件共同决定是否进入后续路径。
                channelDesc.remoteEndpoint.protocol == COMM_PROTOCOL_UB_CTP
                // [中文导读] [AllReduce逐行 S89] 补全本分支/循环判断的`|| channelDesc.remoteEndpoint.protocol == COMM_PROTOCOL_UBC_TP)`（当前通道描述的remoteEndpoint.protocol字段），和前面条件共同决定是否进入后续路径。
                || channelDesc.remoteEndpoint.protocol == COMM_PROTOCOL_UBC_TP) {
                // [中文导读] [AllReduce逐行 S90] 调用reset, new, AicpuTsUrmaChannel，使用已选择的具体Channel对象的reset字段、本地端点句柄、当前通道描述；传入/处理已选择的具体Channel对象的reset字段、本地端点句柄、当前通道描述。
                uniqueChannelPtr.reset(new (std::nothrow) AicpuTsUrmaChannel(endpointHandle, channelDesc));
            // [中文导读] [AllReduce逐行 S91] 仅当`(channelDesc.remoteEndpoint.protocol == COMM_PROTOCOL_HCCS)`（当前通道描述的remoteEndpoint.protocol字段）成立时进入此分支。
            } else if (channelDesc.remoteEndpoint.protocol == COMM_PROTOCOL_HCCS) {
                // [中文导读] [AllReduce逐行 S92] 调用reset, new, AicpuTsHccsChannel，使用已选择的具体Channel对象的reset字段、本地端点句柄、当前通道描述；传入/处理已选择的具体Channel对象的reset字段、本地端点句柄、当前通道描述。
                uniqueChannelPtr.reset(new (std::nothrow) AicpuTsHccsChannel(endpointHandle, channelDesc));
            // [中文导读] [AllReduce逐行 S93] 进入上一个条件未命中的替代路径，按本函数的资源/设备/协议分流继续处理。
            } else {
                // [中文导读] [AllReduce逐行 S94] 记录Channel::CreateChannel的错误诊断；日志本身不执行传输。
                HCCL_ERROR(
                    // [中文导读] [AllReduce逐行 S95] 为当前Channel::CreateChannel诊断/异常表达式提供格式文本；这一物理行没有数据搬运副作用。
                    "[Channel][%s] invalid protocol for engine[%s], protocol[%s]", __func__,
                    // [中文导读] [AllReduce逐行 S96] 为把枚举转换成诊断名称补入`GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str(),`（请求的通信引擎）；本行是参数/结构化初始化续行。
                    GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str(),
                    // [中文导读] [AllReduce逐行 S97] 为把枚举转换成诊断名称补入`GetEnumToString(GetCommProtocolStrMap(), channelDesc.remoteEndpoint.protocol).c_str())`（当前通道描述的remoteEndpoint.protocol字段）；本行是参数/结构化初始化续行。
                    GetEnumToString(GetCommProtocolStrMap(), channelDesc.remoteEndpoint.protocol).c_str());
                // [中文导读] [AllReduce逐行 S98] 返回HCCL_E_NOT_SUPPORT，表示当前引擎/设备/协议不支持此路径；此路径停止本函数的后续处理。
                return HCCL_E_NOT_SUPPORT;
            // [中文导读] [AllReduce逐行 S99] 结束当前局部作用域；控制流返回外层。
            }
            // [中文导读] [AllReduce逐行 S100] 退出当前循环或switch路径，继续其后处理；不代表其他执行流已经完成。
            break;
        // [中文导读] [AllReduce逐行 S101] 匹配`COMM_ENGINE_AIV`的枚举路径；继续执行本case中的操作。
        case COMM_ENGINE_AIV:
            // [中文导读] [AllReduce逐行 S102] 仅当`(channelDesc.remoteEndpoint.protocol == COMM_PROTOCOL_ROCE`（当前通道描述的remoteEndpoint.protocol字段）成立时进入此分支。
            if (channelDesc.remoteEndpoint.protocol == COMM_PROTOCOL_ROCE
                // [中文导读] [AllReduce逐行 S103] 补全本分支/循环判断的`&& (deviceType == DevType::DEV_TYPE_950 || deviceType == DevType::DEV_TYPE_960))`（设备型号），和前面条件共同决定是否进入后续路径。
                && (deviceType == DevType::DEV_TYPE_950 || deviceType == DevType::DEV_TYPE_960)) {
                // [中文导读] [AllReduce逐行 S104] 设置已选择的具体Channel对象为/按`std::make_unique<AicpuTsRoceChannelV2>(endpointHandle, channelDesc, engine)`（本地端点句柄、当前通道描述、请求的通信引擎）。
                uniqueChannelPtr = std::make_unique<AicpuTsRoceChannelV2>(endpointHandle, channelDesc, engine);
            // [中文导读] [AllReduce逐行 S105] 仅当`(`成立时进入此分支。
            } else if (
                // [中文导读] [AllReduce逐行 S106] 补全本分支/循环判断的`channelDesc.remoteEndpoint.protocol == COMM_PROTOCOL_UB_CTP`（当前通道描述的remoteEndpoint.protocol字段），和前面条件共同决定是否进入后续路径。
                channelDesc.remoteEndpoint.protocol == COMM_PROTOCOL_UB_CTP
                // [中文导读] [AllReduce逐行 S107] 补全本分支/循环判断的`|| channelDesc.remoteEndpoint.protocol == COMM_PROTOCOL_UBC_TP`（当前通道描述的remoteEndpoint.protocol字段），和前面条件共同决定是否进入后续路径。
                || channelDesc.remoteEndpoint.protocol == COMM_PROTOCOL_UBC_TP
                // [中文导读] [AllReduce逐行 S108] 补全本分支/循环判断的`|| channelDesc.remoteEndpoint.protocol == COMM_PROTOCOL_UB_RTP)`（当前通道描述的remoteEndpoint.protocol字段），和前面条件共同决定是否进入后续路径。
                || channelDesc.remoteEndpoint.protocol == COMM_PROTOCOL_UB_RTP) {
                // [中文导读] [AllReduce逐行 S109] 仅当`(channelDesc.remoteEndpoint.protocol == COMM_PROTOCOL_UB_RTP && deviceType != DevType::DEV_TYPE_950`（当前通道描述的remoteEndpoint.protocol字段、设备型号）成立时进入此分支。
                if (channelDesc.remoteEndpoint.protocol == COMM_PROTOCOL_UB_RTP && deviceType != DevType::DEV_TYPE_950
                    // [中文导读] [AllReduce逐行 S110] 补全本分支/循环判断的`&& deviceType != DevType::DEV_TYPE_960)`（设备型号），和前面条件共同决定是否进入后续路径。
                    && deviceType != DevType::DEV_TYPE_960) {
                    // [中文导读] [AllReduce逐行 S111] 记录Channel::CreateChannel的错误诊断；日志本身不执行传输。
                    HCCL_ERROR(
                        // [中文导读] [AllReduce逐行 S112] 为当前Channel::CreateChannel诊断/异常表达式提供格式文本；这一物理行没有数据搬运副作用。
                        "[Channel][%s] UB_RTP protocol only support DEV_TYPE_950/960, current deviceType=%d", __func__,
                        // [中文导读] [AllReduce逐行 S113] 为前述多行表达式补入`static_cast<int>(deviceType))`（设备型号）；本行是参数/结构化初始化续行。
                        static_cast<int>(deviceType));
                    // [中文导读] [AllReduce逐行 S114] 返回HCCL_E_NOT_SUPPORT，表示当前引擎/设备/协议不支持此路径；此路径停止本函数的后续处理。
                    return HCCL_E_NOT_SUPPORT;
                // [中文导读] [AllReduce逐行 S115] 结束当前局部作用域；控制流返回外层。
                }
                // [中文导读] [AllReduce逐行 S116] 调用reset, new, AivUrmaChannel，使用已选择的具体Channel对象的reset字段、本地端点句柄、当前通道描述；传入/处理已选择的具体Channel对象的reset字段、本地端点句柄、当前通道描述。
                uniqueChannelPtr.reset(new (std::nothrow) AivUrmaChannel(endpointHandle, channelDesc));
            // [中文导读] [AllReduce逐行 S117] 进入上一个条件未命中的替代路径，按本函数的资源/设备/协议分流继续处理。
            } else {
                // [中文导读] [AllReduce逐行 S118] 调用reset, new, AivUbMemChannel，使用已选择的具体Channel对象的reset字段、本地端点句柄、当前通道描述；传入/处理已选择的具体Channel对象的reset字段、本地端点句柄、当前通道描述。
                uniqueChannelPtr.reset(new (std::nothrow) AivUbMemChannel(endpointHandle, channelDesc));
            // [中文导读] [AllReduce逐行 S119] 结束当前局部作用域；控制流返回外层。
            }
            // [中文导读] [AllReduce逐行 S120] 退出当前循环或switch路径，继续其后处理；不代表其他执行流已经完成。
            break;
        // [中文导读] [AllReduce逐行 S121] 匹配`COMM_ENGINE_CCU`的枚举路径；继续执行本case中的操作。
        case COMM_ENGINE_CCU:
            // [中文导读] [AllReduce逐行 S122] 调用reset, new, CcuUrmaChannel，使用已选择的具体Channel对象的reset字段、本地端点句柄、当前通道描述；传入/处理已选择的具体Channel对象的reset字段、本地端点句柄、当前通道描述。
            uniqueChannelPtr.reset(new (std::nothrow) CcuUrmaChannel(endpointHandle, channelDesc));
            // [中文导读] [AllReduce逐行 S123] 退出当前循环或switch路径，继续其后处理；不代表其他执行流已经完成。
            break;
        // [中文导读] [AllReduce逐行 S124] 处理switch中没有明确匹配的枚举值，具体返回/回退行为由以下代码决定。
        default:
            // [中文导读] [AllReduce逐行 S125] 记录Channel::CreateChannel的错误诊断；日志本身不执行传输。
            HCCL_ERROR("[Channel][%s] invalid type of CommEngine", __func__);
            // [中文导读] [AllReduce逐行 S126] 返回HCCL_E_NOT_FOUND，表示没有找到对应资源/实现；此路径停止本函数的后续处理。
            return HCCL_E_NOT_FOUND;
    // [中文导读] [AllReduce逐行 S127] 结束`switch (engine)`（请求的通信引擎）分支/循环；控制流返回外层。
    }
    // [中文导读] [AllReduce逐行 S128] 检查`uniqueChannelPtr`（已选择的具体Channel对象）不是空对象；宏命中失败条件时立即返回对应指针错误。
    CHK_PTR_NULL(uniqueChannelPtr);
    // [中文导读] [AllReduce逐行 S129] 设置已选择的具体Channel对象的engine_字段为/按`engine`（请求的通信引擎）。
    uniqueChannelPtr->engine_ = engine;
    // [中文导读] [AllReduce逐行 S130] 保存是否共享jetty的标记；传入/处理已选择的具体Channel对象的SetSharedJetty字段、是否启用共享队列。
    uniqueChannelPtr->SetSharedJetty(isSharedQueue);
    // [中文导读] [AllReduce逐行 S131] 初始化已选择的具体通道资源；返回非成功时由检查宏立即向上传递，UNAVAIL资源不足状态保持可识别。
    CHK_RET_UNAVAIL(uniqueChannelPtr->Init());
    // [中文导读] [AllReduce逐行 S132] 设置具体对象/PI出参为/按`std::move(uniqueChannelPtr)`（已选择的具体Channel对象）；调用std::move，使用已选择的具体Channel对象。
    out = std::move(uniqueChannelPtr);
    // [中文导读] [AllReduce逐行 S133] 当前路径返回成功状态；仅说明本函数处理/任务组织成功，完成语义由其具体调用职责决定。
    return HCCL_SUCCESS;
// [中文导读] [AllReduce逐行 S134] 结束Channel::CreateChannel函数体；控制流返回外层。
}

ChannelStatus Channel::TransportStatusToChannelStatus(
    Hccl::TransportStatus ts, const EndpointDesc& localEp, const HcommChannelDesc& channelDesc)
{
    const EndpointDesc& remoteEp = channelDesc.remoteEndpoint;

    if (Hccl::GetPlfDebugConfigValue() & PLF_CHANNEL) {
        Hccl::IpAddress localAddr{};
        std::string localEid = "invalid";
        if (CommAddrToIpAddress(localEp.commAddr, localAddr) == HCCL_SUCCESS) {
            localEid = localAddr.Describe();
        }

        Hccl::IpAddress remoteAddr{};
        std::string remoteEid = "invalid";
        if (CommAddrToIpAddress(remoteEp.commAddr, remoteAddr) == HCCL_SUCCESS) {
            remoteEid = remoteAddr.Describe();
        }

        const char* socketTag = channelDesc.channelName != nullptr ? channelDesc.channelName : "anonymous";

        PLF_CONFIG_INFO(
            PLF_CHANNEL, "status[%d], protocol[%s], localEid[%s], remoteEid[%s], socketTag[%s].", static_cast<int>(ts),
            GetEnumToString(GetCommProtocolStrMap(), remoteEp.protocol).c_str(), localEid.c_str(), remoteEid.c_str(),
            socketTag);
    }
    switch (ts) {
        case Hccl::TransportStatus::INIT:
            return ChannelStatus::INIT;
        case Hccl::TransportStatus::SOCKET_OK:
            return ChannelStatus::SOCKET_OK;
        case Hccl::TransportStatus::SOCKET_TIMEOUT:
            return ChannelStatus::SOCKET_TIMEOUT;
        case Hccl::TransportStatus::READY:
            return ChannelStatus::READY;
        default:
            HCCL_ERROR("[Channel][%s] Invalid TransportStatus[%d]", __func__, ts);
            return ChannelStatus::FAILED;
    }
}

HcclResult Channel::UpdateMemInfo([[maybe_unused]] HcommMemHandle* memHandles, [[maybe_unused]] uint32_t memHandleNum)
{
    HCCL_WARNING("[UpdateMemInfo] not support.");
    return HCCL_SUCCESS;
}

HcommChannelKind Channel::GetChannelKind() const { return channelKind_; }

HcclResult Channel::Serialize(std::shared_ptr<hccl::DeviceMem>& out)
{
    out.reset();
    return HCCL_E_NOT_SUPPORT;
}

void Channel::AddPtrArrayDevMem(std::shared_ptr<hccl::DeviceMem> ptrArrayMem)
{
    if (ptrArrayMem == nullptr || !(*ptrArrayMem)) {
        HCCL_WARNING("[Channel][%s] invalid ptrArrayMem.", __func__);
        return;
    }
    ptrArrayDevMems_.push_back(std::move(ptrArrayMem));
}

void Channel::ReleasePtrArrayDevMems() { ptrArrayDevMems_.clear(); }
} // namespace hcomm
