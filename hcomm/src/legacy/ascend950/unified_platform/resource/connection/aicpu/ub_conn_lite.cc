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
#include <chrono>
#include "ub_conn_lite.h"
#include "log.h"
#include "exception_util.h"
#include "udma_data_struct.h"
#include "internal_exception.h"
#include "string_util.h"
#include "binary_stream.h"
#include "data_type.h"

constexpr u32 MAX_LOG_TIMEOUT_MS = 500;
namespace Hccl {
constexpr u32 ADDR_BIT_OFFSET = 32;
constexpr u32 SQE_SIZE_128 = 128;
constexpr u32 SQE_SIZE_64 = 64;
constexpr u32 SQE_INLINE_DATA_SIZE = 16;
constexpr u32 RAW_SIZE = 16;
constexpr u32 RMT_EID_BYTE_SIZE = 16;
constexpr u32 PI_NUM_TWO = 2;
constexpr u32 WRITE_WITH_NOTIFY_OPCODE = 0x5; // 注意: 与aicpu_task_cache_entry.cc保持一致
constexpr u32 ADDR_BIT_LOW = 0xffffffff;
constexpr u32 UB_DMA_MAX_READ_WEITE_SIZE = 256 * 1024 * 1024; // Byte, UB协议一次传输的最大size
constexpr u32 UB_RELAX_ORDER = 0x1; // Relax Order表示当前SQE与后续Strong Order SQE有保序要求
constexpr u32 UB_STRONG_ORDER = 0x2; // Strong Order表示当前SQE有保序要求，该SQE不能超越前面的Relax Order SQE

static std::map<DataType, u32> g_ubmaDataTypeMap
    = {{DataType::INT8, 0x0},   {DataType::INT16, 0x1},   {DataType::INT32, 0x2}, {DataType::UINT8, 0x3},
       {DataType::UINT16, 0x4}, {DataType::UINT32, 0x5},  {DataType::FP16, 0x6},  {DataType::FP32, 0x7},
       {DataType::BFP16, 0x8},  {DataType::BF16_SAT, 0x9}};

static std::map<ReduceOp, u32> g_ubmaDataOpMap = {{ReduceOp::SUM, 0xA}, {ReduceOp::MAX, 0x8}, {ReduceOp::MIN, 0x9}};

// [中文导读] [AllReduce逐行 S44] UbConnLite::FillCommSqe的接口声明：按读写操作、切片位置和用户配置填写UB公共WQE字段；这些参数属于本函数调用边界。
void UbConnLite::FillCommSqe(
    // [中文导读] [AllReduce逐行 S45] UbConnLite::FillCommSqe的接口声明：当前UB WQE结构、远端缓冲区/地址、UB WQE保序/完成配置、UB读写操作码、当前分片首/中/尾位置；这些参数属于本函数调用边界。
    UdmaSqeCommon* sqe, const RmtRmaBufSliceLite& rmt, const SqeConfigLite& cfg, u32 opCode, SlicePosition slicePos)
// [中文导读] [AllReduce逐行 S46] 进入UbConnLite::FillCommSqe函数体：按读写操作、切片位置和用户配置填写UB公共WQE字段。
{
    // [中文导读] [AllReduce逐行 S47] 设置u32 cqeEn为/按`(cfg.cqeEn && (slicePos == SlicePosition::LAST || slicePos == SlicePosition::ONLY)) ? 1 : 0`（UB WQE保序/完成配置的cqeEn字段、当前分片首/中/尾位置）。
    u32 cqeEn = (cfg.cqeEn && (slicePos == SlicePosition::LAST || slicePos == SlicePosition::ONLY)) ? 1 : 0;
    // [中文导读] [AllReduce逐行 S48] 设置当前UB WQE结构的cqe字段为/按`cqeEn`。
    sqe->cqe = cqeEn;
    // [中文导读] [AllReduce逐行 S49] 设置当前UB WQE结构的owner字段为/按`(pi == (sqDepth_ - 1)) ? 1 : 0`（UB jetty生产指针（16位自然增长）、RTSQ或UB SQ深度）。
    sqe->owner = (pi == (sqDepth_ - 1)) ? 1 : 0;
    // [中文导读] [AllReduce逐行 S50] 设置当前UB WQE结构的opcode字段为/按`opCode`（UB读写操作码）。
    sqe->opcode = opCode;
    // [中文导读] [AllReduce逐行 S51] 设置当前UB WQE结构的tpn字段为/按`tpn_`。
    sqe->tpn = tpn_;

    // [中文导读] [AllReduce逐行 S53] 仅当`(cfg.userConfig)`（UB WQE保序/完成配置的userConfig字段）成立时进入此分支。
    if (cfg.userConfig) {
        // [中文导读] [AllReduce逐行 S54] 设置当前UB WQE结构的placeOdr字段为/按`cfg.placeOdr`（UB WQE保序/完成配置的placeOdr字段）。
        sqe->placeOdr = cfg.placeOdr;
        // [中文导读] [AllReduce逐行 S55] 设置当前UB WQE结构的compOrder字段为/按`cfg.compOrder`（UB WQE保序/完成配置的compOrder字段）。
        sqe->compOrder = cfg.compOrder;
        // [中文导读] [AllReduce逐行 S56] 设置当前UB WQE结构的fence字段为/按`cfg.fence`（UB WQE保序/完成配置的fence字段）。
        sqe->fence = cfg.fence;
    // [中文导读] [AllReduce逐行 S57] 进入上一个条件未命中的替代路径，按本函数的资源/设备/协议分流继续处理。
    } else {
        // 当前片是ONLY片(只有一片的情况)和最后一片的情况，全严格保序
        // [中文导读] [AllReduce逐行 S59] 仅当`(slicePos == SlicePosition::ONLY || slicePos == SlicePosition::LAST)`（当前分片首/中/尾位置）成立时进入此分支。
        if (slicePos == SlicePosition::ONLY || slicePos == SlicePosition::LAST) {
            // [中文导读] [AllReduce逐行 S60] 设置当前UB WQE结构的placeOdr字段为/按`UB_STRONG_ORDER`。
            sqe->placeOdr = UB_STRONG_ORDER;
            // [中文导读] [AllReduce逐行 S61] 设置当前UB WQE结构的compOrder字段为/按`1`。
            sqe->compOrder = 1;
            // [中文导读] [AllReduce逐行 S62] 设置当前UB WQE结构的fence字段为/按`1`。
            sqe->fence = 1;
        // [中文导读] [AllReduce逐行 S63] 进入上一个条件未命中的替代路径，按本函数的资源/设备/协议分流继续处理。
        } else {
            // 中间片写死配置，第一片由全局cfg配置
            // [中文导读] [AllReduce逐行 S65] 设置当前UB WQE结构的placeOdr字段为/按`(slicePos == SlicePosition::MIDDLE) ? UB_RELAX_ORDER : cfg.placeOdr`（当前分片首/中/尾位置、UB WQE保序/完成配置的placeOdr字段）。
            sqe->placeOdr = (slicePos == SlicePosition::MIDDLE) ? UB_RELAX_ORDER : cfg.placeOdr;
            // [中文导读] [AllReduce逐行 S66] 设置当前UB WQE结构的compOrder字段为/按`(slicePos == SlicePosition::MIDDLE) ? 0 : cfg.compOrder`（当前分片首/中/尾位置、UB WQE保序/完成配置的compOrder字段）。
            sqe->compOrder = (slicePos == SlicePosition::MIDDLE) ? 0 : cfg.compOrder;
            // [中文导读] [AllReduce逐行 S67] 设置当前UB WQE结构的fence字段为/按`(slicePos == SlicePosition::MIDDLE) ? 0 : cfg.fence`（当前分片首/中/尾位置、UB WQE保序/完成配置的fence字段）。
            sqe->fence = (slicePos == SlicePosition::MIDDLE) ? 0 : cfg.fence;
        // [中文导读] [AllReduce逐行 S68] 结束`if (slicePos == SlicePosition::ONLY || slicePos == SlicePosition::LAST)`（当前分片首/中/尾位置）分支/循环；控制流返回外层。
        }
    // [中文导读] [AllReduce逐行 S69] 结束`if (cfg.userConfig)`（UB WQE保序/完成配置的userConfig字段）分支/循环；控制流返回外层。
    }

    // [中文导读] [AllReduce逐行 S71] 设置当前UB WQE结构的se字段为/按`1`。
    sqe->se = 1;           // 表示是否使能solicited event
    // [中文导读] [AllReduce逐行 S72] 设置当前UB WQE结构的rmtJettyType字段为/按`1`。
    sqe->rmtJettyType = 1; // 00 JFR  01:JETTY  10:jettyGroup 11:reserved
    // [中文导读] [AllReduce逐行 S73] 设置当前调用状态为/按`memcpy_sp(sqe->rmtEid, RMT_EID_BYTE_SIZE, rmtReverseEid_.raw, RAW_SIZE)`（当前UB WQE结构的rmtEid字段）；把WQE/SQE数据复制到设备映射队列内存，失败抛异常。
    s32 ret = memcpy_sp(sqe->rmtEid, RMT_EID_BYTE_SIZE, rmtReverseEid_.raw, RAW_SIZE);
    // [中文导读] [AllReduce逐行 S74] 仅当`(UNLIKELY(ret != 0))`（当前调用状态）成立时进入此分支。
    if (UNLIKELY(ret != 0)) {
        // [中文导读] [AllReduce逐行 S75] 记录UbConnLite::FillCommSqe的错误诊断，字段包含当前调用状态；日志本身不执行传输。
        HCCL_ERROR("UbConnLite::FillCommSqe FillCommSqe memcpy failed, ret=%d", ret);
        // [中文导读] [AllReduce逐行 S76] 抛出当前内部/驱动异常并中止此操作；上层EXCEPTION捕获边界负责转为返回状态。
        THROW<InternalException>(StringFormat("UbConnLite::FillCommSqe memcpy_sp failed, ret = %d", ret));
    // [中文导读] [AllReduce逐行 S77] 结束`if (UNLIKELY(ret != 0))`（当前调用状态）分支/循环；控制流返回外层。
    }

    // [中文导读] [AllReduce逐行 S79] 设置当前UB WQE结构的sgeNum字段为/按`1`。
    sqe->sgeNum = 1;
    // [中文导读] [AllReduce逐行 S80] 设置当前UB WQE结构的targetHint字段为/按`0`。
    sqe->targetHint = 0;
    // [中文导读] [AllReduce逐行 S81] 设置当前UB WQE结构的rmtObjId字段为/按`rmt.GetTokenId()`（远端缓冲区/地址的GetTokenId字段）；读取当前内存注册token ID。
    sqe->rmtObjId = rmt.GetTokenId();
    // [中文导读] [AllReduce逐行 S82] 设置当前UB WQE结构的tokenEn字段为/按`1`。
    sqe->tokenEn = 1;
    // [中文导读] [AllReduce逐行 S83] 设置当前UB WQE结构的rmtTokenValue字段为/按`rmt.GetTokenValue()`（远端缓冲区/地址的GetTokenValue字段）；读取当前内存注册token value。
    sqe->rmtTokenValue = rmt.GetTokenValue();
    // [中文导读] [AllReduce逐行 S84] 设置当前UB WQE结构的rmtAddrLow字段为/按`rmt.GetAddr() & ADDR_BIT_LOW`（远端缓冲区/地址的GetAddr字段）；读取缓冲区起始地址。
    sqe->rmtAddrLow = rmt.GetAddr() & ADDR_BIT_LOW;
    // [中文导读] [AllReduce逐行 S85] 设置当前UB WQE结构的rmtAddrHigh字段为/按`rmt.GetAddr() >> ADDR_BIT_OFFSET`（远端缓冲区/地址的GetAddr字段）；读取缓冲区起始地址。
    sqe->rmtAddrHigh = rmt.GetAddr() >> ADDR_BIT_OFFSET;
    // [中文导读] [AllReduce逐行 S86] 记录UbConnLite::FillCommSqe的状态/性能诊断；日志本身不执行传输。
    HCCL_INFO(
        // [中文导读] [AllReduce逐行 S87] 为当前UbConnLite::FillCommSqe诊断/异常表达式提供格式文本；这一物理行没有数据搬运副作用。
        "UbConnLite FillCommSqe UdmaSqeCommon slicePos[%d] sqe->cqe = %u, sqe->owner = %u sqe->opcode = %u, "
        // [中文导读] [AllReduce逐行 S88] 为当前UbConnLite::FillCommSqe诊断/异常表达式提供格式文本；这一物理行没有数据搬运副作用。
        "sqe->tpn = %u, sqe->rmtObjId = %u, sqe->rmtAddrLow = %u, sqe->rmtAddrHigh = %u, sqe->placeOdr = %u, "
        // [中文导读] [AllReduce逐行 S89] 为当前UbConnLite::FillCommSqe诊断/异常表达式提供格式文本；这一物理行没有数据搬运副作用。
        "sqe->compOrder = %u, sqe->fence = %u",
        // [中文导读] [AllReduce逐行 S90] 为前述多行表达式补入`slicePos, sqe->cqe, sqe->owner, sqe->opcode, sqe->tpn, sqe->rmtObjId, sqe->rmtAddrLow, sqe->rmtAddrHigh,`（当前分片首/中/尾位置、当前UB WQE结构的cqe字段、当前UB WQE结构的owner字段、当前UB WQE结构的opcode字段、当前UB WQE结构的tpn字段、当前UB WQE结构的rmtObjId字段、当前UB WQE结构的rmtAddrLow字段、当前UB WQE结构的rmtAddrHigh字段）；本行是参数/结构化初始化续行。
        slicePos, sqe->cqe, sqe->owner, sqe->opcode, sqe->tpn, sqe->rmtObjId, sqe->rmtAddrLow, sqe->rmtAddrHigh,
        // [中文导读] [AllReduce逐行 S91] 为前述多行表达式补入`sqe->placeOdr, sqe->compOrder, sqe->fence)`（当前UB WQE结构的placeOdr字段、当前UB WQE结构的compOrder字段、当前UB WQE结构的fence字段）；本行是参数/结构化初始化续行。
        sqe->placeOdr, sqe->compOrder, sqe->fence);
// [中文导读] [AllReduce逐行 S92] 结束UbConnLite::FillCommSqe函数体；控制流返回外层。
}

// [中文导读] [AllReduce逐行 S94] UbConnLite::FillCommSqeReduceInfo的接口声明：WQE公共字段、归约操作、元素数据类型；这些参数属于本函数调用边界。
void UbConnLite::FillCommSqeReduceInfo(UdmaSqeCommon& sqeComm, ReduceOp reduceOp, DataType dataType, u32 udfType) const
// [中文导读] [AllReduce逐行 S95] 进入UbConnLite::FillCommSqeReduceInfo函数体：按设备WQE格式填归约类型和操作编码。
{
    // [中文导读] [AllReduce逐行 S96] 记录UbConnLite::FillCommSqeReduceInfo的状态/性能诊断；日志本身不执行传输。
    HCCL_INFO("[UbConnLite::%s] start", __func__);

    // [中文导读] [AllReduce逐行 S98] 设置WQE公共字段的inlinedata.udfData.udfType字段为/按`udfType`。
    sqeComm.inlinedata.udfData.udfType = udfType; // 0代表inline reduce

    // [中文导读] [AllReduce逐行 S100] 仅当`((g_ubmaDataOpMap.find(reduceOp) != g_ubmaDataOpMap.end())`（归约操作）成立时进入此分支；调用find, end，使用归约操作。
    if ((g_ubmaDataOpMap.find(reduceOp) != g_ubmaDataOpMap.end())
        // [中文导读] [AllReduce逐行 S101] 补全本分支/循环判断的`&& (g_ubmaDataTypeMap.find(dataType) != g_ubmaDataTypeMap.end()))`（元素数据类型），和前面条件共同决定是否进入后续路径。
        && (g_ubmaDataTypeMap.find(dataType) != g_ubmaDataTypeMap.end())) {
        // [中文导读] [AllReduce逐行 S102] 把底层归约枚举映射成UB WQE操作编码；SUM映射为0xA。
        sqeComm.inlinedata.udfData.reduceOp = g_ubmaDataOpMap.at(reduceOp);
        // [中文导读] [AllReduce逐行 S103] 把底层数据类型映射成UB WQE类型编码；FP32映射为0x7。
        sqeComm.inlinedata.udfData.reduceType = g_ubmaDataTypeMap.at(dataType);
    // [中文导读] [AllReduce逐行 S104] 进入上一个条件未命中的替代路径，按本函数的资源/设备/协议分流继续处理。
    } else {
        // [中文导读] [AllReduce逐行 S105] 抛出当前内部/驱动异常并中止此操作；上层EXCEPTION捕获边界负责转为返回状态。
        THROW<InvalidParamsException>(StringFormat(
            // [中文导读] [AllReduce逐行 S106] 为当前UbConnLite::FillCommSqeReduceInfo诊断/异常表达式提供格式文本，将报告归约操作的Describe字段；这一物理行没有数据搬运副作用。
            "%s reduceOp[%s] or type[%s] is not supported.", __func__, reduceOp.Describe().c_str(),
            // [中文导读] [AllReduce逐行 S107] 为组装带上下文的错误或状态文本；取得对象诊断文本用于日志补入`dataType.Describe().c_str()))`（元素数据类型的Describe字段）；本行是参数/结构化初始化续行。
            dataType.Describe().c_str()));
    // [中文导读] [AllReduce逐行 S108] 结束当前局部作用域；控制流返回外层。
    }

    // udf字段是否有效
    // [中文导读] [AllReduce逐行 S111] 设置WQE公共字段的udfFlag字段为/按`1`。
    sqeComm.udfFlag = 1;

    // [中文导读] [AllReduce逐行 S113] 记录UbConnLite::FillCommSqeReduceInfo的状态/性能诊断；日志本身不执行传输。
    HCCL_INFO(
        // [中文导读] [AllReduce逐行 S114] 为当前UbConnLite::FillCommSqeReduceInfo诊断/异常表达式提供格式文本，将报告归约操作的Describe字段；这一物理行没有数据搬运副作用。
        "[UbConnLite::%s] end, reduceOp[%s], reduceType[%s]", __func__, reduceOp.Describe().c_str(),
        // [中文导读] [AllReduce逐行 S115] 为取得对象诊断文本用于日志补入`dataType.Describe().c_str())`（元素数据类型的Describe字段）；本行是参数/结构化初始化续行。
        dataType.Describe().c_str());
// [中文导读] [AllReduce逐行 S116] 结束UbConnLite::FillCommSqeReduceInfo函数体；控制流返回外层。
}

// [中文导读] [AllReduce逐行 S118] UbConnLite::ProcessSlices的接口声明：按最大 sliceSize 切分整片和尾片，标记 FIRST/MIDDLE/LAST/ONLY；这些参数属于本函数调用边界。
void UbConnLite::ProcessSlices(
    // [中文导读] [AllReduce逐行 S119] UbConnLite::ProcessSlices的接口声明：本端缓冲区/地址、远端缓冲区/地址、单WQE最大字节长度；这些参数属于本函数调用边界。
    const RmaBufSliceLite& loc, const RmtRmaBufSliceLite& rmt, u32 maxSliceSize,
    // [中文导读] [AllReduce逐行 S120] UbConnLite::ProcessSlices的接口声明：逐片生成WQE的回调；这些参数属于本函数调用边界。
    std::function<void(const RmaBufSliceLite&, const RmtRmaBufSliceLite&, SlicePosition)> processOneSlice,
    // [中文导读] [AllReduce逐行 S121] UbConnLite::ProcessSlices的接口声明：元素数据类型；这些参数属于本函数调用边界。
    DataType dataType) const
// [中文导读] [AllReduce逐行 S122] 进入UbConnLite::ProcessSlices函数体：按最大 sliceSize 切分整片和尾片，标记 FIRST/MIDDLE/LAST/ONLY。
{
    // [中文导读] [AllReduce逐行 S123] 显式忽略`dataType`（元素数据类型），该接口参数/调用结果在此实现中未参与后续计算。
    (void)dataType;
    // reduce操作需要保证切片大小是数据类型大小的整数倍
    // [中文导读] [AllReduce逐行 S125] 设置单片字节长度为/按`static_cast<u64>(maxSliceSize)`（单WQE最大字节长度）。
    u64 sliceSize = static_cast<u64>(maxSliceSize);

    // [中文导读] [AllReduce逐行 S127] 设置本端请求字节长度为/按`loc.GetSize()`（本端缓冲区/地址的GetSize字段）；读取缓冲区字节长度。
    u64 locBufSize = loc.GetSize();
    // [中文导读] [AllReduce逐行 S128] 设置完整分片条数为/按`locBufSize / sliceSize`（本端请求字节长度、单片字节长度）。
    u64 sliceNum = locBufSize / sliceSize;
    // [中文导读] [AllReduce逐行 S129] 设置不足整片的尾片字节长度为/按`locBufSize % sliceSize`（本端请求字节长度、单片字节长度）。
    u64 lastSliceSize = locBufSize % sliceSize;

    // [中文导读] [AllReduce逐行 S131] 设置累计传输字节数为/按`sliceNum * sliceSize`（完整分片条数、单片字节长度）。
    u64 totalSize = sliceNum * sliceSize;

    // [中文导读] [AllReduce逐行 S133] 仅当`(UNLIKELY(loc.GetAddr() > UINT64_MAX - totalSize || rmt.GetAddr() > UINT64_MAX - totalSize))`（本端缓冲区/地址的GetAddr字段、累计传输字节数、远端缓冲区/地址的GetAddr字段）成立时进入此分支；读取缓冲区起始地址。
    if (UNLIKELY(loc.GetAddr() > UINT64_MAX - totalSize || rmt.GetAddr() > UINT64_MAX - totalSize)) {
        // [中文导读] [AllReduce逐行 S134] 抛出当前内部/驱动异常并中止此操作；上层EXCEPTION捕获边界负责转为返回状态。
        THROW<InternalException>("integer overflow occurs");
    // [中文导读] [AllReduce逐行 S135] 结束`if (UNLIKELY(loc.GetAddr() > UINT64_MAX - totalSize || rmt.GetAddr() > UINT64_MAX - totalSize))`（本端缓冲区/地址的GetAddr字段、累计传输字节数、远端缓冲区/地址的GetAddr字段）分支/循环；控制流返回外层。
    }
    // [中文导读] [AllReduce逐行 S136] 按`(u64 sliceIdx = 0; sliceIdx < sliceNum; sliceIdx++)`（当前完整分片下标、完整分片条数）遍历本批条目/分片；各次处理保持数组对应关系。
    for (u64 sliceIdx = 0; sliceIdx < sliceNum; sliceIdx++) {
        // [中文导读] [AllReduce逐行 S137] 设置当前分片字节偏移为/按`sliceIdx * sliceSize`（当前完整分片下标、单片字节长度）。
        u64 offset = sliceIdx * sliceSize;
        // [中文导读] [AllReduce逐行 S138] 设置当前分片本端地址为/按`loc.GetAddr() + offset`（本端缓冲区/地址的GetAddr字段、当前分片字节偏移）；读取缓冲区起始地址。
        u64 locAddr = loc.GetAddr() + offset;
        // [中文导读] [AllReduce逐行 S139] 设置当前分片远端地址为/按`rmt.GetAddr() + offset`（远端缓冲区/地址的GetAddr字段、当前分片字节偏移）；读取缓冲区起始地址。
        u64 rmtAddr = rmt.GetAddr() + offset;

        // [中文导读] [AllReduce逐行 S141] 记录UbConnLite::ProcessSlices的状态/性能诊断；日志本身不执行传输。
        HCCL_INFO(
            // [中文导读] [AllReduce逐行 S142] 为当前UbConnLite::ProcessSlices诊断/异常表达式提供格式文本；这一物理行没有数据搬运副作用。
            "[UbConnLite::%s] Slice[%llu]: offset=0x%llx, locAddr=0x%llx, rmtAddr=0x%llx, size=0x%llx", __func__,
            // [中文导读] [AllReduce逐行 S143] 为前述多行表达式补入`sliceIdx, offset, locAddr, rmtAddr, sliceSize)`（当前完整分片下标、当前分片字节偏移、当前分片本端地址、当前分片远端地址、单片字节长度）；本行是参数/结构化初始化续行。
            sliceIdx, offset, locAddr, rmtAddr, sliceSize);

        // [中文导读] [AllReduce逐行 S145] 读取当前内存注册token ID；对象涉及当前本端分片、当前分片本端地址、单片字节长度、本端缓冲区/地址的GetTokenId字段。
        RmaBufSliceLite locSlice(locAddr, sliceSize, 0, loc.GetTokenId());

        // [中文导读] [AllReduce逐行 S147] 读取当前内存注册token ID；读取当前内存注册token value；对象涉及当前远端分片、当前分片远端地址、单片字节长度、远端缓冲区/地址的GetTokenId字段、远端缓冲区/地址的GetTokenValue字段。
        RmtRmaBufSliceLite rmtSlice(rmtAddr, sliceSize, 0, rmt.GetTokenId(), rmt.GetTokenValue(), UINT32_MAX);
        // [中文导读] [AllReduce逐行 S148] 设置当前分片首/中/尾位置为/按`(sliceIdx == 0) ? SlicePosition::FIRST : SlicePosition::MIDDLE`（当前完整分片下标）。
        SlicePosition slicePos = (sliceIdx == 0) ? SlicePosition::FIRST : SlicePosition::MIDDLE;
        // [中文导读] [AllReduce逐行 S149] 仅当`((sliceIdx == sliceNum - 1) && lastSliceSize == 0)`（当前完整分片下标、完整分片条数、不足整片的尾片字节长度）成立时进入此分支。
        if ((sliceIdx == sliceNum - 1) && lastSliceSize == 0) {
            // SlicePosition::ONLY表示既是首片又是尾片的情况，只有一片的情况
            // [中文导读] [AllReduce逐行 S151] 设置当前分片首/中/尾位置为/按`(sliceIdx == 0) ? SlicePosition::ONLY : SlicePosition::LAST`（当前完整分片下标）。
            slicePos = (sliceIdx == 0) ? SlicePosition::ONLY : SlicePosition::LAST;
        // [中文导读] [AllReduce逐行 S152] 结束`if ((sliceIdx == sliceNum - 1) && lastSliceSize == 0)`（当前完整分片下标、完整分片条数、不足整片的尾片字节长度）分支/循环；控制流返回外层。
        }
        // [中文导读] [AllReduce逐行 S153] 调用processOneSlice，使用逐片生成WQE的回调、当前本端分片、当前远端分片、当前分片首/中/尾位置；传入/处理逐片生成WQE的回调、当前本端分片、当前远端分片、当前分片首/中/尾位置。
        processOneSlice(locSlice, rmtSlice, slicePos);
    // [中文导读] [AllReduce逐行 S154] 结束`for (u64 sliceIdx = 0; sliceIdx < sliceNum; sliceIdx++)`（当前完整分片下标、完整分片条数）分支/循环；控制流返回外层。
    }

    // [中文导读] [AllReduce逐行 S156] 仅当`(lastSliceSize > 0)`（不足整片的尾片字节长度）成立时进入此分支。
    if (lastSliceSize > 0) {
        // [中文导读] [AllReduce逐行 S157] 读取缓冲区起始地址；读取当前内存注册token ID；对象涉及尾片本端描述、本端缓冲区/地址的GetAddr字段、完整分片条数、单片字节长度、不足整片的尾片字节长度、本端缓冲区/地址的GetTokenId字段。
        RmaBufSliceLite lastLocSlice(loc.GetAddr() + sliceNum * sliceSize, lastSliceSize, 0, loc.GetTokenId());

        // [中文导读] [AllReduce逐行 S159] 调用lastRmtSlice，使用尾片远端描述；对象涉及尾片远端描述。
        RmtRmaBufSliceLite lastRmtSlice(
            // [中文导读] [AllReduce逐行 S160] 为读取缓冲区起始地址；读取当前内存注册token ID；读取当前内存注册token value补入`rmt.GetAddr() + sliceNum * sliceSize, lastSliceSize, 0, rmt.GetTokenId(), rmt.GetTokenValue(), UINT32_MAX)`（远端缓冲区/地址的GetAddr字段、完整分片条数、单片字节长度、不足整片的尾片字节长度、远端缓冲区/地址的GetTokenId字段、远端缓冲区/地址的GetTokenValue字段）；本行是参数/结构化初始化续行。
            rmt.GetAddr() + sliceNum * sliceSize, lastSliceSize, 0, rmt.GetTokenId(), rmt.GetTokenValue(), UINT32_MAX);
        // [中文导读] [AllReduce逐行 S161] 设置当前分片首/中/尾位置为/按`(sliceNum == 0) ? SlicePosition::ONLY : SlicePosition::LAST`（完整分片条数）。
        SlicePosition slicePos = (sliceNum == 0) ? SlicePosition::ONLY : SlicePosition::LAST;
        // [中文导读] [AllReduce逐行 S162] 调用processOneSlice，使用逐片生成WQE的回调、尾片本端描述、尾片远端描述、当前分片首/中/尾位置；传入/处理逐片生成WQE的回调、尾片本端描述、尾片远端描述、当前分片首/中/尾位置。
        processOneSlice(lastLocSlice, lastRmtSlice, slicePos);
        // [中文导读] [AllReduce逐行 S163] 推进/回退`sliceNum++`（完整分片条数），更新当前分片、槽位或状态重试的计数。
        sliceNum++;
    // [中文导读] [AllReduce逐行 S164] 结束`if (lastSliceSize > 0)`（不足整片的尾片字节长度）分支/循环；控制流返回外层。
    }

    // [中文导读] [AllReduce逐行 S166] 记录UbConnLite::ProcessSlices的状态/性能诊断；日志本身不执行传输。
    HCCL_INFO(
        // [中文导读] [AllReduce逐行 S167] 为当前UbConnLite::ProcessSlices诊断/异常表达式提供格式文本；这一物理行没有数据搬运副作用。
        "[UbConnLite::%s] end, locBufSize[%llu], sliceNum[%llu], sliceSize[%llu], lastSliceSize[%llu]", __func__,
        // [中文导读] [AllReduce逐行 S168] 为前述多行表达式补入`locBufSize, sliceNum, sliceSize, lastSliceSize)`（本端请求字节长度、完整分片条数、单片字节长度、不足整片的尾片字节长度）；本行是参数/结构化初始化续行。
        locBufSize, sliceNum, sliceSize, lastSliceSize);
// [中文导读] [AllReduce逐行 S169] 结束UbConnLite::ProcessSlices函数体；控制流返回外层。
}

void UbConnLite::ProcessSlicesWithNotify(
    const RmaBufSliceLite& loc, const RmtRmaBufSliceLite& rmt, u32 maxSliceSize,
    std::function<void(const RmaBufSliceLite&, const RmtRmaBufSliceLite&, SlicePosition)> processOneSlice,
    std::function<void(const RmaBufSliceLite&, const RmtRmaBufSliceLite&, SlicePosition)> processOneSliceWithNotify,
    DataType dataType) const
{
    HCCL_INFO("[UbConnLite::%s] start", __func__);

    // reduce操作需要保证切片大小是数据类型大小的整数倍
    u32 sliceSize = maxSliceSize;
    if (dataType != DataType::INVALID) {
        u32 dataTypeSize = DATA_TYPE_SIZE_MAP.at(dataType);
        sliceSize = maxSliceSize / dataTypeSize * dataTypeSize;
    }

    u64 locBufSize = loc.GetSize();
    u64 sliceNum = locBufSize / sliceSize;
    u64 lastSliceSize = locBufSize % sliceSize;
    if (sliceNum > 0 && lastSliceSize == 0) {
        sliceNum--;
        lastSliceSize = sliceSize;
    }
    u64 totalSize = sliceNum * sliceSize;
    if (UNLIKELY(loc.GetAddr() > UINT64_MAX - totalSize || rmt.GetAddr() > UINT64_MAX - totalSize)) {
        THROW<InternalException>("integer overflow occurs");
    }
    for (u64 sliceIdx = 0; sliceIdx < sliceNum; sliceIdx++) {
        RmaBufSliceLite locSlice(loc.GetAddr() + sliceIdx * sliceSize, sliceSize, 0, loc.GetTokenId());

        RmtRmaBufSliceLite rmtSlice(
            rmt.GetAddr() + sliceIdx * sliceSize, sliceSize, 0, rmt.GetTokenId(), rmt.GetTokenValue(), UINT32_MAX);
        SlicePosition slicePos = (sliceIdx == 0) ? SlicePosition::FIRST : SlicePosition::MIDDLE;
        processOneSlice(locSlice, rmtSlice, slicePos);
    }

    if (lastSliceSize > 0) {
        RmaBufSliceLite lastLocSlice(loc.GetAddr() + sliceNum * sliceSize, lastSliceSize, 0, loc.GetTokenId());

        RmtRmaBufSliceLite lastRmtSlice(
            rmt.GetAddr() + sliceNum * sliceSize, lastSliceSize, 0, rmt.GetTokenId(), rmt.GetTokenValue(), UINT32_MAX);
        SlicePosition slicePos = (sliceNum == 0) ? SlicePosition::ONLY : SlicePosition::LAST;
        processOneSliceWithNotify(lastLocSlice, lastRmtSlice, slicePos);
    }

    HCCL_INFO(
        "[UbConnLite::%s] end, locBufSize[%llu], sliceNum[%llu], sliceSize[%u], lastSliceSize[%llu]", __func__,
        locBufSize, sliceNum, sliceSize, lastSliceSize);
}

// [中文导读] [AllReduce逐行 S220] UbConnLite::FillOneSqeWrite的接口声明：将远端公共字段与本地SGE填入UB数据WQE，零长度时取消SGE；这些参数属于本函数调用边界。
void UbConnLite::FillOneSqeWrite(
    // [中文导读] [AllReduce逐行 S221] UbConnLite::FillOneSqeWrite的接口声明：本端缓冲区/地址、远端缓冲区/地址、UB WQE保序/完成配置、当前UB WQE结构；这些参数属于本函数调用边界。
    const RmaBufSliceLite& loc, const RmtRmaBufSliceLite& rmt, const SqeConfigLite& cfg, UdmaSqeWrite* sqe,
    // [中文导读] [AllReduce逐行 S222] UbConnLite::FillOneSqeWrite的接口声明：UB读写操作码、当前分片首/中/尾位置；这些参数属于本函数调用边界。
    UdmaSqOpcode opCode, SlicePosition slicePos)
// [中文导读] [AllReduce逐行 S223] 进入UbConnLite::FillOneSqeWrite函数体：将远端公共字段与本地SGE填入UB数据WQE，零长度时取消SGE。
{
    // [中文导读] [AllReduce逐行 S224] 记录UbConnLite::FillOneSqeWrite的状态/性能诊断，字段包含本端缓冲区/地址的GetSize字段；日志本身不执行传输。
    HCCL_INFO("[UbConnLite::%s] start, loc size[%llu]", __func__, loc.GetSize());

    // [中文导读] [AllReduce逐行 S226] 设置当前UB WQE结构的comm.inlineEn字段为/按`0`。
    sqe->comm.inlineEn = 0;
    // [中文导读] [AllReduce逐行 S227] 填写远端地址/token、操作码、分片顺序和完成标志；传入/处理当前UB WQE结构的comm字段、远端缓冲区/地址、UB WQE保序/完成配置、UB读写操作码、当前分片首/中/尾位置。
    FillCommSqe(&(sqe->comm), rmt, cfg, opCode, slicePos);
    // [中文导读] [AllReduce逐行 S228] 填写本端SGE地址、token及字节长度；传入/处理当前UB WQE结构的u.sge字段、本端缓冲区/地址。
    FillLocalSgeSqe(&(sqe->u.sge), loc);
    // [中文导读] [AllReduce逐行 S229] 仅当`(sqe->u.sge.length == 0)`（当前UB WQE结构的u.sge.length字段）成立时进入此分支。
    if (sqe->u.sge.length == 0) {
        // [中文导读] [AllReduce逐行 S230] 设置当前UB WQE结构的comm.sgeNum字段为/按`0`。
        sqe->comm.sgeNum = 0;
    // [中文导读] [AllReduce逐行 S231] 结束`if (sqe->u.sge.length == 0)`（当前UB WQE结构的u.sge.length字段）分支/循环；控制流返回外层。
    }

    // [中文导读] [AllReduce逐行 S233] 记录UbConnLite::FillOneSqeWrite的状态/性能诊断；日志本身不执行传输。
    HCCL_INFO("[UbConnLite::%s] end", __func__);
// [中文导读] [AllReduce逐行 S234] 结束UbConnLite::FillOneSqeWrite函数体；控制流返回外层。
}

// [中文导读] [AllReduce逐行 S236] UbConnLite::LaunchOneWqe的接口声明：当前UB WQE结构、UB读写操作码；这些参数属于本函数调用边界。
void UbConnLite::LaunchOneWqe(UdmaSqeWrite* sqe, UdmaSqOpcode opCode)
// [中文导读] [AllReduce逐行 S237] 进入UbConnLite::LaunchOneWqe函数体：维护16位 UB PI，定位 UB SQ 环槽并在未缓存锁定时复制 WQE。
{
    // [中文导读] [AllReduce逐行 S238] 记录UbConnLite::LaunchOneWqe的状态/性能诊断，字段包含UB读写操作码的Describe字段；日志本身不执行传输。
    HCCL_INFO("[UbConnLite::%s] start, opCode[%s]", __func__, opCode.Describe().c_str());

    // sqOffset是用于计算Ubjetty中下wqe位置的偏移，小于sqDepth
    // [中文导读] [AllReduce逐行 S241] 设置UB SQ环内WQE槽位为/按`pi % sqDepth_`（UB jetty生产指针（16位自然增长）、RTSQ或UB SQ深度）。
    u32 sqOffset = pi % sqDepth_;
    // [中文导读] [AllReduce逐行 S242] 仅当`(sqOffset < sqDepth_ && (sqOffset + 1) >= sqDepth_)`（UB SQ环内WQE槽位、RTSQ或UB SQ深度）成立时进入此分支。
    if (sqOffset < sqDepth_ && (sqOffset + 1) >= sqDepth_) {
        // [中文导读] [AllReduce逐行 S243] 推进/回退`piDetourCount++`，更新当前分片、槽位或状态重试的计数。
        piDetourCount++;
    // [中文导读] [AllReduce逐行 S244] 结束`if (sqOffset < sqDepth_ && (sqOffset + 1) >= sqDepth_)`（UB SQ环内WQE槽位、RTSQ或UB SQ深度）分支/循环；控制流返回外层。
    }
    // pi维护用于传入DB Send用于Rtsq 敲door bell，要求u16数据结构并且自然增长
    // [中文导读] [AllReduce逐行 S246] 设置UB jetty生产指针（16位自然增长）为/按`pi + 1`（UB jetty生产指针（16位自然增长））。
    pi = pi + 1;

    // 写wqe到va
    // [中文导读] [AllReduce逐行 S249] 设置当前WQE写入地址为/按`ReinterpretAs<u8*>(sqVa_ + sqOffset * SQE_SIZE_64)`（UB SQ基地址、UB SQ环内WQE槽位）。
    u8* va = ReinterpretAs<u8*>(sqVa_ + sqOffset * SQE_SIZE_64);
    // [中文导读] [AllReduce逐行 S250] 仅当`(!dwqeCacheLocked_)`（WQE直接写队列的缓存锁定状态）成立时进入此分支。
    if (!dwqeCacheLocked_) {
        // [中文导读] [AllReduce逐行 S251] 设置当前调用状态为/按`memcpy_sp(va, SQE_SIZE_64, sqe, SQE_SIZE_64)`（当前WQE写入地址、当前UB WQE结构）；把WQE/SQE数据复制到设备映射队列内存，失败抛异常。
        auto ret = memcpy_sp(va, SQE_SIZE_64, sqe, SQE_SIZE_64);
        // [中文导读] [AllReduce逐行 S252] 仅当`(UNLIKELY(ret != 0))`（当前调用状态）成立时进入此分支。
        if (UNLIKELY(ret != 0)) {
            // [中文导读] [AllReduce逐行 S253] 抛出当前内部/驱动异常并中止此操作；上层EXCEPTION捕获边界负责转为返回状态。
            THROW<InternalException>(StringFormat("[UbConnLite::%s] memcpy_sp failed, ret = %d", __func__, ret));
        // [中文导读] [AllReduce逐行 S254] 结束`if (UNLIKELY(ret != 0))`（当前调用状态）分支/循环；控制流返回外层。
        }
    // [中文导读] [AllReduce逐行 S255] 结束`if (!dwqeCacheLocked_)`（WQE直接写队列的缓存锁定状态）分支/循环；控制流返回外层。
    }

    // [中文导读] [AllReduce逐行 S257] 记录UbConnLite::LaunchOneWqe的状态/性能诊断；日志本身不执行传输。
    HCCL_INFO(
        // [中文导读] [AllReduce逐行 S258] 为当前UbConnLite::LaunchOneWqe诊断/异常表达式提供格式文本；这一物理行没有数据搬运副作用。
        "[UbConnLite::%s] end, dieId_[%u], funcId_[%u], jettyId_[%u], pi[%u], ci[%u]", __func__, dieId_, funcId_,
        // [中文导读] [AllReduce逐行 S259] 为前述多行表达式补入`jettyId_, pi, ci)`（UB jetty生产指针（16位自然增长）、UB jetty消费指针）；本行是参数/结构化初始化续行。
        jettyId_, pi, ci);
// [中文导读] [AllReduce逐行 S260] 结束UbConnLite::LaunchOneWqe函数体；控制流返回外层。
}

void UbConnLite::FillOneWqeWithNotify(
    const RmaBufSliceLite& loc, const RmtRmaBufSliceLite& rmt, const SqeConfigLite& cfg, UdmaSqeWriteWithNotify* sqe,
    const RmtRmaBufSliceLite& notify, u64 notifyData, u32 opCode, SlicePosition slicePos)
{
    HCCL_INFO("[UbConnLite::%s] start, locSize[%u], opCode[%u]", __func__, loc.GetSize(), opCode);

    // 填充sqe
    sqe->comm.inlineEn = 0;
    FillCommSqe(&(sqe->comm), rmt, cfg, WRITE_WITH_NOTIFY_OPCODE, slicePos);
    FillNotifySqe(&(sqe->notify), notify, notifyData);
    FillLocalSgeSqe(&(sqe->localU.sge), loc);
    if (sqe->localU.sge.length == 0) {
        sqe->comm.sgeNum = 0;
    }
    sqe->rsv1 = 0;
    sqe->rsv2 = 0;

    HCCL_INFO("[UbConnLite::%s] end", __func__);
}

void UbConnLite::LaunchOneWqeWithNotify(UdmaSqeWriteWithNotify* sqe, u32 opCode)
{
    HCCL_INFO("[UbConnLite::%s] start, opCode[%u]", __func__, opCode);

    // sqOffset是用于计算Ubjetty中下wqe位置的偏移，小于sqDepth
    u32 sqOffset = pi % sqDepth_;
    if (sqOffset < sqDepth_ && (sqOffset + PI_NUM_TWO) >= sqDepth_) {
        piDetourCount++;
    }
    // pi维护用于传入DB Send用于Rtsq 敲door bell，要求u16数据结构并且自然增长
    pi = pi + PI_NUM_TWO;

    u8* va = ReinterpretAs<u8*>(sqVa_ + sqOffset * SQE_SIZE_64);
    if (!dwqeCacheLocked_) {
        // 带notify的wqe是96字节, 需要占用两个wqebb, 实际占用128字节
        if (sqOffset == sqDepth_ - 1) {
            MemorySetAndCopy(va, SQE_SIZE_64, sqe);
            va = ReinterpretAs<u8*>(sqVa_);
            MemorySetAndCopy(va, SQE_SIZE_64, ReinterpretAs<u8*>(sqe) + SQE_SIZE_64);
        } else {
            MemorySetAndCopy(va, SQE_SIZE_128, sqe);
        }
    }

    HCCL_INFO(
        "[UbConnLite::%s] end, dieId_[%u], funcId_[%u], jettyId_[%u], pi[%u], ci[%u]", __func__, dieId_, funcId_,
        jettyId_, pi, ci);
}

void UbConnLite::MemorySetAndCopy(u8* va, u32 sqeSize, void* sqe)
{
    auto ret = memset_s(va, sqeSize, 0, sqeSize);
    if (UNLIKELY(ret != 0)) {
        THROW<InternalException>(StringFormat("[UbConnLite::%s] memset fail, ret = %d", __func__, ret));
    }
    ret = memcpy_sp(va, sqeSize, sqe, sqeSize);
    if (UNLIKELY(ret != 0)) {
        THROW<InternalException>(StringFormat("[UbConnLite::%s] not support this op type yet.", __func__));
    }
}

void UbConnLite::Read(
    const RmaBufSliceLite& loc, const RmtRmaBufSliceLite& rmt, const SqeConfigLite& cfg, const StreamLite& stream,
    ConnLiteOperationOut& out)
{
    HCCL_INFO("[UbConnLite::%s] start", __func__);

    ProcessSlices(
        loc, rmt, maxReadSize,
        [&](const RmaBufSliceLite& locSlice, const RmtRmaBufSliceLite& rmtSlice, SlicePosition slicePos) {
            UdmaSqeWrite sqe{};
            FillOneSqeWrite(locSlice, rmtSlice, cfg, &sqe, UdmaSqOpcode::UDMA_OPC_READ, slicePos);
            ProcessOneWqe(&sqe, UdmaSqOpcode::UDMA_OPC_READ, stream);

            // 按需更新wqe tasks
            UpdateWqeTasks(sqe);
        });

    out.pi = pi;
    HCCL_INFO("[UbConnLite::%s] end, ConnLiteOperationOut.pi = %u, conn[%s]", __func__, out.pi, Describe().c_str());
}

// [中文导读] [AllReduce逐行 S344] UbConnLite::ReadReduce的接口声明：按最大读长度分 slice，填 READ 与归约字段、写 WQE 队列并返回 UB PI；这些参数属于本函数调用边界。
void UbConnLite::ReadReduce(
    // [中文导读] [AllReduce逐行 S345] UbConnLite::ReadReduce的接口声明：底层归约类型/操作描述、本端缓冲区/地址、远端缓冲区/地址、承载任务的执行流；这些参数属于本函数调用边界。
    ReduceIn reduceIn, const RmaBufSliceLite& loc, const RmtRmaBufSliceLite& rmt, const StreamLite& stream,
    // [中文导读] [AllReduce逐行 S346] UbConnLite::ReadReduce的接口声明：UB WQE保序/完成配置、具体对象/PI出参；这些参数属于本函数调用边界。
    const SqeConfigLite& cfg, ConnLiteOperationOut& out)
// [中文导读] [AllReduce逐行 S347] 进入UbConnLite::ReadReduce函数体：按最大读长度分 slice，填 READ 与归约字段、写 WQE 队列并返回 UB PI。
{
    // [中文导读] [AllReduce逐行 S348] 记录UbConnLite::ReadReduce的状态/性能诊断；日志本身不执行传输。
    HCCL_INFO("[UbConnLite::%s] start", __func__);

    // [中文导读] [AllReduce逐行 S350] 按单WQE最大字节长度分割本端与远端范围，逐片调用构造回调。
    ProcessSlices(
        // [中文导读] [AllReduce逐行 S351] 为按单WQE最大字节长度分割本端与远端范围，逐片调用构造回调补入`loc, rmt, maxReadSize,`（本端缓冲区/地址、远端缓冲区/地址）；本行是参数/结构化初始化续行。
        loc, rmt, maxReadSize,
        // [中文导读] [AllReduce逐行 S352] 定义逐片处理回调，捕获当前连接/配置上下文；回调参数描述本地与远端同一分片及其首尾位置。
        [&](const RmaBufSliceLite& locSlice, const RmtRmaBufSliceLite& rmtSlice, SlicePosition slicePos) {
            // [中文导读] [AllReduce逐行 S353] 准备当前UB WQE结构的局部存储/结构描述，初始化方式以本行声明为准。
            UdmaSqeWrite sqe{};
            // [中文导读] [AllReduce逐行 S354] 填当前读/写WQE的远端公共属性与本地SGE字段；传入/处理当前本端分片、当前远端分片、UB WQE保序/完成配置、当前UB WQE结构、当前分片首/中/尾位置。
            FillOneSqeWrite(locSlice, rmtSlice, cfg, &sqe, UdmaSqOpcode::UDMA_OPC_READ, slicePos);
            // [中文导读] [AllReduce逐行 S355] 给当前READ/WRITE WQE设置归约类型及操作编码；传入/处理当前UB WQE结构的comm字段、底层归约类型/操作描述的reduceOp字段、底层归约类型/操作描述的dataType字段。
            FillCommSqeReduceInfo(sqe.comm, reduceIn.reduceOp, reduceIn.dataType);
            // [中文导读] [AllReduce逐行 S356] 将当前WQE委托给LaunchOneWqe写UB SQ；传入/处理当前UB WQE结构、承载任务的执行流。
            ProcessOneWqe(&sqe, UdmaSqOpcode::UDMA_OPC_READ, stream);

            // 按需更新wqe tasks
            // [中文导读] [AllReduce逐行 S359] 仅开启跟踪时保存生成的WQE供缓存/日志使用；传入/处理当前UB WQE结构。
            UpdateWqeTasks(sqe);
        // [中文导读] [AllReduce逐行 S360] 结束前述调用/回调或结构初始化参数列表，使本次操作的实参完整。
        },
        // [中文导读] [AllReduce逐行 S361] 把归约数据类型传入ProcessSlices；当前普通切片实现显式忽略该参数，切片长度按maxReadSize计算。
        reduceIn.dataType);

    // [中文导读] [AllReduce逐行 S363] 设置具体对象/PI出参的pi字段为/按`pi`（UB jetty生产指针（16位自然增长））。
    out.pi = pi;
    // [中文导读] [AllReduce逐行 S364] 记录UbConnLite::ReadReduce的状态/性能诊断，字段包含具体对象/PI出参的pi字段；日志本身不执行传输。
    HCCL_INFO("[UbConnLite::%s] end, ConnLiteOperationOut.pi = %u, conn[%s]", __func__, out.pi, Describe().c_str());
// [中文导读] [AllReduce逐行 S365] 结束UbConnLite::ReadReduce函数体；控制流返回外层。
}

// [中文导读] [AllReduce逐行 S367] UbConnLite::Write的接口声明：按最大写长度分 slice，构造 WRITE WQE、写队列并返回 UB PI；这些参数属于本函数调用边界。
void UbConnLite::Write(
    // [中文导读] [AllReduce逐行 S368] UbConnLite::Write的接口声明：本端缓冲区/地址、远端缓冲区/地址、UB WQE保序/完成配置、承载任务的执行流；这些参数属于本函数调用边界。
    const RmaBufSliceLite& loc, const RmtRmaBufSliceLite& rmt, const SqeConfigLite& cfg, const StreamLite& stream,
    // [中文导读] [AllReduce逐行 S369] UbConnLite::Write的接口声明：具体对象/PI出参；这些参数属于本函数调用边界。
    ConnLiteOperationOut& out)
// [中文导读] [AllReduce逐行 S370] 进入UbConnLite::Write函数体：按最大写长度分 slice，构造 WRITE WQE、写队列并返回 UB PI。
{
    // [中文导读] [AllReduce逐行 S371] 记录UbConnLite::Write的状态/性能诊断，字段包含本端缓冲区/地址的GetSize字段；日志本身不执行传输。
    HCCL_INFO("[UbConnLite::%s] start, loc size = %llu", __func__, loc.GetSize());

    // [中文导读] [AllReduce逐行 S373] 按单WQE最大字节长度分割本端与远端范围，逐片调用构造回调。
    ProcessSlices(
        // [中文导读] [AllReduce逐行 S374] 为按单WQE最大字节长度分割本端与远端范围，逐片调用构造回调补入`loc, rmt, maxWriteSize,`（本端缓冲区/地址、远端缓冲区/地址）；本行是参数/结构化初始化续行。
        loc, rmt, maxWriteSize,
        // [中文导读] [AllReduce逐行 S375] 定义逐片处理回调，捕获当前连接/配置上下文；回调参数描述本地与远端同一分片及其首尾位置。
        [&](const RmaBufSliceLite& locSlice, const RmtRmaBufSliceLite& rmtSlice, SlicePosition slicePos) {
            // [中文导读] [AllReduce逐行 S376] 准备当前UB WQE结构的局部存储/结构描述，初始化方式以本行声明为准。
            UdmaSqeWrite sqe{};
            // [中文导读] [AllReduce逐行 S377] 填当前读/写WQE的远端公共属性与本地SGE字段；传入/处理当前本端分片、当前远端分片、UB WQE保序/完成配置、当前UB WQE结构、当前分片首/中/尾位置。
            FillOneSqeWrite(locSlice, rmtSlice, cfg, &sqe, UdmaSqOpcode::UDMA_OPC_WRITE, slicePos);
            // [中文导读] [AllReduce逐行 S378] 将当前WQE委托给LaunchOneWqe写UB SQ；传入/处理当前UB WQE结构、承载任务的执行流。
            ProcessOneWqe(&sqe, UdmaSqOpcode::UDMA_OPC_WRITE, stream);

            // 按需更新wqe tasks
            // [中文导读] [AllReduce逐行 S381] 仅开启跟踪时保存生成的WQE供缓存/日志使用；传入/处理当前UB WQE结构。
            UpdateWqeTasks(sqe);
        // [中文导读] [AllReduce逐行 S382] 结束前述调用/回调或结构初始化参数列表，使本次操作的实参完整。
        });

    // [中文导读] [AllReduce逐行 S384] 设置具体对象/PI出参的pi字段为/按`pi`（UB jetty生产指针（16位自然增长））。
    out.pi = pi;
    // [中文导读] [AllReduce逐行 S385] 记录UbConnLite::Write的状态/性能诊断，字段包含具体对象/PI出参的pi字段；日志本身不执行传输。
    HCCL_INFO("[UbConnLite::%s] end, ConnLiteOperationOut.pi = %u, conn[%s]", __func__, out.pi, Describe().c_str());
// [中文导读] [AllReduce逐行 S386] 结束UbConnLite::Write函数体；控制流返回外层。
}

// [中文导读] [AllReduce逐行 S388] UbConnLite::InlineWrite的接口声明：构造内联 WRITE WQE，把通知值直接嵌入 WQE 并写入 UB SQ；这些参数属于本函数调用边界。
void UbConnLite::InlineWrite(
    // [中文导读] [AllReduce逐行 S389] UbConnLite::InlineWrite的接口声明：字节容量或单片字节数、远端缓冲区/地址、UB WQE保序/完成配置、承载任务的执行流；这些参数属于本函数调用边界。
    const u8* data, u16 size, const RmtRmaBufSliceLite& rmt, const SqeConfigLite& cfg, const StreamLite& stream,
    // [中文导读] [AllReduce逐行 S390] UbConnLite::InlineWrite的接口声明：具体对象/PI出参；这些参数属于本函数调用边界。
    ConnLiteOperationOut& out)
// [中文导读] [AllReduce逐行 S391] 进入UbConnLite::InlineWrite函数体：构造内联 WRITE WQE，把通知值直接嵌入 WQE 并写入 UB SQ。
{
    // [中文导读] [AllReduce逐行 S392] 记录UbConnLite::InlineWrite的状态/性能诊断；日志本身不执行传输。
    HCCL_INFO("[UbConnLite::%s] start", __func__);

    // 构造sqe
    // [中文导读] [AllReduce逐行 S395] 准备当前UB WQE结构的局部存储/结构描述，初始化方式以本行声明为准。
    UdmaSqeWrite sqe{};
    // [中文导读] [AllReduce逐行 S396] 设置当前UB WQE结构的comm.inlineEn字段为/按`1`。
    sqe.comm.inlineEn = 1;
    // [中文导读] [AllReduce逐行 S397] 设置当前UB WQE结构的comm.inlineMsgLen字段为/按`size`（字节容量或单片字节数）。
    sqe.comm.inlineMsgLen = size;
    // [中文导读] [AllReduce逐行 S398] 填写远端地址/token、操作码、分片顺序和完成标志；传入/处理当前UB WQE结构的comm字段、远端缓冲区/地址、UB WQE保序/完成配置。
    FillCommSqe(&(sqe.comm), rmt, cfg, UdmaSqOpcode::UDMA_OPC_WRITE);
    // [中文导读] [AllReduce逐行 S399] 设置当前调用状态为/按`memcpy_sp(sqe.u.inlineData.data, SQE_INLINE_DATA_SIZE, data, size)`（当前UB WQE结构的u.inlineData.data字段、字节容量或单片字节数）；把WQE/SQE数据复制到设备映射队列内存，失败抛异常。
    auto ret = memcpy_sp(sqe.u.inlineData.data, SQE_INLINE_DATA_SIZE, data, size);
    // [中文导读] [AllReduce逐行 S400] 仅当`(UNLIKELY(ret != 0))`（当前调用状态）成立时进入此分支。
    if (UNLIKELY(ret != 0)) {
        // [中文导读] [AllReduce逐行 S401] 抛出当前内部/驱动异常并中止此操作；上层EXCEPTION捕获边界负责转为返回状态。
        THROW<InternalException>(StringFormat("[UbConnLite::%s] not support this op type yet.", __func__));
    // [中文导读] [AllReduce逐行 S402] 结束`if (UNLIKELY(ret != 0))`（当前调用状态）分支/循环；控制流返回外层。
    }

    // 写wqe到va
    // [中文导读] [AllReduce逐行 S405] 将当前WQE委托给LaunchOneWqe写UB SQ；传入/处理当前UB WQE结构、承载任务的执行流。
    ProcessOneWqe(&sqe, UdmaSqOpcode::UDMA_OPC_WRITE, stream);

    // 按需更新wqe tasks
    // [中文导读] [AllReduce逐行 S408] 仅开启跟踪时保存生成的WQE供缓存/日志使用；传入/处理当前UB WQE结构。
    UpdateWqeTasks(sqe);

    // [中文导读] [AllReduce逐行 S410] 设置具体对象/PI出参的pi字段为/按`pi`（UB jetty生产指针（16位自然增长））。
    out.pi = pi;
    // [中文导读] [AllReduce逐行 S411] 记录UbConnLite::InlineWrite的状态/性能诊断；日志本身不执行传输。
    HCCL_INFO(
        // [中文导读] [AllReduce逐行 S412] 为当前UbConnLite::InlineWrite诊断/异常表达式提供格式文本；这一物理行没有数据搬运副作用。
        "[UbConnLite::%s] end, ConnLiteOperationOut.pi = %u, ConnLiteOperationOut.datasize = %u, conn[%s]", __func__,
        // [中文导读] [AllReduce逐行 S413] 为取得对象诊断文本用于日志补入`out.pi, out.dataSize, Describe().c_str())`（具体对象/PI出参的pi字段、具体对象/PI出参的dataSize字段）；本行是参数/结构化初始化续行。
        out.pi, out.dataSize, Describe().c_str());
// [中文导读] [AllReduce逐行 S414] 结束UbConnLite::InlineWrite函数体；控制流返回外层。
}

void UbConnLite::FillNotifySqe(struct UdmaSqeNotify* sqe, const RmtRmaBufSliceLite& notify, u64 notifyData) const
{
    sqe->notifyTokenId = notify.GetTokenId();
    sqe->notifyTokenValue = notify.GetTokenValue();
    sqe->notifyAddrLow = notify.GetAddr() & ADDR_BIT_LOW;
    sqe->notifyAddrHigh = notify.GetAddr() >> ADDR_BIT_OFFSET;
    sqe->notifyDataLow = notifyData & ADDR_BIT_LOW;
    sqe->notifyDataHigh = notifyData >> ADDR_BIT_OFFSET;
    HCCL_INFO(
        "UbConnLite FillNotifySqe sqe->notifyAddrLow = %u "
        "sqe->notifyAddrHigh = %u, sqe->notifyDataLow = %u, sqe->notifyDataHigh = %u",
        sqe->notifyAddrLow, sqe->notifyAddrHigh, sqe->notifyDataLow, sqe->notifyDataHigh);
}

// [中文导读] [AllReduce逐行 S430] UbConnLite::FillLocalSgeSqe的接口声明：当前UB WQE结构、本端缓冲区/地址；这些参数属于本函数调用边界。
void UbConnLite::FillLocalSgeSqe(UdmaNormalSge* sqe, const RmaBufSliceLite& loc) const
// [中文导读] [AllReduce逐行 S431] 进入UbConnLite::FillLocalSgeSqe函数体：填入本端地址、token和字节长度字段。
{
    // [中文导读] [AllReduce逐行 S432] 设置当前UB WQE结构的length字段为/按`loc.GetSize()`（本端缓冲区/地址的GetSize字段）；读取缓冲区字节长度。
    sqe->length = loc.GetSize();
    // [中文导读] [AllReduce逐行 S433] 设置当前UB WQE结构的tokenId字段为/按`loc.GetTokenId()`（本端缓冲区/地址的GetTokenId字段）；读取当前内存注册token ID。
    sqe->tokenId = loc.GetTokenId();
    // [中文导读] [AllReduce逐行 S434] 设置当前UB WQE结构的dataAddrLow字段为/按`loc.GetAddr() & ADDR_BIT_LOW`（本端缓冲区/地址的GetAddr字段）；读取缓冲区起始地址。
    sqe->dataAddrLow = loc.GetAddr() & ADDR_BIT_LOW;
    // [中文导读] [AllReduce逐行 S435] 设置当前UB WQE结构的dataAddrHigh字段为/按`loc.GetAddr() >> ADDR_BIT_OFFSET`（本端缓冲区/地址的GetAddr字段）；读取缓冲区起始地址。
    sqe->dataAddrHigh = loc.GetAddr() >> ADDR_BIT_OFFSET;
    // [中文导读] [AllReduce逐行 S436] 记录UbConnLite::FillLocalSgeSqe的状态/性能诊断；日志本身不执行传输。
    HCCL_INFO(
        // [中文导读] [AllReduce逐行 S437] 为当前UbConnLite::FillLocalSgeSqe诊断/异常表达式提供格式文本；这一物理行没有数据搬运副作用。
        "UbConnLite FillLocalSgeSqe sqe->length = %u, sqe->dataAddrLow = %u "
        // [中文导读] [AllReduce逐行 S438] 为当前UbConnLite::FillLocalSgeSqe诊断/异常表达式提供格式文本；这一物理行没有数据搬运副作用。
        "sqe->dataAddrHigh = %u",
        // [中文导读] [AllReduce逐行 S439] 为前述多行表达式补入`sqe->length, sqe->dataAddrLow, sqe->dataAddrHigh)`（当前UB WQE结构的length字段、当前UB WQE结构的dataAddrLow字段、当前UB WQE结构的dataAddrHigh字段）；本行是参数/结构化初始化续行。
        sqe->length, sqe->dataAddrLow, sqe->dataAddrHigh);
// [中文导读] [AllReduce逐行 S440] 结束UbConnLite::FillLocalSgeSqe函数体；控制流返回外层。
}

void UbConnLite::WriteReduce(
    DataType dataType, ReduceOp reduceOp, const RmaBufSliceLite& loc, const StreamLite& stream,
    const RmtRmaBufSliceLite& rmt, const SqeConfigLite& cfg, ConnLiteOperationOut& out)
{
    HCCL_INFO(
        "[UbConnLite::%s] start, dataType = %u, reduceOp %u, loc.addr = %llu, "
        "rmt.addr = %llu, cfg.cqeEn = %u, out.pi = %u",
        __func__, dataType, reduceOp, loc.GetAddr(), rmt.GetAddr(), cfg.cqeEn, out.pi);

    ProcessSlices(
        loc, rmt, maxWriteSize,
        [&](const RmaBufSliceLite& locSlice, const RmtRmaBufSliceLite& rmtSlice, SlicePosition slicePos) {
            UdmaSqeWrite sqe{};
            FillCommSqeReduceInfo(sqe.comm, reduceOp, dataType);
            FillOneSqeWrite(locSlice, rmtSlice, cfg, &sqe, UdmaSqOpcode::UDMA_OPC_WRITE, slicePos);
            ProcessOneWqe(&sqe, UdmaSqOpcode::UDMA_OPC_WRITE, stream);

            // 按需更新wqe tasks
            UpdateWqeTasks(sqe);
        },
        dataType);

    out.pi = pi;
    HCCL_INFO("[UbConnLite::%s] end, ConnLiteOperationOut.pi = %u, conn[%s]", __func__, out.pi, Describe().c_str());
}

void UbConnLite::WriteWithNotify(
    const RmaBufSliceLite& loc, const RmtRmaBufSliceLite& rmt, const SqeConfigLite& cfg, ConnLiteOperationOut& out,
    const RmtRmaBufSliceLite& notify, const StreamLite& stream, u64 notifyData)
{
    HCCL_INFO("[UbConnLite::%s] start", __func__);

    ProcessSlicesWithNotify(
        loc, rmt, maxWriteSize,
        [&](const RmaBufSliceLite& locSlice, const RmtRmaBufSliceLite& rmtSlice, SlicePosition slicePos) {
            UdmaSqeWrite sqe{};
            FillOneSqeWrite(locSlice, rmtSlice, cfg, &sqe, UdmaSqOpcode::UDMA_OPC_WRITE, slicePos);
            ProcessOneWqe(&sqe, UdmaSqOpcode::UDMA_OPC_WRITE, stream);

            // 按需更新wqe tasks
            UpdateWqeTasks(sqe);
        },
        [&](const RmaBufSliceLite& locSlice, const RmtRmaBufSliceLite& rmtSlice, SlicePosition slicePos) {
            UdmaSqeWriteWithNotify sqe{};
            FillOneWqeWithNotify(locSlice, rmtSlice, cfg, &sqe, notify, notifyData, WRITE_WITH_NOTIFY_OPCODE, slicePos);
            ProcessOneWqeWithNotify(&sqe, WRITE_WITH_NOTIFY_OPCODE, stream);

            // 按需更新wqe tasks
            UpdateWqeTasks(sqe);
        });

    out.pi = pi;
    HCCL_INFO("[UbConnLite::%s] end, ConnLiteOperationOut.pi = %u, conn[%s]", __func__, out.pi, Describe().c_str());
}

void UbConnLite::WriteReduceWithNotify(
    DataType dataType, ReduceOp reduceOp, const RmaBufSliceLite& loc, const RmtRmaBufSliceLite& rmt,
    const SqeConfigLite& cfg, const StreamLite& stream, ConnLiteOperationOut& out, const RmtRmaBufSliceLite& notify,
    u64 notifyData)
{
    HCCL_INFO("[UbConnLite::%s] start", __func__);

    ProcessSlicesWithNotify(
        loc, rmt, maxWriteSize,
        [&](const RmaBufSliceLite& locSlice, const RmtRmaBufSliceLite& rmtSlice, SlicePosition slicePos) {
            UdmaSqeWrite sqe{};
            FillCommSqeReduceInfo(sqe.comm, reduceOp, dataType);
            FillOneSqeWrite(locSlice, rmtSlice, cfg, &sqe, UdmaSqOpcode::UDMA_OPC_WRITE, slicePos);
            ProcessOneWqe(&sqe, UdmaSqOpcode::UDMA_OPC_WRITE, stream);

            // 按需更新wqe tasks
            UpdateWqeTasks(sqe);
        },
        [&](const RmaBufSliceLite& locSlice, const RmtRmaBufSliceLite& rmtSlice, SlicePosition slicePos) {
            UdmaSqeWriteWithNotify sqe{};
            FillCommSqeReduceInfo(sqe.comm, reduceOp, dataType);
            FillOneWqeWithNotify(locSlice, rmtSlice, cfg, &sqe, notify, notifyData, WRITE_WITH_NOTIFY_OPCODE, slicePos);
            ProcessOneWqeWithNotify(&sqe, WRITE_WITH_NOTIFY_OPCODE, stream);

            // 按需更新wqe tasks
            UpdateWqeTasks(sqe);
        },
        dataType);

    out.pi = pi;
    HCCL_INFO("[UbConnLite::%s] end, ConnLiteOperationOut.pi = %u, conn[%s]", __func__, out.pi, Describe().c_str());
}

void UbConnLite::CustomizeSqeByOneSidedComm(UdmaSqeCommon* sqe, bool isLastWqe) const
{
    /* 表示SQE是否需要上报CQE:为1表示此SQE需要上报CQE，为0表示不需要 */
    sqe->cqe = isLastWqe;

    /* 2’b00:No order，表示当前报文与其他报文无保序要求
       2’b01:Relax Order，表示当前报文与后续的Strong Order报文有保序要求，strong order报文不能超越relax order报文执行。
       2’b10：Strong Order，表示当前报文有保序要求，该报文与前面的Relax Order报文有保序要求。
       2’b11：Reserved。
    */
    sqe->placeOdr = (isLastWqe == true ? 0x02 : 0x01);

    /* ODR[2]表示请求报文在目的端的completion order属性，表示当前报文和前面报文是否存在completion序：
       1’b0 :no order，表示当前报文和前面报文没有completion序要求，报文对应的CQE可以乱序上报。
       1’b1 :表示当前报文和前面报文有completion序要求，报文对应的CQE需要保序上报
    */
    sqe->compOrder = 1;

    /* 表示是否使能fence保序。为1时表示使能，为0时表示不使能。对于send/write/atomic SQE
       当fence为1时需要等待前面所有read和Atomic完成才开始执行，即等待前面发出的read或Atomic接收到所有response
    */
    sqe->fence = (isLastWqe == true ? 0x01 : 0x00);

    HCCL_INFO(
        "UbConnLite CustomizeSqeByOneSidedComm sqe->cqe =%u, sqe->placeOdr = %u sqe->compOrder =%u, sqe->fence = %u",
        sqe->cqe, sqe->placeOdr, sqe->compOrder, sqe->fence);
}

void UbConnLite::FillBatchOneWqe(
    const RmaBufSliceLite& loc, const RmtRmaBufSliceLite& rmt, const SqeConfigLite& cfg, bool isLastWqe, u32 opCode,
    const StreamLite& stream)
{
    (void)stream;
    HCCL_INFO("UbConnLite FillBatchOneWqe start, loc[%s], rmt[%s]", loc.Describe().c_str(), rmt.Describe().c_str());

    u32 sqOffset = pi % sqDepth_;
    pi = pi + 1;
    if (UNLIKELY(pi > sqDepth_)) {
        pi = pi % sqDepth_;
    }

    // 写入wqe数据到out.data
    UdmaSqeWrite sqe{};
    sqe.comm.inlineEn = 0;
    FillCommSqe(&(sqe.comm), rmt, cfg, opCode);
    FillLocalSgeSqe(&(sqe.u.sge), loc);

    if (UNLIKELY(sqe.u.sge.length == 0)) {
        sqe.comm.sgeNum = 0;
    }

    CustomizeSqeByOneSidedComm(&(sqe.comm), isLastWqe);

    HCCL_INFO("UbConnLite BatchWrite cp data to va %llu, pi %u", sqVa_, pi);
    u8* va = ReinterpretAs<u8*>(sqVa_ + sqOffset * SQE_SIZE_64);
    if (dwqeCacheLocked_ == false) {
        auto ret = memcpy_sp(va, SQE_SIZE_64, &sqe, sizeof(UdmaSqeWrite));
        if (UNLIKELY(ret != 0)) {
            HCCL_ERROR("UbConnLite::BatchWrite FillCommSqe memcpy failed, ret=%d", ret);
            THROW<InternalException>(StringFormat("UbConnLite::BatchWrite memcpy_sp failed, ret = %d", ret));
        }
    }
    HCCL_INFO("UbConnLite BatchWrite cp data to va end va(%p)", va);

    // 按需更新wqe tasks
    UpdateWqeTasks(sqe);
}

void UbConnLite::BatchProcessOneSlice(
    const RmaBufSliceLite& loc, const RmtRmaBufSliceLite& rmt, const SqeConfigLite& cfg, u32 maxSliceSize,
    bool isLastSlice, u32 opCode, const StreamLite& stream)
{
    u64 dataSize = loc.GetSize();
    // 按照UDMA能力切分数据
    bool isLastWqe;
    u64 offset = 0;

    // 使用整数除法和取余运算优化循环
    u64 numIterations = dataSize / maxSliceSize;
    u64 remainingSize = dataSize % maxSliceSize;

    for (u64 i = 0; i < numIterations; ++i) {
        isLastWqe = false;
        if ((remainingSize == 0) && (i == numIterations - 1) && isLastSlice) {
            isLastWqe = true;
        }

        // 构造本次wqe的log和rmt RmaBufSilce
        RmaBufSliceLite locTmp(loc.GetAddr() + offset, UB_DMA_MAX_READ_WEITE_SIZE, loc.GetLkey(), loc.GetTokenId());
        RmtRmaBufSliceLite rmtTmp(
            rmt.GetAddr() + offset, UB_DMA_MAX_READ_WEITE_SIZE, rmt.GetRkey(), rmt.GetTokenId(), rmt.GetTokenValue(),
            UINT32_MAX);

        FillBatchOneWqe(locTmp, rmtTmp, cfg, isLastWqe, opCode, stream);

        offset += UB_DMA_MAX_READ_WEITE_SIZE;
    }

    // 处理剩余的数据
    if (remainingSize > 0 && isLastSlice) {
        isLastWqe = true;

        RmaBufSliceLite locTmp(loc.GetAddr() + offset, remainingSize, loc.GetLkey(), loc.GetTokenId());
        RmtRmaBufSliceLite rmtTmp(
            rmt.GetAddr() + offset, remainingSize, rmt.GetRkey(), rmt.GetTokenId(), rmt.GetTokenValue(), UINT32_MAX);
        FillBatchOneWqe(locTmp, rmtTmp, cfg, isLastWqe, opCode, stream);
    }
}

void UbConnLite::BatchCommDataProcess(
    const vector<RmaBufSliceLite>& loc, const vector<RmtRmaBufSliceLite>& rmt, const SqeConfigLite& cfg,
    u32 maxSliceSize, u32 opCode, const StreamLite& stream)
{
    u64 siliceSize = loc.size();
    // 按照UDMA能力切分数据, 组装wqe
    for (u64 i = 0; i < siliceSize; i++) {
        BatchProcessOneSlice(loc[i], rmt[i], cfg, maxSliceSize, (i == (siliceSize - 1)), opCode, stream);
    }

    return;
}

void UbConnLite::BatchOneSidedRead(
    const vector<RmaBufSliceLite>& loc, const vector<RmtRmaBufSliceLite>& rmt, const SqeConfigLite& cfg,
    const StreamLite& stream, ConnLiteOperationOut& out)
{
    // 按照UDMA能力切分数据, 组装wqe
    BatchCommDataProcess(loc, rmt, cfg, maxReadSize, UdmaSqOpcode::UDMA_OPC_READ, stream);

    // 更新connlite的输出信息
    out.pi = pi;
    HCCL_INFO("UbConnLite BatchRead end, out.pi = %u", out.pi);
}

void UbConnLite::BatchOneSidedWrite(
    const vector<RmaBufSliceLite>& loc, const vector<RmtRmaBufSliceLite>& rmt, const SqeConfigLite& cfg,
    const StreamLite& stream, ConnLiteOperationOut& out)
{
    // 按照UDMA能力切分数据, 组装wqe
    BatchCommDataProcess(loc, rmt, cfg, maxWriteSize, UdmaSqOpcode::UDMA_OPC_WRITE, stream);

    // 更新connlite的输出信息
    out.pi = pi;
    HCCL_INFO("UbConnLite BatchWrite end, out.pi = %u", out.pi);
}

std::string UbConnLite::Describe()
{
    return StringFormat(
        "UbConnLite[dieId=%u, funcId=%u, jettyId=%u, dbAddr=0x%llx, sqVa=0x%llx, sqDepth=%u, "
        "jfcPollMode=%u, tpn=%u, dwqeCacheLocked=%d, locEid=%s, rmtEid=%s,jettyPi=%u, jettyCi=%u]",
        dieId_, funcId_, jettyId_, dbAddr_, sqVa_, sqDepth_, jfcPollMode_, tpn_, dwqeCacheLocked_,
        Bytes2hex(locEid_.raw, sizeof(locEid_.raw)).c_str(), Bytes2hex(rmtEid_.raw, sizeof(rmtEid_.raw)).c_str(), pi,
        ci);
}

constexpr uint32_t UB_WQE_NUM_PER_SQE = 4; // URMA约束每个SQE包含4个WQEBB
UbConnLite::UbConnLite(const UbConnLiteParam& liteParam)
{
    HCCL_INFO("[UbConnLite::%s] liteParam[%s]", __func__, liteParam.Describe().c_str());
    dieId_ = liteParam.dieId;
    funcId_ = liteParam.funcId;
    jettyId_ = liteParam.jettyId;
    dbAddr_ = liteParam.dbAddr;
    sqVa_ = liteParam.sqVa;
    // host侧创建jetty指定的sqDepth为sqeBBNum,device侧需要感知wqebbnum,URMA约束每个SQE包含4个WQEBB
    sqDepth_ = liteParam.sqDepth * UB_WQE_NUM_PER_SQE;
    dwqeCacheLocked_ = liteParam.dwqeCacheLocked;
    jfcPollMode_ = liteParam.jfcPollMode;
    tpn_ = liteParam.tpn;

    maxReadSize = liteParam.maxReadSize;
    maxWriteSize = liteParam.maxWriteSize;

    (void)memcpy_sp(rmtEid_.raw, URMA_EID_LEN, liteParam.rmtEid.raw, URMA_EID_LEN);
    (void)memcpy_sp(locEid_.raw, URMA_EID_LEN, liteParam.locEid.raw, URMA_EID_LEN);
    rmtReverseEid_ = rmtEid_.Reversed();
    jettyHandle_ = liteParam.jettyHandle;
    HCCL_INFO("%s", Describe().c_str());
}

UbConnLite::UbConnLite(const UbJettyLiteId& id, const UbJettyLiteAttr& attr, const Eid& rmtInfo)
    : RmaConnLite(id, attr, rmtInfo),
      maxReadSize(UB_DMA_MAX_READ_WEITE_SIZE),
      maxWriteSize(UB_DMA_MAX_READ_WEITE_SIZE)
{
    rmtReverseEid_ = rmtEid_.Reversed();
}

std::string UbConnLiteParam::Describe() const
{
    return StringFormat(
        "UbConnLiteParam[dieId=%u, funcId=%u, jettyId=%u, dbAddr=0x%llx, sqVa=0x%llx, sqDepth=%u, "
        "jfcPollMode=%u, tpn=%u, dwqeCacheLocked=%d, sqCiAddr=0x%llx, rmtEid=%s, localEid=%s, "
        "maxReadSize=%u, maxWriteSize=%u, jettyHandle=%llu]",
        dieId, funcId, jettyId, dbAddr, sqVa, sqDepth, jfcPollMode, tpn, dwqeCacheLocked, sqCiAddr,
        Bytes2hex(rmtEid.raw, sizeof(rmtEid.raw)).c_str(), Bytes2hex(locEid.raw, sizeof(locEid.raw)).c_str(),
        maxReadSize, maxWriteSize, jettyHandle);
}

UbConnLiteParam::UbConnLiteParam(std::vector<char>& uniqueId)
{
    BinaryStream binaryStream(uniqueId);
    binaryStream >> dieId;
    binaryStream >> funcId;
    binaryStream >> jettyId;

    binaryStream >> jfcPollMode;
    binaryStream >> dwqeCacheLocked;
    binaryStream >> dbAddr;
    binaryStream >> sqCiAddr;
    binaryStream >> sqVa;
    binaryStream >> sqDepth;
    binaryStream >> tpn;
    binaryStream >> rmtEid.raw;
    binaryStream >> locEid.raw;
    binaryStream >> maxReadSize;
    binaryStream >> maxWriteSize;
    binaryStream >> jettyHandle;

    static auto lastPrintTime = std::chrono::steady_clock::now();
    const auto now = std::chrono::steady_clock::now();
    const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - lastPrintTime).count();
    if (UNLIKELY(duration >= MAX_LOG_TIMEOUT_MS)) {
        HCCL_INFO("%s", Describe().c_str());
        lastPrintTime = now;
    }
    HCCL_INFO(
        "[UbConnLiteParam::%s] locEid[%s], rmtEid[%s]", __func__, locEid.Describe().c_str(), rmtEid.Describe().c_str());
}

} // namespace Hccl
