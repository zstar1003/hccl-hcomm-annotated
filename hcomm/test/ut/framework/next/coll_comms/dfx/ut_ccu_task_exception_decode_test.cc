/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include <mockcpp/mockcpp.hpp>
#include <cstring>
#include <cstdint>
#include <sstream>
#include <string>
#include <sys/mman.h>
#include <unistd.h>
#include "rt_external_base.h"

#define private public
#define protected public
#include "ccuTaskException.h"
#include "ccu_error_info_v1.h"
#include "hcomm_adapter_hccp.h"
#undef protected
#undef private

namespace hcomm {

struct CcuMissionInfo;
struct CcuLoopInfo;
struct CcuVersionOps {
    const char* name;
    void (*printCcumDfxInfo)(const void* rawData, std::ostringstream& oss);
    HcclResult (*getMissionInfo)(const void* rawData, CcuMissionInfo* out);
    HcclResult (*getLoopInfo)(const void* rawData, CcuLoopInfo* out);
};

HcclResult GetCcuOps(const CcuVersionOps*& ops);

struct TestCcumDfxInfoV1 {
    unsigned int queryResult;
    unsigned int ccumSqeRecvCnt;
    unsigned int ccumSqeSendCnt;
    unsigned int ccumMissionDfx;
    unsigned int ccumSqeDropCnt;
    unsigned int ccumSqeAddrLenErrDropCnt;
    unsigned int lqcCcuSecReg0;
    unsigned int ccumTifSqeCnt;
    unsigned int ccumTifCqeCnt;
    unsigned int ccumCifSqeCnt;
    unsigned int ccumCifCqeCnt;
};

// 需要手动修改测试数据结构与正式代码ccumDfxInfoV2一致 且满足ABI要求
struct TestCcumDfxInfoV2 {
    union {
        struct {
            unsigned int queryResult : 1; // 0:success, 1:fail
            unsigned int sqeRecvCnt : 1;
            unsigned int sqeSendCnt : 1;
            unsigned int missionDfx : 1;
            unsigned int sqeDropCnt : 1;
            unsigned int sqeErrDropCnt : 1;
            unsigned int secReg0 : 1;
            unsigned int tifSqeCnt : 1;
            unsigned int tifCqeCnt : 1;
            unsigned int cifSqeCnt : 1;
            unsigned int cifCqeCnt : 1;
            unsigned int mcmDfx : 1;
            unsigned int resv : 20;
        } bs;
        unsigned int validBits : 32;
    };
    union {
        struct {
            unsigned int ccumSqeRecvCnt;
            unsigned int ccumSqeSendCnt;
            unsigned int ccumMissionDfx;
            unsigned int ccumSqeDropCnt;
            unsigned int ccumSqeAddrLenErrDropCnt;
            unsigned int lqcCcuSecReg0;
            unsigned int ccumTifSqeCnt;
            unsigned int ccumTifCqeCnt;
            unsigned int ccumCifSqeCnt;
            unsigned int ccumCifCqeCnt;
            unsigned int ccumMcmDfx;
            unsigned int resv[20];
        } dfxInfo;
        unsigned int regVal[31];
    };
};
static_assert(sizeof(TestCcumDfxInfoV1) <= 128U, "TestCcumDfxInfoV1 layout changed unexpectedly");
static_assert(sizeof(TestCcumDfxInfoV2) <= 128U, "TestCcumDfxInfoV2 must mirror 128-byte raw layout");

// ABI tripwires: mission/loop raw 协议固定 32 个 16-bit part = 64 字节。
// 任一 schema struct 尺寸偏离 raw 协议都会编译期失败，提示重新评估 raw helper 容量。
// 放在 UT 文件而非生产 .cc：上库必经 UT 编译，效果等价且不影响生产 TU。
constexpr size_t CCU_CTX_RAW_CAPACITY = 64;
static_assert(
    sizeof(CcuMissionContext) == CCU_CTX_RAW_CAPACITY,
    "CcuMissionContext size must match CCU raw ABI (32 x u16 = 64B)");
static_assert(
    sizeof(CcuMissionContextV2) == CCU_CTX_RAW_CAPACITY,
    "CcuMissionContextV2 size must match CCU raw ABI (32 x u16 = 64B)");
static_assert(
    sizeof(CcuLoopContext) == CCU_CTX_RAW_CAPACITY, "CcuLoopContext size must match CCU raw ABI (32 x u16 = 64B)");
static_assert(
    sizeof(CcuLoopContextV2) == CCU_CTX_RAW_CAPACITY, "CcuLoopContextV2 size must match CCU raw ABI (32 x u16 = 64B)");

struct CcuMissionInfo {
    uint16_t currentIns;
    uint16_t endIns;
    uint16_t startIns;
};

// 必须与生产 hcomm::CcuLoopInfo 字段顺序/类型严格一致（ODR）。
// H1 已从生产侧删除 currentIns，UT 镜像同步删除，避免将来若 UT 真去读字段时错位。
struct CcuLoopInfo {
    uint16_t currentCnt;
    uint32_t addrStride;
};

static std::string FormatPanicLogWithOps(DevType deviceType, const uint8_t* panicLog)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(deviceType)).will(returnValue(HCCL_SUCCESS));

    const CcuVersionOps* ops = nullptr;
    const HcclResult ret = GetCcuOps(ops);
    if (ret != HCCL_SUCCESS || ops == nullptr || ops->printCcumDfxInfo == nullptr) {
        ADD_FAILURE() << "FormatPanicLogWithOps failed to resolve printer ops, ret=" << ret;
        return "";
    }

    std::ostringstream oss;
    ops->printCcumDfxInfo(panicLog, oss);
    return oss.str();
}

class CcuTaskExceptionDecodeTest : public testing::Test {
protected:
    void TearDown() override { GlobalMockObject::verify(); }
};

TEST_F(CcuTaskExceptionDecodeTest, Ut_PrintPanicLogInfo_WhenNullPtr_ExpectNoCrash)
{
    EXPECT_NO_FATAL_FAILURE(CcuTaskException::PrintPanicLogInfo(nullptr));
}

TEST_F(CcuTaskExceptionDecodeTest, Ut_GetCcuOps_WhenDeviceTypeIs950_ExpectV1Ops)
{
    DevType deviceType = DevType::DEV_TYPE_950;
    MOCKER(hrtGetDeviceType).stubs().with(outBound(deviceType)).will(returnValue(HCCL_SUCCESS));

    const CcuVersionOps* ops = nullptr;
    EXPECT_EQ(GetCcuOps(ops), HCCL_SUCCESS);
    ASSERT_NE(ops, nullptr);
    EXPECT_STREQ(ops->name, "CCU_V1");
}

TEST_F(CcuTaskExceptionDecodeTest, Ut_GetCcuOps_WhenDeviceTypeIs960_ExpectV2Ops)
{
    DevType deviceType = DevType::DEV_TYPE_960;
    MOCKER(hrtGetDeviceType).stubs().with(outBound(deviceType)).will(returnValue(HCCL_SUCCESS));

    const CcuVersionOps* ops = nullptr;
    EXPECT_EQ(GetCcuOps(ops), HCCL_SUCCESS);
    ASSERT_NE(ops, nullptr);
    EXPECT_STREQ(ops->name, "CCU_V2");
}

TEST_F(CcuTaskExceptionDecodeTest, Ut_GetCcuOps_WhenDeviceTypeQueryFails_ExpectErrorReturned)
{
    MOCKER(hrtGetDeviceType).stubs().with(any()).will(returnValue(HCCL_E_PARA));

    const CcuVersionOps* ops = nullptr;
    EXPECT_EQ(GetCcuOps(ops), HCCL_E_PARA);
    EXPECT_EQ(ops, nullptr);
}

TEST_F(CcuTaskExceptionDecodeTest, Ut_GetCcuOps_WhenSwitchDeviceTypeInProcess_ExpectNoCrossCasePollution)
{
    DevType deviceTypeV1 = DevType::DEV_TYPE_950;
    DevType deviceTypeV2 = DevType::DEV_TYPE_960;
    MOCKER(hrtGetDeviceType).expects(once()).with(outBound(deviceTypeV1)).will(returnValue(HCCL_SUCCESS));
    const CcuVersionOps* opsV1 = nullptr;
    const CcuVersionOps* opsV2 = nullptr;
    EXPECT_EQ(GetCcuOps(opsV1), HCCL_SUCCESS);
    GlobalMockObject::verify();
    MOCKER(hrtGetDeviceType).expects(once()).with(outBound(deviceTypeV2)).will(returnValue(HCCL_SUCCESS));
    EXPECT_EQ(GetCcuOps(opsV2), HCCL_SUCCESS);
    ASSERT_NE(opsV1, nullptr);
    ASSERT_NE(opsV2, nullptr);
    EXPECT_STREQ(opsV1->name, "CCU_V1");
    EXPECT_STREQ(opsV2->name, "CCU_V2");
    EXPECT_NE(opsV1, opsV2);
}

TEST_F(CcuTaskExceptionDecodeTest, Ut_PrintCcumDfxInfoV1_WhenDeviceTypeIs950_ExpectReadableString)
{
    uint8_t panicLog[sizeof(TestCcumDfxInfoV1)];
    std::memset(panicLog, 0, sizeof(panicLog));
    TestCcumDfxInfoV1* info = reinterpret_cast<TestCcumDfxInfoV1*>(panicLog);
    info->ccumSqeRecvCnt = 11;
    info->ccumSqeSendCnt = 12;
    info->ccumMissionDfx = 13;
    info->ccumTifSqeCnt = 14;
    info->ccumTifCqeCnt = 15;
    info->ccumCifSqeCnt = 16;
    info->ccumCifCqeCnt = 17;
    info->ccumSqeDropCnt = 18;
    info->ccumSqeAddrLenErrDropCnt = 19;
    info->lqcCcuSecReg0 = 1;

    const std::string output = FormatPanicLogWithOps(DevType::DEV_TYPE_950, panicLog);
    const std::string expected = " SQE_RECV_CNT[11] SQE_SEND_CNT[12] MISSION_DFX[13] "
                                 "TIF_SQE_CNT[14] TIF_CQE_CNT[15] CIF_SQE_CNT[16] CIF_CQE_CNT[17] SQE_DROP_CNT[18] "
                                 "SQE_ADDR_LEN_ERR_DROP_CNT[19] ccumIsEnable[1]";
    EXPECT_EQ(output, expected);
    EXPECT_EQ(output.find("INVALID(hardware)"), std::string::npos);
}

TEST_F(CcuTaskExceptionDecodeTest, Ut_PrintCcumDfxInfoV1_WhenQueryResultNonZero_ExpectStillPrintable)
{
    uint8_t panicLog[sizeof(TestCcumDfxInfoV1)];
    std::memset(panicLog, 0, sizeof(panicLog));
    TestCcumDfxInfoV1* info = reinterpret_cast<TestCcumDfxInfoV1*>(panicLog);
    info->queryResult = 1;
    info->ccumSqeRecvCnt = 41;
    info->ccumSqeSendCnt = 42;
    info->ccumMissionDfx = 43;
    info->ccumSqeDropCnt = 44;
    info->ccumSqeAddrLenErrDropCnt = 45;
    info->lqcCcuSecReg0 = 1;
    info->ccumTifSqeCnt = 46;
    info->ccumTifCqeCnt = 47;
    info->ccumCifSqeCnt = 48;
    info->ccumCifCqeCnt = 49;

    const std::string output = FormatPanicLogWithOps(DevType::DEV_TYPE_950, panicLog);
    EXPECT_NE(output.find("SQE_RECV_CNT[41]"), std::string::npos);
    EXPECT_NE(output.find("SQE_SEND_CNT[42]"), std::string::npos);
    EXPECT_NE(output.find("ccumIsEnable[1]"), std::string::npos);
}

TEST_F(CcuTaskExceptionDecodeTest, Ut_PrintCcumDfxInfoV2_WhenDeviceTypeIs960_ExpectReadableString)
{
    // panicLog 严格按 RTS ABI 大小分配，不要用 sizeof(TestCcumDfxInfoV2) 掩盖生产越界读。
    static_assert(
        sizeof(TestCcumDfxInfoV2) <= MAX_CCU_EXCEPTION_INFO_SIZE,
        "TestCcumDfxInfoV2 must not exceed panicLog raw buffer");
    uint8_t panicLog[MAX_CCU_EXCEPTION_INFO_SIZE];
    std::memset(panicLog, 0, sizeof(panicLog));
    TestCcumDfxInfoV2* info = reinterpret_cast<TestCcumDfxInfoV2*>(panicLog);
    info->validBits = 0xFFEU;
    info->dfxInfo.ccumSqeRecvCnt = 21;
    info->dfxInfo.ccumSqeSendCnt = 22;
    info->dfxInfo.ccumMissionDfx = 23;
    info->dfxInfo.ccumTifSqeCnt = 24;
    info->dfxInfo.ccumTifCqeCnt = 25;
    info->dfxInfo.ccumCifSqeCnt = 26;
    info->dfxInfo.ccumCifCqeCnt = 27;
    info->dfxInfo.ccumSqeDropCnt = 28;
    info->dfxInfo.ccumSqeAddrLenErrDropCnt = 29;
    info->dfxInfo.lqcCcuSecReg0 = 1;
    info->dfxInfo.ccumMcmDfx = 30;

    const std::string output = FormatPanicLogWithOps(DevType::DEV_TYPE_960, panicLog);
    const std::string expected = " SQE_RECV_CNT[21] SQE_SEND_CNT[22] MISSION_DFX[23] "
                                 "TIF_SQE_CNT[24] TIF_CQE_CNT[25] CIF_SQE_CNT[26] CIF_CQE_CNT[27] SQE_DROP_CNT[28] "
                                 "SQE_ADDR_LEN_ERR_DROP_CNT[29] ccumIsEnable[1] MCM_DFX[30]";
    EXPECT_EQ(output, expected);
    EXPECT_EQ(output.find("INVALID(hardware)"), std::string::npos);
}

TEST_F(CcuTaskExceptionDecodeTest, Ut_PrintCcumDfxInfoV2_WhenQueryResultNonZero_ExpectStillPrintable)
{
    uint8_t panicLog[MAX_CCU_EXCEPTION_INFO_SIZE];
    std::memset(panicLog, 0, sizeof(panicLog));
    TestCcumDfxInfoV2* info = reinterpret_cast<TestCcumDfxInfoV2*>(panicLog);
    info->validBits = 0xFFFFU;
    info->bs.queryResult = 1;
    info->dfxInfo.ccumSqeRecvCnt = 51;
    info->dfxInfo.ccumSqeSendCnt = 52;
    info->dfxInfo.ccumMissionDfx = 53;
    info->dfxInfo.ccumSqeDropCnt = 54;
    info->dfxInfo.ccumSqeAddrLenErrDropCnt = 55;
    info->dfxInfo.lqcCcuSecReg0 = 1;
    info->dfxInfo.ccumTifSqeCnt = 56;
    info->dfxInfo.ccumTifCqeCnt = 57;
    info->dfxInfo.ccumCifSqeCnt = 58;
    info->dfxInfo.ccumCifCqeCnt = 59;
    info->dfxInfo.ccumMcmDfx = 60;

    const std::string output = FormatPanicLogWithOps(DevType::DEV_TYPE_960, panicLog);
    EXPECT_NE(output.find("SQE_RECV_CNT[51]"), std::string::npos);
    EXPECT_NE(output.find("MCM_DFX[60]"), std::string::npos);
}

TEST_F(CcuTaskExceptionDecodeTest, Ut_GetCcuMissionAndLoopInfoV1_WhenRawIsValid_ExpectDecodedFields)
{
    DevType deviceType = DevType::DEV_TYPE_950;
    MOCKER(hrtGetDeviceType).stubs().with(outBound(deviceType)).will(returnValue(HCCL_SUCCESS));

    const CcuVersionOps* ops = nullptr;
    ASSERT_EQ(GetCcuOps(ops), HCCL_SUCCESS);
    ASSERT_NE(ops, nullptr);
    ASSERT_NE(ops->getMissionInfo, nullptr);
    ASSERT_NE(ops->getLoopInfo, nullptr);

    CcuMissionContext missionCtx{};
    missionCtx.part4.currentIns = 0x123;
    missionCtx.part5.currentIns = 0x5;
    missionCtx.part6.endIns = 0x4;
    missionCtx.part5.endIns = 0x155;
    missionCtx.part6.startIns = 0x11;
    missionCtx.part7.startIns = 0x3;

    CcuMissionInfo missionInfo{};
    ASSERT_EQ(ops->getMissionInfo(&missionCtx, &missionInfo), HCCL_SUCCESS);
    EXPECT_EQ(missionInfo.currentIns, static_cast<uint16_t>(0x2923));
    EXPECT_EQ(missionInfo.endIns, static_cast<uint16_t>(0x2155));
    EXPECT_EQ(missionInfo.startIns, static_cast<uint16_t>(0x1811));

    CcuLoopContext loopCtx{};
    loopCtx.part13.currentCnt = 0x234;
    loopCtx.part14.currentCnt = 1;
    loopCtx.part10.addrStride = 0x15;
    loopCtx.part11.addrStride = 0x2345;
    loopCtx.part12.addrStride = 0x12A;

    CcuLoopInfo loopInfo{};
    ASSERT_EQ(ops->getLoopInfo(&loopCtx, &loopInfo), HCCL_SUCCESS);
    EXPECT_EQ(loopInfo.currentCnt, static_cast<uint16_t>(0x1234));
    const uint32_t expectedAddrStride
        = static_cast<uint32_t>(0x15) | (static_cast<uint32_t>(0x2345) << 6) | (static_cast<uint32_t>(0x12A) << 22);
    EXPECT_EQ(loopInfo.addrStride, expectedAddrStride);
}

TEST_F(CcuTaskExceptionDecodeTest, Ut_GetCcuMissionAndLoopInfoV2_WhenRawIsValid_ExpectDecodedFields)
{
    DevType deviceType = DevType::DEV_TYPE_960;
    MOCKER(hrtGetDeviceType).stubs().with(outBound(deviceType)).will(returnValue(HCCL_SUCCESS));

    const CcuVersionOps* ops = nullptr;
    ASSERT_EQ(GetCcuOps(ops), HCCL_SUCCESS);
    ASSERT_NE(ops, nullptr);
    ASSERT_NE(ops->getMissionInfo, nullptr);
    ASSERT_NE(ops->getLoopInfo, nullptr);

    CcuMissionContextV2 missionCtx{};
    missionCtx.part4.currentIns = 0x12A;
    missionCtx.part5.currentIns = 0x35;
    missionCtx.part5.endIns = 0x11A;
    missionCtx.part6.endIns = 0x5;
    missionCtx.part6.startIns = 0x111;
    missionCtx.part7.startIns = 0x3;

    CcuMissionInfo missionInfo{};
    ASSERT_EQ(ops->getMissionInfo(&missionCtx, &missionInfo), HCCL_SUCCESS);
    EXPECT_EQ(missionInfo.currentIns, static_cast<uint16_t>(0x6B2A));
    EXPECT_EQ(missionInfo.endIns, static_cast<uint16_t>(0xB1A));
    EXPECT_EQ(missionInfo.startIns, static_cast<uint16_t>(0x711));

    CcuLoopContextV2 loopCtx{};
    loopCtx.part10.currentCnt = 0x456;
    loopCtx.part5.addrStride = 0x1AA;
    loopCtx.part6.addrStride = 0x155;
    loopCtx.part7.addrStride = 0x3F;

    CcuLoopInfo loopInfo{};
    ASSERT_EQ(ops->getLoopInfo(&loopCtx, &loopInfo), HCCL_SUCCESS);
    EXPECT_EQ(loopInfo.currentCnt, static_cast<uint16_t>(0x456));
    const uint32_t expectedAddrStride
        = static_cast<uint32_t>(0x1AA) | (static_cast<uint32_t>(0x155) << 9) | (static_cast<uint32_t>(0x3F) << 25);
    EXPECT_EQ(loopInfo.addrStride, expectedAddrStride);
}

TEST_F(CcuTaskExceptionDecodeTest, Ut_PrintCcumDfxInfoV2_WhenFieldInvalid_ExpectInvalidMarkerWithoutRawValue)
{
    uint8_t panicLog[MAX_CCU_EXCEPTION_INFO_SIZE];
    std::memset(panicLog, 0, sizeof(panicLog));
    TestCcumDfxInfoV2* info = reinterpret_cast<TestCcumDfxInfoV2*>(panicLog);
    info->bs.sqeRecvCnt = 1;
    info->bs.missionDfx = 1;
    info->bs.tifSqeCnt = 1;
    info->bs.tifCqeCnt = 1;
    info->bs.cifSqeCnt = 1;
    info->bs.cifCqeCnt = 1;
    info->bs.sqeDropCnt = 1;
    info->bs.sqeErrDropCnt = 1;
    info->bs.secReg0 = 1;
    info->bs.mcmDfx = 1;
    info->dfxInfo.ccumSqeRecvCnt = 31;
    info->dfxInfo.ccumSqeSendCnt = 222;
    info->dfxInfo.ccumMissionDfx = 33;
    info->dfxInfo.ccumTifSqeCnt = 34;
    info->dfxInfo.ccumTifCqeCnt = 35;
    info->dfxInfo.ccumCifSqeCnt = 36;
    info->dfxInfo.ccumCifCqeCnt = 37;
    info->dfxInfo.ccumSqeDropCnt = 38;
    info->dfxInfo.ccumSqeAddrLenErrDropCnt = 39;
    info->dfxInfo.lqcCcuSecReg0 = 1;
    info->dfxInfo.ccumMcmDfx = 40;

    const std::string output = FormatPanicLogWithOps(DevType::DEV_TYPE_960, panicLog);
    EXPECT_NE(output.find("SQE_RECV_CNT[31]"), std::string::npos);
    EXPECT_NE(output.find("SQE_SEND_CNT[INVALID(hardware)]"), std::string::npos);
    EXPECT_EQ(output.find("SQE_SEND_CNT[222]"), std::string::npos);
    EXPECT_NE(output.find("MCM_DFX[40]"), std::string::npos);
}

TEST_F(CcuTaskExceptionDecodeTest, Ut_PrintCcumDfxInfoV1_WhenRawDataNull_ExpectReadableNullMarker)
{
    const std::string output = FormatPanicLogWithOps(DevType::DEV_TYPE_950, nullptr);
    EXPECT_EQ(output, " [rawData is null]");
}

TEST_F(CcuTaskExceptionDecodeTest, Ut_PrintCcumDfxInfoV2_WhenRawDataNull_ExpectReadableNullMarker)
{
    const std::string output = FormatPanicLogWithOps(DevType::DEV_TYPE_960, nullptr);
    EXPECT_EQ(output, " [rawData is null]");
}

// 专项校验 V2 panicLog 在页边界读取时不会越界。
// panicLog 紧贴第一页末尾，第二页设置为 PROT_NONE；若实现超出 128B 读取将触发 SIGSEGV。
TEST_F(CcuTaskExceptionDecodeTest, Ut_PrintCcumDfxInfoV2_WhenPanicLogAtPageBoundary_ExpectNoOOBRead)
{
    const long pageSizeRaw = sysconf(_SC_PAGESIZE);
    ASSERT_GT(pageSizeRaw, 0);
    const size_t pageSize = static_cast<size_t>(pageSizeRaw);
    ASSERT_GE(pageSize, MAX_CCU_EXCEPTION_INFO_SIZE);

    // 连续两页：第一页可读写，第二页 PROT_NONE 作为守护页。
    void* base = mmap(nullptr, pageSize * 2, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    ASSERT_NE(base, MAP_FAILED);
    ASSERT_EQ(mprotect(static_cast<uint8_t*>(base) + pageSize, pageSize, PROT_NONE), 0);

    // panicLog 紧贴第一页末尾（最后 MAX_CCU_EXCEPTION_INFO_SIZE 字节）。
    uint8_t* panicLog = static_cast<uint8_t*>(base) + pageSize - MAX_CCU_EXCEPTION_INFO_SIZE;
    std::memset(panicLog, 0, MAX_CCU_EXCEPTION_INFO_SIZE);

    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_960)).will(returnValue(HCCL_SUCCESS));
    const CcuVersionOps* ops = nullptr;
    ASSERT_EQ(GetCcuOps(ops), HCCL_SUCCESS);
    ASSERT_NE(ops, nullptr);
    ASSERT_NE(ops->printCcumDfxInfo, nullptr);

    std::ostringstream oss;
    EXPECT_NO_FATAL_FAILURE(ops->printCcumDfxInfo(panicLog, oss));

    ASSERT_EQ(munmap(base, pageSize * 2), 0);
}

TEST_F(CcuTaskExceptionDecodeTest, Ut_PrintPanicLogInfo_WhenDeviceTypeQueryFails_ExpectNoCrash)
{
    MOCKER(hrtGetDeviceType).stubs().with(any()).will(returnValue(HCCL_E_PARA));

    uint8_t panicLog[16] = {0};
    EXPECT_NO_FATAL_FAILURE(CcuTaskException::PrintPanicLogInfo(panicLog));
}

TEST_F(CcuTaskExceptionDecodeTest, Ut_GetCcuMissionContext_WhenGetDevicePhyIdFail_ExpectZeroInitializedContext)
{
    MOCKER(HccpRaTlvRequestForCustomChannel)
        .stubs()
        .with(any(), any(), any(), any())
        .will(returnValue(HCCL_E_INTERNAL));

    uint8_t raw[CCU_CTX_RAW_CAPACITY] = {0xAB};
    size_t copiedLen = 123;
    EXPECT_NE(CcuTaskException::GetCcuMissionContextRaw(0, 0, 0, raw, sizeof(raw), copiedLen), HCCL_SUCCESS);
    EXPECT_EQ(copiedLen, 0U);
    EXPECT_EQ(raw[0], 0U);
}

TEST_F(CcuTaskExceptionDecodeTest, Ut_GetCcuLoopContext_WhenGetDevicePhyIdFail_ExpectZeroInitializedContext)
{
    MOCKER(HccpRaTlvRequestForCustomChannel)
        .stubs()
        .with(any(), any(), any(), any())
        .will(returnValue(HCCL_E_INTERNAL));

    uint8_t raw[CCU_CTX_RAW_CAPACITY] = {0xCD};
    size_t copiedLen = 123;
    EXPECT_NE(CcuTaskException::GetCcuLoopContextRaw(0, 0, 0, raw, sizeof(raw), copiedLen), HCCL_SUCCESS);
    EXPECT_EQ(copiedLen, 0U);
    EXPECT_EQ(raw[0], 0U);
}

// ====== raw helper（方案 A）UT ======
// raw helper 是新引入的“去类型耦合”入口，作为生产路径的唯一拉 raw 通道，
// 必须覆盖：buffer 校验、devicePhy 失败、HccpRaTlvCcuCustomChannel 失败 三条边界。

TEST_F(CcuTaskExceptionDecodeTest, Ut_GetCcuMissionContextRaw_WhenBufNull_ExpectParaError)
{
    size_t copiedLen = 0;
    EXPECT_EQ(
        CcuTaskException::GetCcuMissionContextRaw(0, 0, 0, nullptr, CCU_CTX_RAW_CAPACITY, copiedLen), HCCL_E_PARA);
    EXPECT_EQ(copiedLen, 0U);
}

TEST_F(CcuTaskExceptionDecodeTest, Ut_GetCcuMissionContextRaw_WhenBufTooSmall_ExpectParaError)
{
    uint8_t buf[CCU_CTX_RAW_CAPACITY - 1] = {0};
    size_t copiedLen = 0;
    EXPECT_EQ(CcuTaskException::GetCcuMissionContextRaw(0, 0, 0, buf, sizeof(buf), copiedLen), HCCL_E_PARA);
    EXPECT_EQ(copiedLen, 0U);
}

TEST_F(CcuTaskExceptionDecodeTest, Ut_GetCcuMissionContextRaw_WhenGetDevicePhyIdFail_ExpectErrorAndZeroBuf)
{
    MOCKER(HccpRaTlvRequestForCustomChannel)
        .stubs()
        .with(any(), any(), any(), any())
        .will(returnValue(HCCL_E_INTERNAL));

    uint8_t buf[CCU_CTX_RAW_CAPACITY];
    std::memset(buf, 0xAB, sizeof(buf));
    size_t copiedLen = 0;
    EXPECT_NE(CcuTaskException::GetCcuMissionContextRaw(0, 0, 0, buf, sizeof(buf), copiedLen), HCCL_SUCCESS);
    EXPECT_EQ(copiedLen, 0U);
    for (size_t i = 0; i < sizeof(buf); ++i) {
        EXPECT_EQ(buf[i], 0U);
    }
}

TEST_F(CcuTaskExceptionDecodeTest, Ut_GetCcuLoopContextRaw_WhenBufTooSmall_ExpectParaError)
{
    uint8_t buf[CCU_CTX_RAW_CAPACITY - 1] = {0};
    size_t copiedLen = 0;
    EXPECT_EQ(CcuTaskException::GetCcuLoopContextRaw(0, 0, 0, buf, sizeof(buf), copiedLen), HCCL_E_PARA);
    EXPECT_EQ(copiedLen, 0U);
}

TEST_F(CcuTaskExceptionDecodeTest, Ut_GetCcuLoopContextRaw_WhenGetDevicePhyIdFail_ExpectErrorAndZeroBuf)
{
    MOCKER(HccpRaTlvRequestForCustomChannel)
        .stubs()
        .with(any(), any(), any(), any())
        .will(returnValue(HCCL_E_INTERNAL));

    uint8_t buf[CCU_CTX_RAW_CAPACITY];
    std::memset(buf, 0xCD, sizeof(buf));
    size_t copiedLen = 0;
    EXPECT_NE(CcuTaskException::GetCcuLoopContextRaw(0, 0, 0, buf, sizeof(buf), copiedLen), HCCL_SUCCESS);
    EXPECT_EQ(copiedLen, 0U);
    for (size_t i = 0; i < sizeof(buf); ++i) {
        EXPECT_EQ(buf[i], 0U);
    }
}

// HccpRaTlvCcuCustomChannel 失败时：raw helper 必须返回错误码、copiedLen 清零，
// 且 buffer 已在请求前被清零（不会保留调用方传入的旧数据）。
TEST_F(CcuTaskExceptionDecodeTest, Ut_GetCcuMissionContextRaw_WhenChannelFail_ExpectErrorAndZeroBuf)
{
    MOCKER(HccpRaTlvRequestForCustomChannel)
        .stubs()
        .with(any(), any(), any(), any())
        .will(returnValue(HCCL_E_INTERNAL));

    uint8_t buf[CCU_CTX_RAW_CAPACITY];
    std::memset(buf, 0xEF, sizeof(buf));
    size_t copiedLen = 123;
    EXPECT_NE(CcuTaskException::GetCcuMissionContextRaw(0, 0, 0, buf, sizeof(buf), copiedLen), HCCL_SUCCESS);
    EXPECT_EQ(copiedLen, 0U);
    for (size_t i = 0; i < sizeof(buf); ++i) {
        EXPECT_EQ(buf[i], 0U);
    }
}

TEST_F(CcuTaskExceptionDecodeTest, Ut_GetCcuLoopContextRaw_WhenChannelFail_ExpectErrorAndZeroBuf)
{
    MOCKER(HccpRaTlvRequestForCustomChannel)
        .stubs()
        .with(any(), any(), any(), any())
        .will(returnValue(HCCL_E_INTERNAL));

    uint8_t buf[CCU_CTX_RAW_CAPACITY];
    std::memset(buf, 0x12, sizeof(buf));
    size_t copiedLen = 123;
    EXPECT_NE(CcuTaskException::GetCcuLoopContextRaw(0, 0, 0, buf, sizeof(buf), copiedLen), HCCL_SUCCESS);
    EXPECT_EQ(copiedLen, 0U);
    for (size_t i = 0; i < sizeof(buf); ++i) {
        EXPECT_EQ(buf[i], 0U);
    }
}

TEST(Ut_CcuMissionContextV1, WhenStatusPartsSet_ExpectStatusCombinedCorrectly)
{
    CcuMissionContext ctx{};
    memset(&ctx, 0, sizeof(ctx));
    ctx.part2.status = 0x1FFF;
    ctx.part3.status = 0x7;

    uint16_t expected = static_cast<uint16_t>((0x7U << 13) | 0x1FFFU);
    EXPECT_EQ(ctx.GetStatus(), expected);
}

TEST(Ut_CcuMissionContextV1, WhenCurrentInsPartsSet_ExpectCurrentInsCombinedCorrectly)
{
    CcuMissionContext ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.part4.currentIns = 0x7FF;
    ctx.part5.currentIns = 0x1F;

    uint16_t expected = static_cast<uint16_t>((0x1FU << 11) | 0x7FFU);
    EXPECT_EQ(ctx.GetCurrentIns(), expected);
}

TEST(Ut_CcuMissionContextV1, WhenStartInsPartsSet_ExpectStartInsCombinedCorrectly)
{
    CcuMissionContext ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.part6.startIns = 0x7FF;
    ctx.part7.startIns = 0x1F;

    uint16_t expected = static_cast<uint16_t>((0x1FU << 11) | 0x7FFU);
    EXPECT_EQ(ctx.GetStartIns(), expected);
}

TEST(Ut_CcuMissionContextV1, WhenEndInsPartsSet_ExpectEndInsCombinedCorrectly)
{
    CcuMissionContext ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.part5.endIns = 0x7FF;
    ctx.part6.endIns = 0x1F;

    uint16_t expected = static_cast<uint16_t>((0x1FU << 11) | 0x7FFU);
    EXPECT_EQ(ctx.GetEndIns(), expected);
}

TEST(Ut_CcuMissionContextV2, WhenStatusPartsSet_ExpectStatusCombinedCorrectly)
{
    CcuMissionContextV2 ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.part2.statusLo = 0x1FFF;
    ctx.part3.statusHi = 0x7;

    uint16_t expected = static_cast<uint16_t>((0x7U << 13) | 0x1FFFU);
    EXPECT_EQ(ctx.GetStatus(), expected);
}

TEST(Ut_CcuMissionContextV2, WhenCurrentInsPartsSet_ExpectCurrentInsCombinedCorrectly)
{
    CcuMissionContextV2 ctx{};
    std::memset(&ctx, 0, sizeof(ctx));
    ctx.part4.currentIns = 0x1FF;
    ctx.part5.currentIns = 0x7F;

    uint16_t expected = static_cast<uint16_t>((0x7FU << 9) | 0x1FFU);
    EXPECT_EQ(ctx.GetCurrentIns(), expected);
}

TEST(Ut_CcuMissionContextV2, WhenStartInsPartsSet_ExpectStartInsCombinedCorrectly)
{
    CcuMissionContextV2 ctx{};
    std::memset(&ctx, 0, sizeof(ctx));
    ctx.part6.startIns = 0x1FF;
    ctx.part7.startIns = 0x7F;

    uint16_t expected = static_cast<uint16_t>((0x7FU << 9) | 0x1FFU);
    EXPECT_EQ(ctx.GetStartIns(), expected);
}

TEST(Ut_CcuMissionContextV2, WhenEndInsPartsSet_ExpectEndInsCombinedCorrectly)
{
    CcuMissionContextV2 ctx{};
    std::memset(&ctx, 0, sizeof(ctx));
    ctx.part5.endIns = 0x1FF;
    ctx.part6.endIns = 0x7F;

    uint16_t expected = static_cast<uint16_t>((0x7FU << 9) | 0x1FFU);
    EXPECT_EQ(ctx.GetEndIns(), expected);
}

TEST(Ut_CcuLoopContextV1, WhenCurrentInsPartsSet_ExpectCurrentInsCombinedCorrectly)
{
    CcuLoopContext ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.part9.currentIns = 0x3F;
    ctx.part10.currentIns = 0x3FF;

    uint16_t expected = static_cast<uint16_t>((0x3FFU << 6) | 0x3FU);
    EXPECT_EQ(ctx.GetCurrentIns(), expected);
}

TEST(Ut_CcuLoopContextV1, WhenCurrentCntPartsSet_ExpectCurrentCntCombinedCorrectly)
{
    CcuLoopContext ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.part13.currentCnt = 0xFFF;
    ctx.part14.currentCnt = 0x1;

    uint16_t expected = static_cast<uint16_t>((0x1U << 12) | 0xFFFU);
    EXPECT_EQ(ctx.GetCurrentCnt(), expected);
}

TEST(Ut_CcuLoopContextV1, WhenAddrStridePartsSet_ExpectAddrStrideCombinedCorrectly)
{
    CcuLoopContext ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.part10.addrStride = 0x3F;
    ctx.part11.addrStride = 0xFFFF;
    ctx.part12.addrStride = 0x3FF;

    uint32_t expected
        = (static_cast<uint32_t>(0x3FFU) << 22) | (static_cast<uint32_t>(0xFFFFU) << 6) | static_cast<uint32_t>(0x3FU);
    EXPECT_EQ(ctx.GetAddrStride(), expected);
}

TEST(Ut_CcuLoopContextV2, WhenCurrentInsPartsSet_ExpectCurrentInsCombinedCorrectly)
{
    CcuLoopContextV2 ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.part4.currentIns = 0x1FF;
    ctx.part5.currentIns = 0x7F;

    uint16_t expected = static_cast<uint16_t>((0x7FU << 9) | 0x1FFU);
    EXPECT_EQ(ctx.GetCurrentIns(), expected);
}

TEST(Ut_CcuLoopContextV2, WhenCurrentCntSet_ExpectCurrentCntReturnedDirectly)
{
    CcuLoopContextV2 ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.part10.currentCnt = 0x1FFF;

    EXPECT_EQ(ctx.GetCurrentCnt(), 0x1FFFU);
}

TEST(Ut_CcuLoopContextV2, WhenAddrStridePartsSet_ExpectAddrStrideCombinedCorrectly)
{
    CcuLoopContextV2 ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.part5.addrStride = 0x1FF;
    ctx.part6.addrStride = 0xFFFF;
    ctx.part7.addrStride = 0x7F;

    uint32_t expected
        = (static_cast<uint32_t>(0x7FU) << 25) | (static_cast<uint32_t>(0xFFFFU) << 9) | static_cast<uint32_t>(0x1FFU);
    EXPECT_EQ(ctx.GetAddrStride(), expected);
}

TEST_F(CcuTaskExceptionDecodeTest, Ut_GetCcuMissionInfoV1_WhenRawDataIsNull_ExpectErrorReturned)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));

    const CcuVersionOps* ops = nullptr;
    EXPECT_EQ(GetCcuOps(ops), HCCL_SUCCESS);
    ASSERT_NE(ops, nullptr);
    ASSERT_NE(ops->getMissionInfo, nullptr);

    CcuMissionInfo missionInfo{};
    EXPECT_NE(ops->getMissionInfo(nullptr, &missionInfo), HCCL_SUCCESS);
}

TEST_F(CcuTaskExceptionDecodeTest, Ut_GetCcuMissionInfoV1_WhenOutIsNull_ExpectErrorReturned)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));

    const CcuVersionOps* ops = nullptr;
    EXPECT_EQ(GetCcuOps(ops), HCCL_SUCCESS);
    ASSERT_NE(ops, nullptr);
    ASSERT_NE(ops->getMissionInfo, nullptr);

    CcuMissionContext ctx{};
    EXPECT_NE(ops->getMissionInfo(&ctx, nullptr), HCCL_SUCCESS);
}

TEST_F(CcuTaskExceptionDecodeTest, Ut_GetCcuLoopInfoV1_WhenRawDataIsNull_ExpectErrorReturned)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));

    const CcuVersionOps* ops = nullptr;
    EXPECT_EQ(GetCcuOps(ops), HCCL_SUCCESS);
    ASSERT_NE(ops, nullptr);
    ASSERT_NE(ops->getLoopInfo, nullptr);

    CcuLoopInfo loopInfo{};
    EXPECT_NE(ops->getLoopInfo(nullptr, &loopInfo), HCCL_SUCCESS);
}

TEST_F(CcuTaskExceptionDecodeTest, Ut_GetCcuLoopInfoV1_WhenOutIsNull_ExpectErrorReturned)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));

    const CcuVersionOps* ops = nullptr;
    EXPECT_EQ(GetCcuOps(ops), HCCL_SUCCESS);
    ASSERT_NE(ops, nullptr);
    ASSERT_NE(ops->getLoopInfo, nullptr);

    CcuLoopContext ctx{};
    EXPECT_NE(ops->getLoopInfo(&ctx, nullptr), HCCL_SUCCESS);
}

TEST_F(CcuTaskExceptionDecodeTest, Ut_GetCcuMissionInfoV2_WhenRawDataIsNull_ExpectErrorReturned)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_960)).will(returnValue(HCCL_SUCCESS));

    const CcuVersionOps* ops = nullptr;
    EXPECT_EQ(GetCcuOps(ops), HCCL_SUCCESS);
    ASSERT_NE(ops, nullptr);
    ASSERT_NE(ops->getMissionInfo, nullptr);

    CcuMissionInfo missionInfo{};
    EXPECT_NE(ops->getMissionInfo(nullptr, &missionInfo), HCCL_SUCCESS);
}

TEST_F(CcuTaskExceptionDecodeTest, Ut_GetCcuMissionInfoV2_WhenOutIsNull_ExpectErrorReturned)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_960)).will(returnValue(HCCL_SUCCESS));

    const CcuVersionOps* ops = nullptr;
    EXPECT_EQ(GetCcuOps(ops), HCCL_SUCCESS);
    ASSERT_NE(ops, nullptr);
    ASSERT_NE(ops->getMissionInfo, nullptr);

    CcuMissionContextV2 ctx{};
    EXPECT_NE(ops->getMissionInfo(&ctx, nullptr), HCCL_SUCCESS);
}

TEST_F(CcuTaskExceptionDecodeTest, Ut_GetCcuLoopInfoV2_WhenRawDataIsNull_ExpectErrorReturned)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_960)).will(returnValue(HCCL_SUCCESS));

    const CcuVersionOps* ops = nullptr;
    EXPECT_EQ(GetCcuOps(ops), HCCL_SUCCESS);
    ASSERT_NE(ops, nullptr);
    ASSERT_NE(ops->getLoopInfo, nullptr);

    CcuLoopInfo loopInfo{};
    EXPECT_NE(ops->getLoopInfo(nullptr, &loopInfo), HCCL_SUCCESS);
}

TEST_F(CcuTaskExceptionDecodeTest, Ut_GetCcuLoopInfoV2_WhenOutIsNull_ExpectErrorReturned)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_960)).will(returnValue(HCCL_SUCCESS));

    const CcuVersionOps* ops = nullptr;
    EXPECT_EQ(GetCcuOps(ops), HCCL_SUCCESS);
    ASSERT_NE(ops, nullptr);
    ASSERT_NE(ops->getLoopInfo, nullptr);

    CcuLoopContextV2 ctx{};
    EXPECT_NE(ops->getLoopInfo(&ctx, nullptr), HCCL_SUCCESS);
}

} // namespace hcomm
