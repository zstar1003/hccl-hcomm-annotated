/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ut_HcommCcuControlApi_common.h"

#define private public
#define protected public
#include "ccu_kernel_impl/ccu_cke_nop_demo.h"
#undef protected
#undef private

#include <cstdio>
#include <cstdlib>
#include <vector>

// ============================================================================
// A6 后端优化 CKE 补 NOP 行为验证 (算子 demo: ccu_kernel_impl/ccu_cke_nop_demo.h)
// ----------------------------------------------------------------------------
// 用真实 kernel 走完整翻译 + 后端优化流程, 统计翻译后指令序列里 NOP 的条数, 验证:
//   - 单对 record+wait (同一 event): wait(clearType=1) 是写者但其后无读同一 bit 的读者,
//     不构成写后读 -> 不补 NOP (nopBetweenPair == -1)
//   - 两对近邻 record+wait: wait#1(写者) 与 wait#2(读) 同一 bit 构成写后读 -> 补 (L-2) 条 NOP
// 说明: set 与 wait 必须成对 (单独 wait 未被 set 的 event 硬件会死锁), 且 EventRecord/EventWait 都只
// 能在循环体外使用 (LOOP_BLOCK 内均非法), 故 wait 走 profiling 分支翻译成 setcke。setcke 是"既 wait
// 又 set" 的 CKE 写者, record 的 setCKEId 与随后 wait 的 waitCKEId 同 bit 构成写后读, 补 NOP。
// ============================================================================
namespace {
// V2 microcode opcode (取自 ccu_microcode_opt/config/barrier_config.h)。
constexpr uint16_t CKE_DEMO_LOAD_TYPE = 0x0;
constexpr uint16_t CKE_DEMO_NOP_CODE = 0x9;
constexpr uint16_t CKE_DEMO_CTRL_TYPE = 0x1;
constexpr uint16_t CKE_DEMO_SETCKBIT_CODE = 0x2;
constexpr uint16_t CKE_DEMO_CLEARCKBIT_CODE = 0x4;

struct CkeDemoInstrStat {
    uint32_t setcke = 0;   // 全序列 setcke 条数
    uint32_t clearcke = 0; // 全序列 clearcke 条数
    // "关注对"之间插入的 NOP 数: 第一条 def 某 CKE bit 的写者(setcke/clearcke) 与紧随其后、read
    // 同一 bit 的下一条读者(setcke/clearcke) 之间的 NOP 条数。用于精确隔离"写后读补的 NOP",
    // 排除 FinishMainBlock 等结构性 NOP 的干扰。
    int32_t nopBetweenPair = -1; // -1 表示没找到这样的一对
};

inline bool CkeDemoIsNop(const hcomm::CcuRep::CcuInstr& in)
{
    return in.header.type == CKE_DEMO_LOAD_TYPE && in.header.code == CKE_DEMO_NOP_CODE;
}
inline bool CkeDemoIsSetCke(const hcomm::CcuRep::CcuInstr& in)
{
    return in.header.type == CKE_DEMO_CTRL_TYPE && in.header.code == CKE_DEMO_SETCKBIT_CODE;
}
inline bool CkeDemoIsClearCke(const hcomm::CcuRep::CcuInstr& in)
{
    return in.header.type == CKE_DEMO_CTRL_TYPE && in.header.code == CKE_DEMO_CLEARCKBIT_CODE;
}

// 该指令 def 的 CKE bit (写者身份): 只有 clearType=1 自动清零的 waitCKEId 才是触发写后读的写者
// (与 extract_operands.cc 一致); setCKEId / clearCKEId 只是置位/主动清位, 不算写者。
inline uint16_t CkeDemoWriterBit(const hcomm::CcuRep::CcuInstr& in)
{
    if (CkeDemoIsSetCke(in) && in.v2.setCKE.clearType == 1) {
        return in.v2.setCKE.waitCKEId;
    }
    if (CkeDemoIsClearCke(in) && in.v2.clearCKE.clearType == 1) {
        return in.v2.clearCKE.waitCKEId;
    }
    return 0;
}

// 该指令 read 的 CKE bit (读者身份): setcke / clearcke 的 waitCKEId (阻塞等待读)。
inline uint16_t CkeDemoReaderBit(const hcomm::CcuRep::CcuInstr& in)
{
    if (CkeDemoIsSetCke(in) || CkeDemoIsClearCke(in)) {
        return CkeDemoIsSetCke(in) ? in.v2.setCKE.waitCKEId : in.v2.clearCKE.waitCKEId;
    }
    return 0;
}

// 定位第一对真实"写后读": 某写者 def bit B, 其后最近的读者 read 同一 bit B。跳过如 reserveCke
// 这类只被 def 却无后续读者的结构性 setcke, 返回这对之间的 NOP 数; 找不到返回 -1。
inline int32_t CkeDemoCountNopBetweenPair(const std::vector<hcomm::CcuRep::CcuInstr>& v)
{
    for (size_t w = 0; w < v.size(); ++w) {
        uint16_t wb = CkeDemoWriterBit(v[w]);
        if (wb == 0) {
            continue;
        }
        for (size_t r = w + 1; r < v.size(); ++r) {
            if (CkeDemoReaderBit(v[r]) != wb) {
                continue;
            }
            uint32_t nopCnt = 0;
            for (size_t k = w + 1; k < r; ++k) {
                if (CkeDemoIsNop(v[k])) {
                    nopCnt++;
                }
            }
            return static_cast<int32_t>(nopCnt);
        }
    }
    return -1;
}

// 统计翻译后指令里 setcke / clearcke 条数, 以及"关注对"之间插入的 NOP 数。
CkeDemoInstrStat CountCkeDemoInstrStat(const std::vector<hcomm::CcuRep::CcuInstr>& v, bool dbg)
{
    CkeDemoInstrStat stat{};
    for (size_t i = 0; i < v.size(); ++i) {
        const auto& ins = v[i];
        if (dbg) {
            std::printf(
                "[%3zu] type=%u code=%u writerBit=%u readerBit=%u\n", i, ins.header.type, ins.header.code,
                CkeDemoWriterBit(ins), CkeDemoReaderBit(ins));
        }
        if (CkeDemoIsSetCke(ins)) {
            stat.setcke++;
        } else if (CkeDemoIsClearCke(ins)) {
            stat.clearcke++;
        }
    }
    stat.nopBetweenPair = CkeDemoCountNopBetweenPair(v);
    return stat;
}

// 注册并翻译一个 V2 demo kernel(RegisterEnd 内部会跑 A6 后端优化), 统计翻译后指令里
// setcke / clearcke 条数, 以及"关注对"之间插入的 NOP 数。
CkeDemoInstrStat RunV2DemoAndCountInstr(void* demoFunc, const char* name)
{
    constexpr uint32_t fakeDevId = MAX_MODULE_DEVICE_NUM - 2;
    constexpr hcomm::CcuVersion fakeCcuVersion = hcomm::CcuVersion::CCU_V2;
    int32_t fakeDeviceLogicId = MockCcuDeviceEnv(fakeDevId, fakeCcuVersion);

    HcommCcuResDescHandle resDescs[hcomm::CCU_MAX_IODIE_NUM] = {0, 0};
    CreateCcuResDescsPair(resDescs, fakeCcuVersion);
    CcuInsHandle insHandle{0};
    EXPECT_EQ(HcommCcuInsCreate(resDescs, 2, &insHandle), CcuResult::CCU_SUCCESS);
    EXPECT_EQ(HcommCcuKernelRegisterStart(insHandle), CcuResult::CCU_SUCCESS);

    int32_t dummyArg = 0;
    CcuKernelArg kernelArg = static_cast<CcuKernelArg>(&dummyArg);
    const void* kernelArgs[] = {kernelArg};
    CcuKernelHandle kernelHandle{0};
    EXPECT_EQ(
        HcommCcuKernelRegister(insHandle, 0, const_cast<char*>(name), demoFunc, kernelArgs, 1, &kernelHandle),
        CcuResult::CCU_SUCCESS);
    // RegisterEnd 触发翻译 + 后端优化, 之后 kernel->instrInfo_ 即为优化后的指令序列。
    EXPECT_EQ(HcommCcuKernelRegisterEnd(insHandle), CcuResult::CCU_SUCCESS);

    CkeDemoInstrStat stat{};
    auto* kernel = hcomm::CcuKernelMgr::GetInstance(fakeDeviceLogicId).GetKernel(kernelHandle);
    EXPECT_NE(kernel, nullptr);
    if (kernel != nullptr) {
        const bool dbg = (getenv("CKE_DEMO_DUMP") != nullptr);
        stat = CountCkeDemoInstrStat(kernel->instrInfo_.instrVec, dbg);
    }

    EXPECT_EQ(HcommCcuInsDestroy(insHandle), CcuResult::CCU_SUCCESS);
    DestroyCcuResDescs(resDescs);
    return stat;
}
} // namespace

// 场景2: 循环体外两对近邻 record+wait (同一 event) -> 四条 setcke。写者身份只认 clearType=1 自动清零的
// wait: record(置位, 非写者) -> wait#1(clearType=1 自清, 写者) -> record#2(非写者) -> wait#2(读)。
// wait#1(写者) 与 wait#2(读) 同一 CKE bit 构成写后读, 后端优化在 wait#1 与 wait#2 之间补
// (CCU_CKE_RAW_LATENCY - 2) 条 NOP (NOP 落在 wait#2 前, 与 wait#1 之间还夹着 record#2, 故为 L-2)。
TEST_F(HcommCcuControlApiTest, Ut_CkeNop_When_SetClearPairs_SameEvent_Expect_InsertNop)
{
    CkeDemoInstrStat stat = RunV2DemoAndCountInstr(
        reinterpret_cast<void*>(CcuCkeSetClearPairsHazardDemoKernel), "CcuCkeSetClearPairsHazardDemoKernel");

    EXPECT_GE(stat.setcke, 4u) << "两对 record+wait 应翻译出四条 setcke";
    ASSERT_GE(stat.nopBetweenPair, 0) << "未定位到 def/read 同一 CKE bit 的写后读对";
    EXPECT_EQ(static_cast<uint32_t>(stat.nopBetweenPair), hcomm::CcuRep::CCU_CKE_RAW_LATENCY - 2)
        << "wait#1(clearType=1 自清=写者) -> wait#2(读) 是写后读, 两者之间(夹 record#2)应补 (L-2) 条 NOP";
}

// 场景1: 循环体外单对 record+wait (同一 event) -> 两条 setcke。写者身份只认 clearType=1 自动清零的 wait:
// record(置位, 非写者) -> wait(clearType=1 自清, 写者)。wait 是写者但其后无读同一 bit 的读者, 不构成写后读,
// 后端优化不补 NOP。
TEST_F(HcommCcuControlApiTest, Ut_CkeNop_When_SetClearPair_SameEvent_Expect_NoNop)
{
    CkeDemoInstrStat stat = RunV2DemoAndCountInstr(
        reinterpret_cast<void*>(CcuCkeSetClearPairHazardDemoKernel), "CcuCkeSetClearPairHazardDemoKernel");

    EXPECT_GE(stat.setcke, 2u) << "单对 record+wait 应翻译出两条 setcke";
    EXPECT_EQ(stat.nopBetweenPair, -1)
        << "单对 record+wait: wait 是写者但无后续读同一 bit 的读者, 不构成写后读, 不应补 NOP";
}
