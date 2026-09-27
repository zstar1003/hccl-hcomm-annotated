/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_primitives.hpp"
#include "ccu_types.h"

namespace ccu = ::AscendC::ccu;

struct CcuLoopAddKernelArg {
    uint32_t numA{0};
    uint32_t numB{0};
};

CcuResult CcuLoopAddDemoKernel(CcuKernelArg arg)
{
    using namespace ccu;
    auto* args = static_cast<CcuLoopAddKernelArg*>(arg);

    Variable r1{}, r2{}, r3{}, r4{}, r5{}, r6{}, r7{}, numA{}, numB{};

    numA = args->numA;
    numB = args->numB;

    r1 = numA + numB;

    // ========== LoopGroup 1 (config-based): two config loops, no unroll ==========
    Func body1([&]() {
        r2 = numA + numB;
    });
    Func body2([&]() {
        r3 = numA + numB;
    });

    LoopConfig cfg1 = {.addrOffset = 0, .iterNum = 2};
    LoopConfig cfg2 = {.addrOffset = 0, .iterNum = 2};
    Loop loop1(cfg1, body1);
    Loop loop2(cfg2, body2);

    LoopGroupConfig grpCfg1
        = {.cloneNum = 0, .cloneLoopOffset = 0, .addrOffset = 0, .ccuBufferOffset = 0, .eventOffset = 0};
    LoopGroup group1(grpCfg1, /*maxLoopNum=*/2, {loop1, loop2});

    r4 = numA + numB;

    // ========== LoopGroup 2: reuse loop2 + offset loop3 ==========
    Func body3([&]() {
        r5 = numA + numB;
    });
    LoopConfig cfg3 = {.addrOffset = 4096, .iterNum = 4};
    Loop loop3(cfg3, body3);

    LoopGroupConfig grpCfg2
        = {.cloneNum = 3, .cloneLoopOffset = 1, .addrOffset = 4096, .ccuBufferOffset = 1, .eventOffset = 1};
    LoopGroup group2(grpCfg2, /*maxLoopNum=*/2, {loop2, loop3});

    // ========== LoopGroup 3 (var-based): variable group with two distinct var-loops ==========
    Variable varLoopParam4{}, varLoopParam5{}, varParallel{}, varOffset{};

    varLoopParam4 = 0x0001000200030000ULL;
    varLoopParam5 = 0x0002000300040000ULL;
    varParallel = 0x0002000100020000ULL;
    varOffset = 0x1000000100010000ULL;

    Func body4([&]() {
        r6 = numA + numB;
    });
    Func body5([&]() {
        r7 = numA + numB;
    });
    Loop loop4(varLoopParam4, body4);
    Loop loop5(varLoopParam5, body5);

    LoopGroup group3(varParallel, varOffset, /*maxLoopNum=*/2, {loop4, loop5});

    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuV2LoopGroupDemoKernel(CcuKernelArg arg)
{
    using namespace ccu;
    (void)arg;

    Variable numA{}, numB{}, r1{};
    numA = 3;
    numB = 4;

    Variable iter1{}, gsa1{}, iter2{}, gsa2{};
    Variable parallel{}, offset{}, xnOffset{};

    iter1 = 2;
    gsa1 = 0;
    iter2 = 4;
    gsa2 = 4096;
    parallel = 0x0002000100020000ULL;
    offset = 0x1000000100010000ULL;
    xnOffset = 0x0000000300040000ULL;

    Func body1([&]() {
        r1 = numA + numB;
    });
    Func body2([&]() {
        r1 = numA + numB;
    });

    Loop loop1(iter1, gsa1, body1);
    Loop loop2(iter2, gsa2, body2);

    LoopGroup group(parallel, offset, xnOffset, /*maxLoopNum=*/2, {loop1, loop2});

    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuV2CompatLoopGroupDemoKernel(CcuKernelArg arg)
{
    using namespace ccu;
    (void)arg;

    Variable numA{}, numB{}, r1{};
    numA = 3;
    numB = 4;

    Variable varLoopParam1{}, varLoopParam2{}, varParallel{}, varOffset{};
    varLoopParam1 = 0x0001000200030000ULL;
    varLoopParam2 = 0x0002000300040000ULL;
    varParallel = 0x0002000100020000ULL;
    varOffset = 0x1000000100010000ULL;

    Func body1([&]() {
        r1 = numA + numB;
    });
    Func body2([&]() {
        r1 = numA + numB;
    });

    Loop loop1(varLoopParam1, body1);
    Loop loop2(varLoopParam2, body2);

    LoopGroup group(varParallel, varOffset, /*maxLoopNum=*/2, {loop1, loop2});

    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuV2ConfigLoopGroupDemoKernel(CcuKernelArg arg)
{
    using namespace ccu;
    (void)arg;

    Variable numA{}, numB{}, r1{}, r2{};
    numA = 3;
    numB = 4;

    Func body1([&]() {
        r1 = numA + numB;
    });
    Func body2([&]() {
        r2 = numA + numB;
    });

    LoopConfig cfg1 = {.addrOffset = 0, .iterNum = 2};
    LoopConfig cfg2 = {.addrOffset = 4096, .iterNum = 4};
    Loop loop1(cfg1, body1);
    Loop loop2(cfg2, body2);

    LoopGroupConfig grpCfg
        = {.cloneNum = 3, .cloneLoopOffset = 1, .addrOffset = 4096, .ccuBufferOffset = 1, .eventOffset = 1};
    LoopGroup group(grpCfg, /*maxLoopNum=*/2, {loop1, loop2});

    return CcuResult::CCU_SUCCESS;
}

// config group 内混入三种不同构造方式的 loop，验证每 loop 按自身方式翻译
CcuResult CcuV2MixedLoopGroupDemoKernel(CcuKernelArg arg)
{
    using namespace ccu;
    (void)arg;

    Variable numA{}, numB{}, r1{}, r2{}, r3{};
    numA = 3;
    numB = 4;

    Variable varLoopParam2{}, iter3{}, gsa3{};
    varLoopParam2 = 0x0002000300040000ULL;
    iter3 = 4;
    gsa3 = 4096;

    Func body1([&]() {
        r1 = numA + numB;
    });
    Func body2([&]() {
        r2 = numA + numB;
    });
    Func body3([&]() {
        r3 = numA + numB;
    });

    LoopConfig cfg1 = {.addrOffset = 0, .iterNum = 2};
    Loop loop1(cfg1, body1);          // config loop
    Loop loop2(varLoopParam2, body2); // packed-var loop
    Loop loop3(iter3, gsa3, body3);   // v2 var-direct loop

    LoopGroupConfig grpCfg
        = {.cloneNum = 3, .cloneLoopOffset = 1, .addrOffset = 4096, .ccuBufferOffset = 1, .eventOffset = 1};
    LoopGroup group(grpCfg, /*maxLoopNum=*/3, {loop1, loop2, loop3});

    return CcuResult::CCU_SUCCESS;
}

// 版本化 cfg 全路径 demo：Loop(CcuLoopCfg) + LoopGroup(CcuLoopGroupCfg)（含 varOffset）
CcuResult CcuV2LoopGroupCfgDemoKernel(CcuKernelArg arg)
{
    using namespace ccu;
    (void)arg;

    Variable numA{}, numB{}, r1{}, r2{};
    numA = 3;
    numB = 4;

    Func body1([&]() {
        r1 = numA + numB;
    });
    Func body2([&]() {
        r2 = numA + numB;
    });

    LoopCfg cfg1 = {.addrOffset = 0, .iterNum = 2};
    LoopCfg cfg2 = {.addrOffset = 4096, .iterNum = 4};
    Loop loop1(cfg1, body1);
    Loop loop2(cfg2, body2);

    LoopGroupCfg grpCfg
        = {.cloneNum = 3,
           .cloneLoopOffset = 1,
           .addrOffset = 4096,
           .ccuBufferOffset = 1,
           .eventOffset = 1,
           .varOffset = 7};
    LoopGroup group(grpCfg, /*maxLoopNum=*/2, {loop1, loop2});

    return CcuResult::CCU_SUCCESS;
}

// A5(CCU_V1) cfg/variable 混用 loop 计数 demo：
//  - config group + [config loop, packed-var loop] -> loops=2, totalLoopNum=2
//  - var group    + [config loop, packed-var loop] -> loops=2, totalLoopNum=0
CcuResult CcuA5MixedLoopCountDemoKernel(CcuKernelArg arg)
{
    using namespace ccu;
    (void)arg;

    Variable numA{}, numB{}, r1{}, r2{}, r3{}, r4{};
    numA = 3;
    numB = 4;

    Variable varLoopParam1{}, varLoopParam2{}, varParallel{}, varOffset{};
    varLoopParam1 = 0x0001000200030000ULL;
    varLoopParam2 = 0x0002000300040000ULL;
    varParallel = 0x0002000100020000ULL;
    varOffset = 0x1000000100010000ULL;

    Func body1([&]() {
        r1 = numA + numB;
    });
    Func body2([&]() {
        r2 = numA + numB;
    });
    Func body3([&]() {
        r3 = numA + numB;
    });
    Func body4([&]() {
        r4 = numA + numB;
    });

    // config group 混入 1 个 config loop + 1 个 packed-var loop
    LoopConfig cfg1 = {.addrOffset = 0, .iterNum = 2};
    Loop cfgLoop1(cfg1, body1);
    Loop varLoop1(varLoopParam1, body2);
    LoopGroupConfig grpCfg
        = {.cloneNum = 3, .cloneLoopOffset = 1, .addrOffset = 4096, .ccuBufferOffset = 1, .eventOffset = 1};
    LoopGroup cfgGroup(grpCfg, /*maxLoopNum=*/2, {cfgLoop1, varLoop1});

    // var group 混入 1 个 config loop + 1 个 packed-var loop（totalLoopNum 应保持 0）
    LoopConfig cfg2 = {.addrOffset = 0, .iterNum = 2};
    Loop cfgLoop2(cfg2, body3);
    Loop varLoop2(varLoopParam2, body4);
    LoopGroup varGroup(varParallel, varOffset, /*maxLoopNum=*/2, {cfgLoop2, varLoop2});

    return CcuResult::CCU_SUCCESS;
}

// A6(CCU_V2) cfg/variable 混用 loop 计数 demo：
//  - config group     + [config, packed-var, v2-var] -> loops=3, totalLoopNum=3
//  - var group(V2)    + [v2-var, config]             -> loops=2, totalLoopNum=0
CcuResult CcuA6MixedLoopCountDemoKernel(CcuKernelArg arg)
{
    using namespace ccu;
    (void)arg;

    Variable numA{}, numB{}, r1{}, r2{}, r3{}, r4{}, r5{};
    numA = 3;
    numB = 4;

    Variable packedVar{}, iterA{}, gsaA{}, iterB{}, gsaB{};
    packedVar = 0x0002000300040000ULL;
    iterA = 2;
    gsaA = 0;
    iterB = 4;
    gsaB = 4096;

    Variable parallelV2{}, offsetV2{}, xnOffsetV2{};
    parallelV2 = 0x0002000100020000ULL;
    offsetV2 = 0x1000000100010000ULL;
    xnOffsetV2 = 0x0000000300040000ULL;

    Func body1([&]() {
        r1 = numA + numB;
    });
    Func body2([&]() {
        r2 = numA + numB;
    });
    Func body3([&]() {
        r3 = numA + numB;
    });
    Func body4([&]() {
        r4 = numA + numB;
    });
    Func body5([&]() {
        r5 = numA + numB;
    });

    // config group 混入 config + packed-var + v2-var 三种 loop
    LoopConfig cfg1 = {.addrOffset = 0, .iterNum = 2};
    Loop cfgLoop1(cfg1, body1);
    Loop packedLoop(packedVar, body2);
    Loop v2Loop1(iterA, gsaA, body3);
    LoopGroupConfig grpCfg
        = {.cloneNum = 3, .cloneLoopOffset = 1, .addrOffset = 4096, .ccuBufferOffset = 1, .eventOffset = 1};
    LoopGroup cfgGroup(grpCfg, /*maxLoopNum=*/3, {cfgLoop1, packedLoop, v2Loop1});

    // V2 var group 混入 v2-var loop + config loop（totalLoopNum 应保持 0）
    LoopConfig cfg2 = {.addrOffset = 0, .iterNum = 2};
    Loop cfgLoop2(cfg2, body4);
    Loop v2Loop2(iterB, gsaB, body5);
    LoopGroup varGroupV2(parallelV2, offsetV2, xnOffsetV2, /*maxLoopNum=*/2, {v2Loop2, cfgLoop2});

    return CcuResult::CCU_SUCCESS;
}

// strict 校验 demo：裸 C ABI 传入未初始化版本头(magic 非法),期望被拒绝、注册失败
inline CcuResult CcuLoopGroupCreateCfgBadHeaderDemoKernel(CcuKernelArg arg)
{
    (void)arg;
    CcuLoopGroup group = 0;
    CcuLoopGroupCfg badCfg{};
    return ::CcuLoopGroupCreateCfg(&group, 2, &badCfg);
}

inline CcuResult CcuLoopGroupAddLoopCfgBadHeaderDemoKernel(CcuKernelArg arg)
{
    (void)arg;
    CcuLoopCfg badCfg{};
    return ::CcuLoopGroupAddLoopCfg(1, 1, &badCfg);
}

struct CcuLoopGroupMaxLoopNumKernelArg {
    uint32_t maxLoopNum{1};
};

inline CcuResult CcuLoopGroupMaxLoopNumDemoKernel(CcuKernelArg arg)
{
    using namespace ccu;
    auto* args = static_cast<CcuLoopGroupMaxLoopNumKernelArg*>(arg);
    Func body([&]() {});
    LoopConfig loopCfg = {.addrOffset = 0, .iterNum = 1};
    Loop loop(loopCfg, body);
    LoopGroupConfig grpCfg
        = {.cloneNum = 0, .cloneLoopOffset = 0, .addrOffset = 0, .ccuBufferOffset = 0, .eventOffset = 0};
    LoopGroup group(grpCfg, args->maxLoopNum, {loop});
    return CcuResult::CCU_SUCCESS;
}

inline CcuResult CcuIfInLoopInvalidDemoKernel(CcuKernelArg arg)
{
    using namespace ccu;
    (void)arg;
    Variable v{};
    v = 0;
    Func body([&]() {
        CCU_IF(v == 0)
        {
            Variable t{};
            t = v + v;
        }
    });
    LoopConfig dummyCfg{};
    Loop loop(dummyCfg, body);
    return CcuResult::CCU_SUCCESS;
}

inline CcuResult CcuNotifyRecordInLoopInvalidDemoKernel(CcuKernelArg arg)
{
    using namespace ccu;
    (void)arg;
    ChannelHandle ch = 0;
    Func body([&]() {
        (void)NotifyRecord(ch, 0);
    });
    LoopConfig dummyCfg{};
    Loop loop(dummyCfg, body);
    return CcuResult::CCU_SUCCESS;
}

inline CcuResult CcuWriteVarWithNotifyInLoopInvalidDemoKernel(CcuKernelArg arg)
{
    using namespace ccu;
    (void)arg;
    ChannelHandle ch = 0;
    Variable v{};
    v = 1;
    Func body([&]() {
        (void)WriteVariableWithNotify(ch, v, 0, 0);
    });
    LoopConfig dummyCfg{};
    Loop loop(dummyCfg, body);
    return CcuResult::CCU_SUCCESS;
}

inline CcuResult CcuEventRecordTagInLoopInvalidDemoKernel(CcuKernelArg arg)
{
    using namespace ccu;
    (void)arg;
    Func body([&]() {
        (void)EventRecord("evt_tag", 1);
    });
    LoopConfig dummyCfg{};
    Loop loop(dummyCfg, body);
    return CcuResult::CCU_SUCCESS;
}

inline CcuResult CcuEventRecordInLoopInvalidDemoKernel(CcuKernelArg arg)
{
    using namespace ccu;
    (void)arg;
    Event evt{};
    Func body([&]() {
        (void)EventRecord(evt, 1);
    });
    LoopConfig dummyCfg{};
    Loop loop(dummyCfg, body);
    return CcuResult::CCU_SUCCESS;
}

// 全组合覆盖：group 类型（config / var） × loop 类型（config / var / 混用）。
// 校验点（按 group 创建顺序在日志中出现 LoopGroupBundle[loops=N, totalLoopNum=M]）：
//   - config 型 group：无论内部 loop 是 config / var / 混用，totalLoopNum 必须等于 loop 总数；
//   - var 型 group：parallelParam 走运行期寄存器，bundle 的 totalLoopNum 不参与，保持 0。
CcuResult CcuLoopCfgDemoKernel(CcuKernelArg arg)
{
    using namespace ccu;
    (void)arg;

    Variable a{}, b{};
    a = 3;
    b = 4;

    // 每个 Loop 各自 compose 一份独立 body
    Variable s1{}, s2{}, s3{}, s4{}, s5{}, s6{};
    Func body1([&]() {
        s1 = a + b;
    });
    Func body2([&]() {
        s2 = a + b;
    });
    Func body3([&]() {
        s3 = a + b;
    });
    Func body4([&]() {
        s4 = a + b;
    });
    Func body5([&]() {
        s5 = a + b;
    });
    Func body6([&]() {
        s6 = a + b;
    });

    // config 型 loop
    LoopConfig cfg = {.addrOffset = 0, .iterNum = 2};
    Loop lc1(cfg, body1);
    Loop lc2(cfg, body2);
    Loop lc3(cfg, body3);

    // var 型 loop
    Variable lp1{}, lp2{}, lp3{};
    lp1 = 2;
    lp2 = 2;
    lp3 = 2;
    Loop lv1(lp1, body4);
    Loop lv2(lp2, body5);
    Loop lv3(lp3, body6);

    LoopGroupConfig cfgGrp
        = {.cloneNum = 0, .cloneLoopOffset = 0, .addrOffset = 0, .ccuBufferOffset = 0, .eventOffset = 0};

    // ===== config 型 group（totalLoopNum 以立即数编码，必须等于 loop 总数）=====
    // 1) config group + config loops  -> loops=2, totalLoopNum=2
    LoopGroup g1(cfgGrp, /*maxLoopNum=*/2, {lc1, lc2});
    // 2) config group + var loops      -> loops=3, totalLoopNum=3（
    LoopGroup g2(cfgGrp, /*maxLoopNum=*/3, {lv1, lv2, lv3});
    // 3) config group + mixed loops    -> loops=2, totalLoopNum=2
    LoopGroup g3(cfgGrp, /*maxLoopNum=*/2, {lc1, lv1});

    // ===== var 型 group（parallel/offset 走运行期寄存器，bundle totalLoopNum 保持 0）=====
    Variable par{}, off{};
    par = 0x0002000100020000ULL;
    off = 0x1000000100010000ULL;
    // 4) var group + var loops         -> loops=2, totalLoopNum=0
    LoopGroup g4(par, off, /*maxLoopNum=*/2, {lv1, lv2});
    // 5) var group + config loops      -> loops=3, totalLoopNum=0
    LoopGroup g5(par, off, /*maxLoopNum=*/3, {lc1, lc2, lc3});
    // 6) var group + mixed loops       -> loops=2, totalLoopNum=0
    LoopGroup g6(par, off, /*maxLoopNum=*/2, {lc1, lv1});

    return CcuResult::CCU_SUCCESS;
}
