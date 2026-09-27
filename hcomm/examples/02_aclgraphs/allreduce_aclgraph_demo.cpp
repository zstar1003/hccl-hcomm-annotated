/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * 对应文档 docs/zh/aclgraph/aclgraph_introduction.md §2.1 的 4 步生命周期：
 *   1. aclmdlRICaptureBegin  —— 在主流上开始捕获
 *   2. HcclAllReduce         —— 通信任务被记录，不立即执行
 *   3. aclmdlRICaptureEnd    —— 结束捕获，产出 aclmdlRI
 *   4. aclmdlRIExecuteAsync  —— 整体重放（可多次）
 *
 * 运行前提：CANN >= 8.5（通信 kernel 下发 stream 能进入 model_ri）；多卡环境通过 MPI 拉起。
 * 编译示例（链接 hccl / acl / mpi）：
 *   mpic++ -std=c++14 allreduce_aclgraph_demo.cpp -lhccl -lacl -lmpi -o allreduce_aclgraph_demo
 * 运行示例：
 *   mpirun -n 2 ./allreduce_aclgraph_demo
 *
 * 参考：ACL Graph 基础 API 用法见 cann/runtime 仓示例
 * https://gitcode.com/cann/runtime/blob/master/example/2_advanced_features/model_ri/0_simple_model/main.cpp
 */

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include <hccl/hccl.h>
#include <hccl/hccl_types.h>
#include <mpi.h>

// 重放次数：捕获一次、重放多次，把 Host 调度开销摊销到一次 capture。
static constexpr int32_t REPLAY_TIMES = 4;

// ===== 错误检查辅助 =====
// 原 ACLCHECK/HCCLCHECK 宏在 main 中用 return 透传错误码；拆分到子函数后改用如下函数，
// 由调用方按返回值决定是否提前退出，错误码通过 out 参数回传给 main。
static bool CheckAcl(int32_t ret, int32_t* errCode, const char* file, int line)
{
    if (ret != ACL_SUCCESS) {
        printf("[ACL ERROR] %s:%d retcode=%d\n", file, line, ret);
        *errCode = ret;
        return false;
    }
    return true;
}

static bool CheckHccl(int32_t ret, int32_t* errCode, const char* file, int line)
{
    if (ret != HCCL_SUCCESS) {
        printf("[HCCL ERROR] %s:%d retcode=%d\n", file, line, ret);
        *errCode = ret;
        return false;
    }
    return true;
}

#define ACLCHECK(ret, errCode) CheckAcl((ret), (errCode), __FILE__, __LINE__)
#define HCCLCHECK(ret, errCode) CheckHccl((ret), (errCode), __FILE__, __LINE__)

// 阶段间需要传递的资源句柄，集中放一个结构体，避免函数参数列表过长。
struct DemoCtx {
    uint32_t devId;
    uint32_t devCount;
    uint64_t count;     // 每个 rank 输入的 float 个数，本 demo 取 devCount
    size_t mallocSize;  // count * sizeof(float)
    void* sendBuf;      // 设备显存
    void* recvBuf;      // 设备显存
    void* hostBuf;      // 锁页内存，用于 H2D 初始化
    aclrtStream stream; // 捕获 / 重放所用非默认 stream
    HcclComm hcclComm;  // 通信域
    aclmdlRI modelRI;   // 捕获产出的可执行图对象
};

// ===== 0. MPI + ACL + HCCL 初始化（与 examples/01_communicators 同一套路）=====
static bool InitEnv(int argc, char* argv[], DemoCtx* ctx, int32_t* errCode)
{
    MPI_Init(&argc, &argv);
    int procSize = 0;
    int procRank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &procSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &procRank);
    ctx->devId = static_cast<uint32_t>(procRank);
    ctx->devCount = static_cast<uint32_t>(procSize);

    if (!ACLCHECK(aclInit(nullptr), errCode)) {
        return false;
    }
    if (!ACLCHECK(aclrtSetDevice(static_cast<int32_t>(ctx->devId)), errCode)) {
        return false;
    }

    // rank0 生成 rootInfo 并广播给通信域内其余 rank。
    HcclRootInfo rootInfo{};
    const uint32_t rootRank = 0;
    if (ctx->devId == rootRank) {
        if (!HCCLCHECK(HcclGetRootInfo(&rootInfo), errCode)) {
            return false;
        }
    }
    MPI_Bcast(&rootInfo, HCCL_ROOT_INFO_BYTES, MPI_CHAR, rootRank, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    HcclCommConfig config;
    HcclCommConfigInit(&config);
    config.hcclDeterministic = 1; // 开启确定性归约，便于多卡结果比对
    ctx->hcclComm = nullptr;
    if (!HCCLCHECK(
            HcclCommInitRootInfoConfig(ctx->devCount, &rootInfo, ctx->devId, &config, &ctx->hcclComm), errCode)) {
        return false;
    }
    return true;
}

// ===== 1. 捕获前准备：内存分配 + 数据初始化 + H2D 拷贝 =====
//    关键约束（§2.3 约束 3）：capture 期间 aclrtMemcpy / aclrtMemset 在 GLOBAL 模式下非法，
//    因此所有同步内存操作必须在 CaptureBegin 之前完成。
static bool PrepareBuffers(DemoCtx* ctx, int32_t* errCode)
{
    ctx->count = ctx->devCount; // 每个 rank 输入 1 个 float，count = 卡数
    ctx->mallocSize = ctx->count * sizeof(float);

    ctx->sendBuf = nullptr;
    ctx->recvBuf = nullptr;
    if (!ACLCHECK(aclrtMalloc(&ctx->sendBuf, ctx->mallocSize, ACL_MEM_MALLOC_HUGE_ONLY), errCode)) {
        return false;
    }
    if (!ACLCHECK(aclrtMalloc(&ctx->recvBuf, ctx->mallocSize, ACL_MEM_MALLOC_HUGE_ONLY), errCode)) {
        return false;
    }

    ctx->hostBuf = nullptr;
    if (!ACLCHECK(aclrtMallocHost(&ctx->hostBuf, ctx->mallocSize), errCode)) {
        return false;
    }
    auto* tmpHost = static_cast<float*>(ctx->hostBuf);
    for (uint64_t i = 0; i < ctx->count; ++i) {
        tmpHost[i] = static_cast<float>(ctx->devId); // rank k 的输入全为 k
    }
    if (!ACLCHECK(
            aclrtMemcpy(ctx->sendBuf, ctx->mallocSize, ctx->hostBuf, ctx->mallocSize, ACL_MEMCPY_HOST_TO_DEVICE),
            errCode)) {
        return false;
    }

    // 主流：capture 必须在同一个非默认 stream 上 Begin/End（§2.3 约束 1）。
    if (!ACLCHECK(aclrtCreateStream(&ctx->stream), errCode)) {
        return false;
    }
    return true;
}

// ===== 2. 捕获：Begin → HcclAllReduce → End =====
//    捕获期间 HCCL 会自动把"通信主 stream / 通信 kernel 下发 stream"纳入 model_ri（§5.3），
//    用户只需在主流上提交 HcclAllReduce，无需关心内部多 stream 拓扑。
static bool CaptureAllReduce(DemoCtx* ctx, int32_t* errCode)
{
    ctx->modelRI = nullptr;
    if (!ACLCHECK(aclmdlRICaptureBegin(ctx->stream, ACL_MODEL_RI_CAPTURE_MODE_GLOBAL), errCode)) {
        return false;
    }
    if (!HCCLCHECK(
            HcclAllReduce(
                ctx->sendBuf, ctx->recvBuf, ctx->count, HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM, ctx->hcclComm,
                ctx->stream),
            errCode)) {
        return false;
    }
    if (!ACLCHECK(aclmdlRICaptureEnd(ctx->stream, &ctx->modelRI), errCode)) {
        return false;
    }
    // CaptureEnd 内部完成 aclmdlRIBuildModel，modelRI 已是可执行图对象。
    return true;
}

// ===== 3. 重放：一次 capture，多次 1-syscall replay =====
//    注意：被捕获的 sendBuf/recvBuf 地址已固化进图，每次 replay 操作同一块显存。
//    如需逐次更换输入，走 §3 的任务更新（TaskUpdate）或重新捕获，本 demo 不展开。
static bool ReplayGraph(DemoCtx* ctx, int32_t* errCode)
{
    for (int32_t i = 0; i < REPLAY_TIMES; ++i) {
        if (!ACLCHECK(aclmdlRIExecuteAsync(ctx->modelRI, ctx->stream), errCode)) {
            return false;
        }
        if (!ACLCHECK(aclrtSynchronizeStream(ctx->stream), errCode)) {
            return false;
        }
    }
    return true;
}

// ===== 4. 取回结果并打印 =====
//    期望：每张卡 recvBuf = sum(0..devCount-1) = devCount*(devCount-1)/2。
static bool PrintResult(const DemoCtx* ctx, int32_t* errCode)
{
    void* resultBuf = nullptr;
    if (!ACLCHECK(aclrtMallocHost(&resultBuf, ctx->mallocSize), errCode)) {
        return false;
    }
    if (!ACLCHECK(
            aclrtMemcpy(resultBuf, ctx->mallocSize, ctx->recvBuf, ctx->mallocSize, ACL_MEMCPY_DEVICE_TO_HOST),
            errCode)) {
        aclrtFreeHost(resultBuf);
        return false;
    }
    auto* tmpRes = static_cast<float*>(resultBuf);
    printf("[rank %u] AllReduce(sum) replayed %d times, recvBuf = [", ctx->devId, REPLAY_TIMES);
    for (uint64_t i = 0; i < ctx->count; ++i) {
        printf(" %.0f", tmpRes[i]);
    }
    printf(" ]\n");
    aclrtFreeHost(resultBuf);
    return true;
}

// ===== 5. 释放（顺序约束见 §2.4）=====
//    先销毁图对象 → 再销毁通信域 → 最后释放设备资源。
//    颠倒会因通信域 work 元数据指向已释放 buffer 导致 use-after-free。
static void Cleanup(DemoCtx* ctx)
{
    // modelRI 可能为 nullptr（捕获前失败），aclmdlRIDestroy 对 nullptr 容错，这里仍判空避免误用。
    if (ctx->modelRI != nullptr) {
        aclmdlRIDestroy(ctx->modelRI);
        ctx->modelRI = nullptr;
    }
    if (ctx->stream != nullptr) {
        aclrtDestroyStream(ctx->stream);
        ctx->stream = nullptr;
    }
    if (ctx->sendBuf != nullptr) {
        aclrtFree(ctx->sendBuf);
        ctx->sendBuf = nullptr;
    }
    if (ctx->recvBuf != nullptr) {
        aclrtFree(ctx->recvBuf);
        ctx->recvBuf = nullptr;
    }
    if (ctx->hostBuf != nullptr) {
        aclrtFreeHost(ctx->hostBuf);
        ctx->hostBuf = nullptr;
    }
    if (ctx->hcclComm != nullptr) {
        HcclCommDestroy(ctx->hcclComm);
        ctx->hcclComm = nullptr;
    }
    // 设备/进程级资源：尽力释放，忽略返回码。
    aclrtResetDevice(static_cast<int32_t>(ctx->devId));
    aclFinalize();
    MPI_Finalize();
}

int main(int argc, char* argv[])
{
    DemoCtx ctx{};
    int32_t errCode = 0;

    // 0. 初始化 → 1. 准备显存 → 2. 捕获 → 3. 重放 → 4. 打印；任一步失败即跳到清理。
    if (!InitEnv(argc, argv, &ctx, &errCode)) {
        // InitEnv 失败时 MPI/ACL 可能已部分初始化，仍调用 Cleanup 尝试收尾。
        Cleanup(&ctx);
        return errCode;
    }
    if (!PrepareBuffers(&ctx, &errCode)) {
        Cleanup(&ctx);
        return errCode;
    }
    if (!CaptureAllReduce(&ctx, &errCode)) {
        Cleanup(&ctx);
        return errCode;
    }
    if (!ReplayGraph(&ctx, &errCode)) {
        Cleanup(&ctx);
        return errCode;
    }
    if (!PrintResult(&ctx, &errCode)) {
        Cleanup(&ctx);
        return errCode;
    }

    Cleanup(&ctx);
    return 0;
}
