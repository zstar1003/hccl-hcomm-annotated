/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <vector>
#include <memory>
#include <thread>
#include <chrono>
#include <cstring>

#include <hccl/hccl.h>
#include <hccl/hccl_types.h>
#include <mpi.h>

#define ACLCHECK(ret)                                                                          \
    do {                                                                                       \
        if (ret != ACL_SUCCESS) {                                                              \
            printf("acl interface return err %s:%d, retcode: %d \n", __FILE__, __LINE__, ret); \
            return ret;                                                                        \
        }                                                                                      \
    } while (0)

#define HCCLCHECK(ret)                                                                          \
    do {                                                                                        \
        if (ret != HCCL_SUCCESS) {                                                              \
            printf("hccl interface return err %s:%d, retcode: %d \n", __FILE__, __LINE__, ret); \
            return ret;                                                                         \
        }                                                                                       \
    } while (0)

struct ThreadContext {
    HcclComm comm;
    uint32_t device;
    uint32_t devCount;
};

int Sample(void* arg)
{
    ThreadContext* ctx = (ThreadContext*)arg;
    void* hostBuf = nullptr;
    void* sendBuf = nullptr;
    void* recvBuf = nullptr;
    uint32_t device = ctx->device;
    uint64_t count = ctx->devCount;
    size_t mallocSize = count * sizeof(float);

    // 申请集合通信操作的 Device 内存
    ACLCHECK(aclrtMalloc(&sendBuf, mallocSize, ACL_MEM_MALLOC_HUGE_ONLY));
    ACLCHECK(aclrtMalloc(&recvBuf, mallocSize, ACL_MEM_MALLOC_HUGE_ONLY));

    // 申请 Host 内存用于存放输入数据，并将内容初始化为：0~7
    ACLCHECK(aclrtMallocHost(&hostBuf, mallocSize));
    float* tmpHostBuff = static_cast<float*>(hostBuf);
    for (uint64_t i = 0; i < count; ++i) {
        tmpHostBuff[i] = static_cast<float>(i);
    }
    // 将 Host 侧输入数据拷贝到 Device 侧
    ACLCHECK(aclrtMemcpy(sendBuf, mallocSize, hostBuf, mallocSize, ACL_MEMCPY_HOST_TO_DEVICE));

    // 创建任务流
    aclrtStream stream;
    ACLCHECK(aclrtCreateStream(&stream));

    // 执行 AllReduce，将通信域内所有节点的 sendBuf 进行相加后，再把结果发送到所有节点的 recvBuf
    HCCLCHECK(HcclAllReduce(sendBuf, recvBuf, count, HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM, ctx->comm, stream));
    // 阻塞等待任务流中的集合通信任务执行完成
    ACLCHECK(aclrtSynchronizeStream(stream));

    // 将 Device 侧集合通信任务结果拷贝到 Host，并打印结果
    std::this_thread::sleep_for(std::chrono::seconds(device));
    void* resultBuff;
    ACLCHECK(aclrtMallocHost(&resultBuff, mallocSize));
    ACLCHECK(aclrtMemcpy(resultBuff, mallocSize, recvBuf, mallocSize, ACL_MEMCPY_DEVICE_TO_HOST));
    float* tmpResBuff = static_cast<float*>(resultBuff);
    std::cout << "rankId: " << device << ", output: [";
    for (uint64_t i = 0; i < count; ++i) {
        std::cout << " " << tmpResBuff[i];
    }
    std::cout << " ]" << std::endl;
    ACLCHECK(aclrtFreeHost(resultBuff));

    // 释放资源
    ACLCHECK(aclrtFree(sendBuf));         // 释放 Device 侧内存
    ACLCHECK(aclrtFree(recvBuf));         // 释放 Device 侧内存
    ACLCHECK(aclrtFreeHost(hostBuf));     // 释放 Host 侧内存
    ACLCHECK(aclrtDestroyStream(stream)); // 销毁任务流
    return 0;
}

int main(int argc, char* argv[])
{
    // 初始化 MPI 资源
    MPI_Init(&argc, &argv);
    int procSize = 0;
    int procRank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &procSize); // 进程总数
    MPI_Comm_rank(MPI_COMM_WORLD, &procRank); // 当前进程在所属进程组的编号
    uint32_t devId = static_cast<uint32_t>(procRank);
    uint32_t devCount = static_cast<uint32_t>(procSize);
    // 设备资源初始化
    ACLCHECK(aclInit(NULL));
    // 指定集合通信操作使用的设备
    ACLCHECK(aclrtSetDevice(static_cast<int32_t>(devId)));

    // 将 rank0 作为 root 节点
    // 生成 root 节点的 rank 标识信息
    HcclRootInfo rootInfo;
    uint32_t rootRank = 0;
    if (devId == rootRank) {
        HCCLCHECK(HcclGetRootInfo(&rootInfo));
    }
    // 将 root_info 广播到通信域内的其他 rank
    MPI_Bcast(&rootInfo, HCCL_ROOT_INFO_BYTES, MPI_CHAR, rootRank, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    // 创建并初始化通信域配置项
    HcclCommConfig config;
    HcclCommConfigInit(&config);
    // 按需修改通信域配置
    config.hcclBufferSize = 1024; // 共享数据的缓存区大小，单位为：MB，取值需 >= 1，默认值为：200
    config.hcclDeterministic = 1; // 开启归约类通信算子的确定性计算，默认值为：0，表示关闭确定性计算功能
    std::strcpy(config.hcclCommName, "comm_1");
    // 初始化集合通信域
    HcclComm hcclComm;
    HCCLCHECK(HcclCommInitRootInfoConfig(devCount, &rootInfo, devId, &config, &hcclComm));

    // 执行集合通信示例
    struct ThreadContext args;
    args.comm = hcclComm;
    args.device = devId;
    args.devCount = devCount;
    Sample((void*)&args);

    // 释放资源
    HCCLCHECK(HcclCommDestroy(hcclComm));                    // 销毁通信域
    ACLCHECK(aclrtResetDevice(static_cast<int32_t>(devId))); // 重置设备，释放设备资源
    ACLCHECK(aclFinalize());                                 // 设备去初始化
    MPI_Finalize();                                          // 释放 MPI 资源
    return 0;
}
