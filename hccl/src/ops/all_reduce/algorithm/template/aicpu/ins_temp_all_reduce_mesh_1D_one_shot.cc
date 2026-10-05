/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_temp_all_reduce_mesh_1D_one_shot.h"
#include "alg_data_trans_wrapper.h"

namespace ops_hccl {
// [中文导读] [AllReduce逐行 S15] 定义 Mesh OneShot 模板构造器，继承通用算法参数与 Rank 信息。
InsTempAllReduceMesh1DOneShot::InsTempAllReduceMesh1DOneShot(
    // [中文导读] [AllReduce逐行 S16] 接收统一算子参数和通信域用户 Rank，保留用户 Rank 原编号。
    const OpParam& param, const u32 rankId, // 传通信域的rankId，userRank
    // [中文导读] [AllReduce逐行 S17] 接收算法第零层子通信域 Rank 列表，供 Peer 映射和槽数量计算。
    const std::vector<std::vector<u32>>& subCommRanks)
    // [中文导读] [AllReduce逐行 S18] 交基类保存模式、用户 Rank、子通信域、归约运算及绕路标志。
    : InsAlgTemplateBase(param, rankId, subCommRanks)
// [中文导读] [AllReduce逐行 S19] OneShot 构造器没有额外初始化语句。
{}

// [中文导读] [AllReduce逐行 S21] OneShot 使用空析构实现，资源生命周期由上层持有对象管理。
InsTempAllReduceMesh1DOneShot::~InsTempAllReduceMesh1DOneShot() {}

// [中文导读] [AllReduce逐行 S23] 为新成本选择器计算 OneShot 的四项成本系数。
std::vector<CostModelParam> InsTempAllReduceMesh1DOneShot::CalcCostCoeff(CalcCostCoeffParam param)
// [中文导读] [AllReduce逐行 S24] 进入 CalcCostCoeff 的实现作用域；OneShot 成本系数：排除超过 8 Rank，按端口并发度估计网络时延/传输比例及本地拷贝和归约。
{
    // [中文导读] [AllReduce逐行 S25] 成本模型仅支持通信域 Rank 数不超过 8；这是新成本模型限制，旧阈值选择器没有调用此函数。
    if (param.rankSize > 8) {
        // [中文导读] [AllReduce逐行 S26] 超过8 Rank 不提供成本候选，返回空系数列表。
        return {};
    // [中文导读] [AllReduce逐行 S27] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
    }
    // [中文导读] [AllReduce逐行 S28] Pod 的 CLOS 且至少有两个端口组时合并前两组端口数。
    int portNum = (param.isPod && param.netType == CommTopo::COMM_TOPO_CLOS && param.portNum.size() >= 2) ?
                      // [中文导读] [AllReduce逐行 S29] 两组端口数量相加，作为 Pod CLOS 的有效并发端口数。
                      (param.portNum[0] + param.portNum[1]) :
                      // [中文导读] [AllReduce逐行 S30] 其它拓扑只取第一个端口组的数量估计传输并发度。
                      param.portNum[0];
    // [中文导读] [AllReduce逐行 S31] 该 OneShot 成本估计只计一个 kernel 的固定时延。
    int kernelNum = 1;
    // [中文导读] [AllReduce逐行 S32] 先估算 Peer 传输任务数。
    int taskNum = CostModelManager::CalcTransTaskNum(param.rankSize)
                  // [中文导读] [AllReduce逐行 S33] 再加两次主从同步任务和 N−1 个本地归约任务。
                  + CostModelManager::CalcSyncTaskNum(param.rankSize) * 2 + (param.rankSize - 1);
    // [中文导读] [AllReduce逐行 S34] 初始化网络传输成本系数 A。
    float A = 0.0f;
    // [中文导读] [AllReduce逐行 S35] 初始化本地拷贝/归约成本系数 B。
    float B = 0.0f;
    // [中文导读] [AllReduce逐行 S36] 初始化 kernel 固定时延系数 C。
    float C = 0.0f;
    // [中文导读] [AllReduce逐行 S37] 初始化任务发射开销系数 D。
    float D = 0.0f;
    // 因为不知道是twoshot还是oneshot，executor层统一传twoshot需要的一片数据大小
    // [中文导读] [AllReduce逐行 S39] 将执行器传入的数据比例乘 Rank 数，换算为 OneShot 全量输入的数据比例。
    float n = param.dataRatio * param.rankSize;
    // 同时有localreduce和localcopy，所以需要调用两个接口获取两个步骤的B并相加
    // [中文导读] [AllReduce逐行 S41] 单独保存一次本地拷贝的数据量成本。
    float B1 = 0.0f;
    // [中文导读] [AllReduce逐行 S42] 单独保存一次本地归约的数据量成本。
    float B2 = 0.0f;
    // [中文导读] [AllReduce逐行 S43] 结合网络类型、端口并发、Rank 数和 Pod 属性估算全量 Mesh 交换成本 A。
    CostModelManager::Global()->CalcMeshParam(n, param.netType, portNum, param.rankSize, A, param.isPod);
    // [中文导读] [AllReduce逐行 S44] 输入与 scratch 为不同缓冲区时，成本模型需要计入本地拷贝。
    if (param.inputBuffer != param.scratchBuffer) {
        // [中文导读] [AllReduce逐行 S45] 计算 AICPU 模式一次完整输入拷贝的成本 B1。
        CostModelManager::Global()->CalcLocalCopyParams(n, EngineType::AICPU, B1);
    // [中文导读] [AllReduce逐行 S46] 前面的条件未满足时进入本分支，执行此处替代算法/协议/校验路径。
    } else {
        // [中文导读] [AllReduce逐行 S47] 输入与 scratch 相同时，本成本模型把拷贝成本置零。
        B1 = 0.0f;
    // [中文导读] [AllReduce逐行 S48] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
    }
    // [中文导读] [AllReduce逐行 S49] 计算 AICPU 模式一次完整输入归约的成本 B2。
    CostModelManager::Global()->CalcLocalReduceParams(n, EngineType::AICPU, B2);
    // [中文导读] [AllReduce逐行 S50] 本地数据量成本等于一次拷贝加 N−1 次完整输入归约。
    B = B1 + (param.rankSize - 1) * B2;
    // [中文导读] [AllReduce逐行 S51] 计算一个 AICPU kernel 的固定时延 C。
    CostModelManager::Global()->CalcLatencyParams(kernelNum, EngineType::AICPU, C);
    // [中文导读] [AllReduce逐行 S52] 根据估算的任务数量计算 AICPU 发射开销 D。
    CostModelManager::Global()->CalcLaunchParams(taskNum, EngineType::AICPU, D);

    // [中文导读] [AllReduce逐行 S54] 创建该模板可提供的成本参数列表。
    std::vector<CostModelParam> params;
    // [中文导读] [AllReduce逐行 S55] 加入一项网络、本地处理、kernel 时延和发射开销系数。
    params.push_back({A, B, C, D});
    // [中文导读] [AllReduce逐行 S56] 开始 HCCL_DEBUG 诊断输出，记录 CalcCostCoeff 当前阶段的参数、候选或错误；日志本身不决定返回码。
    HCCL_DEBUG("[%s] CalcCostCoeff A=%f B=%f C=%f D=%f.", __func__, A, B, C, D);
    // [中文导读] [AllReduce逐行 S57] 返回此 OneShot 候选的成本系数列表。
    return params;
// [中文导读] [AllReduce逐行 S58] 结束 CalcCostCoeff 实现；其返回状态或已写回字段由调用者接收。
}

// [中文导读] [AllReduce逐行 S60] 定义 CalcRes 入口：构造具体 InsAlgTemplate 并计算资源请求，函数本身不创建实际 Thread/Channel。
HcclResult InsTempAllReduceMesh1DOneShot::CalcRes(
    // [中文导读] [AllReduce逐行 S61] 传入通信域、统一算子参数和已经解析的物理拓扑。
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    // [中文导读] [AllReduce逐行 S62] 通过 resourceRequest 输出 Thread、通知槽和通道描述请求。
    AlgResourceRequest& resourceRequest)
// [中文导读] [AllReduce逐行 S63] 进入 CalcRes 的实现作用域；构造具体 InsAlgTemplate 并计算资源请求，函数本身不创建实际 Thread/Channel。
{
    // mesh 算法只做level 0 层级的
    // [中文导读] [AllReduce逐行 S65] 总 Thread 请求数至少为 1，通常等于当前算法通信域 Rank 数。
    u32 threadNum = templateRankSize_ > 1 ? templateRankSize_ : 1;
    // [中文导读] [AllReduce逐行 S66] 申请总 Thread 数减 1 个从 Thread；主 Thread 可从传入 stream 转换取得。
    resourceRequest.slaveThreadNum = threadNum - 1; // 主thread可以通过接口传入的stream来做转换
    // [中文导读] [AllReduce逐行 S67] 为每个从 Thread 请求一个通知槽，前同步统一使用各自槽0。
    resourceRequest.notifyNumPerThread.assign(resourceRequest.slaveThreadNum, 1);
    // [中文导读] [AllReduce逐行 S68] 主 Thread 请求从 Thread 数量个汇合槽，后同步逐从 Thread 独立等待。
    resourceRequest.notifyNumOnMainThread = threadNum - 1;

    // [中文导读] [AllReduce逐行 S70] 暂存第零层 Mesh 的 Peer 通道请求。
    std::vector<HcclChannelDesc> level0Channels;
    // [中文导读] [AllReduce逐行 S71] 为 Mesh 第零层其它 Rank 计算通道描述请求；失败立即返回，不在模板里直接创建运行时通道。
    CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, subCommRanks_, level0Channels));
    // [中文导读] [AllReduce逐行 S72] 把第零层请求加入按通信层组织的资源请求列表。
    resourceRequest.channels.push_back(level0Channels);
    // [中文导读] [AllReduce逐行 S73] 该 warning 文本称暂未进行资源计算，但上文实际已填写 Thread、notify 和 channel 请求；不能按日志解释为空请求。
    HCCL_WARNING("Resource calculation is temporarily not performed in the template.");
    // [中文导读] [AllReduce逐行 S74] 资源需求填写成功，实际申请由上层执行。
    return HCCL_SUCCESS;
// [中文导读] [AllReduce逐行 S75] 结束 CalcRes 实现；其返回状态或已写回字段由调用者接收。
}

// [中文导读] [AllReduce逐行 S77] 返回 OneShot 保存每个 Rank 完整输入所需的 CCL 容量倍数。
u64 InsTempAllReduceMesh1DOneShot::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
// [中文导读] [AllReduce逐行 S78] 进入 CalcScratchMultiple 的实现作用域；OneShot 要为每个 Rank 保存完整数据，scratch 倍数等于模板通信域 Rank 数。
{
    // [中文导读] [AllReduce逐行 S79] 该倍数不随输入缓冲区类型改变。
    (void)inBuffType;
    // [中文导读] [AllReduce逐行 S80] 该倍数不随输出缓冲区类型改变。
    (void)outBuffType;
    // [中文导读] [AllReduce逐行 S81] 返回 Rank 总数作为 CCL 临时内存倍数：每个 Rank 的完整输入占一个槽。
    u64 scratchMultiple = templateRankSize_;
    // [中文导读] [AllReduce逐行 S82] 返回 Rank 数倍的临时内存需求。
    return scratchMultiple;
// [中文导读] [AllReduce逐行 S83] 结束 CalcScratchMultiple 实现；其返回状态或已写回字段由调用者接收。
}

// [中文导读] [AllReduce逐行 S85] 按本轮字节数计算所有 Rank 的完整输入槽布局。
HcclResult InsTempAllReduceMesh1DOneShot::CalcSlice(const u64 dataSize, RankSliceInfo& sliceInfoVec) const
// [中文导读] [AllReduce逐行 S86] 进入 CalcSlice 的实现作用域；为每个 Rank 计算等长完整数据槽 offset=rankIdx×dataSize，验证最后槽终点等于总槽跨度。
{
    // [中文导读] [AllReduce逐行 S87] 每个 Rank 仅需要一个连续完整输入片，先建立单片列表。
    std::vector<SliceInfo> tmp(1);
    // [中文导读] [AllReduce逐行 S88] 将切片表扩成 N 个 Rank，每个 Rank 一个槽。
    sliceInfoVec.resize(templateRankSize_, tmp);

    // [中文导读] [AllReduce逐行 S90] 首个 Rank 槽从 CCL 字节偏移0开始。
    u64 accumOff = 0;
    // [中文导读] [AllReduce逐行 S91] 逐 Rank 槽计算完整输入片起点，槽大小均为本轮输入字节数。
    for (u32 rankIdx = 0; rankIdx < sliceInfoVec.size(); rankIdx++) {
        // [中文导读] [AllReduce逐行 S92] 当前槽起点采用累积偏移，长度采用本轮完整输入大小。
        SliceInfo slice = {accumOff, dataSize};
        // [中文导读] [AllReduce逐行 S93] 把该完整输入片保存到当前 Rank 的第一个且唯一槽。
        sliceInfoVec[rankIdx][0] = slice;
        // [中文导读] [AllReduce逐行 S94] 累积偏移推进一个完整输入大小，形成下一个 Rank 的槽起点。
        accumOff += dataSize;
    // [中文导读] [AllReduce逐行 S95] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
    }
    // [中文导读] [AllReduce逐行 S96] 检查 N 个槽是否连续覆盖 N 倍输入容量，不符则返回内部错误。
    CHK_PRT_RET(
        // [中文导读] [AllReduce逐行 S97] 末槽起点加完整数据字节数必须等于 Rank 数乘数据字节数；不一致返回内部错误。
        (sliceInfoVec[templateRankSize_ - 1][0].offset + sliceInfoVec[templateRankSize_ - 1][0].size
         // [中文导读] [AllReduce逐行 S98] 完整槽跨度应等于本轮输入字节数乘 Rank 数。
         != dataSize * templateRankSize_),
        // [中文导读] [AllReduce逐行 S99] 开始 HCCL_ERROR 诊断输出，记录 CalcSlice 当前阶段的参数、候选或错误；日志本身不决定返回码。
        HCCL_ERROR("[InsTempAllReduceMesh1DOneShot] Rank [%d], SliceInfo calculation error!", myRank_),
        // [中文导读] [AllReduce逐行 S100] 切片总跨度不一致时向调用者返回内部错误。
        HcclResult::HCCL_E_INTERNAL);
    // [中文导读] [AllReduce逐行 S101] 所有 Rank 的槽计算与末端校验成功。
    return HCCL_SUCCESS;
// [中文导读] [AllReduce逐行 S102] 结束 CalcSlice 实现；其返回状态或已写回字段由调用者接收。
}

// [中文导读] [AllReduce逐行 S104] 定义 KernelRun 入口：OneShot 展开：校验 Thread 数，计算 Rank 槽，主从前同步，Peer 全量交换，后同步，普通内存再本地归约。
HcclResult InsTempAllReduceMesh1DOneShot::KernelRun(
    // [中文导读] [AllReduce逐行 S105] 接收算子参数、本轮数据偏移/count，以及已申请的 Thread/通道资源。
    const OpParam& param, const TemplateDataParams& tempAlgParams, TemplateResource& templateResource)
// [中文导读] [AllReduce逐行 S106] 进入 KernelRun 的实现作用域；OneShot 展开：校验 Thread 数，计算 Rank 槽，主从前同步，Peer 全量交换，后同步，普通内存再本地归约。
{
    // [中文导读] [AllReduce逐行 S107] 读取实际 Thread 总数，供一个主 Thread 加 N−1 个 Peer Thread 使用。
    threadNum_ = templateResource.threads.size();
    // [中文导读] [AllReduce逐行 S108] 记录本轮完整输入字节数，所有 Peer 都交换此长度。
    processSize_ = tempAlgParams.sliceSize;
    // [中文导读] [AllReduce逐行 S109] 记录本轮元素数，交给本地或远端归约原语。
    count_ = tempAlgParams.count;
    // [中文导读] [AllReduce逐行 S110] 从统一算子描述读取当前元素类型。
    dataType_ = param.DataDes.dataType;
    // [中文导读] [AllReduce逐行 S111] 保存本轮是否允许直接访问 Peer 对称用户输入。
    supportSymmetricMemAccess_ = param.supportSymmetricMemory;
    // [中文导读] [AllReduce逐行 S112] 计算本轮是否需要 AICPU 软件归约。
    needAicpuReduce_
        // [中文导读] [AllReduce逐行 S113] INT64 或 UINT64 归约采用软件路径。
        = dataType_ == HcclDataType::HCCL_DATA_TYPE_INT64 || dataType_ == HcclDataType::HCCL_DATA_TYPE_UINT64
          // [中文导读] [AllReduce逐行 S114] FP64 或 PROD 也进入软件路径，其余普通支持类型用 HCOMM 本地归约。
          || dataType_ == HcclDataType::HCCL_DATA_TYPE_FP64 || param.reduceType == HcclReduceOp::HCCL_REDUCE_PROD;
    // [中文导读] [AllReduce逐行 S115] 开始 HCCL_INFO 诊断输出，记录 KernelRun 当前阶段的参数、候选或错误；日志本身不决定返回码。
    HCCL_INFO("[InsTempAllReduceMesh1DOneShot] Run Start");
    // [中文导读] [AllReduce逐行 S116] 实际 Thread 列表长度必须等于算法通信域 Rank 数，才能按 Peer 使用从 Thread。
    CHK_PRT_RET(
        // [中文导读] [AllReduce逐行 S117] Thread 数必须等于算法通信域 Rank 数，才能一一对应主/从任务。
        threadNum_ != templateRankSize_,
        // [中文导读] [AllReduce逐行 S118] 开始 HCCL_ERROR 诊断输出，记录 KernelRun 当前阶段的参数、候选或错误；日志本身不决定返回码。
        HCCL_ERROR(
            // [中文导读] [AllReduce逐行 S119] 续接 KernelRun 的诊断格式串，展示此行列出的字段标签；它是编译期字符串拼接，不发起额外操作。
            "[InsTempAllReduceMesh1DOneShot][KernelRun] thread num is invalid, need[%u], actual[%u].",
            // [中文导读] [AllReduce逐行 S120] 为 KernelRun 的诊断/错误宏提供实参：算法通信域 Rank 数, 当前实际 Thread 总数，与前面的格式占位依次对应。
            templateRankSize_, threadNum_),
        // [中文导读] [AllReduce逐行 S121] Thread 资源数量不匹配时终止该轮模板执行。
        HcclResult::HCCL_E_INTERNAL);

    // [中文导读] [AllReduce逐行 S123] 创建本轮按 Rank 槽组织的切片表。
    RankSliceInfo sliceInfoVec;
    // [中文导读] [AllReduce逐行 S124] 按本轮完整数据字节数构造各 Rank 槽及总跨度。
    CHK_RET(CalcSlice(processSize_, sliceInfoVec));
    // [中文导读] [AllReduce逐行 S125] 存在 Peer 时才安排主 Thread 对从 Thread 的放行依赖。
    if (threadNum_ > 1) {
        // [中文导读] [AllReduce逐行 S126] Thread0 为主 Thread，其余句柄组成 Peer 通信从 Thread 列表。
        std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
        // [中文导读] [AllReduce逐行 S127] 生成每个从 Thread 的前同步索引，各从 Thread 都用自己的槽0。
        GetNotifyIdxMainToSub(notifyIdxMainToSub_);
        // [中文导读] [AllReduce逐行 S128] 先将主 Thread 的依赖放行到每个从 Thread，通知索引长度必须匹配。
        CHK_RET(PreSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxMainToSub_));
    // [中文导读] [AllReduce逐行 S129] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
    }
    // [中文导读] [AllReduce逐行 S130] 传本轮切片、通道和 Thread 列表，复制本地输入并交换 Peer 全量输入。
    CHK_RET(RunAllReduce(param, templateResource.channels, templateResource.threads, tempAlgParams, sliceInfoVec));
    // [中文导读] [AllReduce逐行 S131] 存在 Peer 时安排全部通信从 Thread 向主 Thread 汇合。
    if (threadNum_ > 1) {
        // [中文导读] [AllReduce逐行 S132] 再次取 Thread1..N−1，作为后同步的完成来源。
        std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
        // [中文导读] [AllReduce逐行 S133] 生成各从 Thread 向主 Thread 报告完成的不同槽索引。
        GetNotifyIdxSubToMain(notifyIdxSubToMain_);
        // [中文导读] [AllReduce逐行 S134] 所有从 Thread 的通信完成汇合到主 Thread，再允许主 Thread 执行本地归约。
        CHK_RET(PostSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxSubToMain_));
    // [中文导读] [AllReduce逐行 S135] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
    }
    // [中文导读] [AllReduce逐行 S136] 普通内存需要把已接收的 CCL 输入槽合并；对称内存已在远端 ReadReduce 中合并。
    if (!supportSymmetricMemAccess_) {
        // [中文导读] [AllReduce逐行 S137] CHK_PRT 只记录 PostLocalReduce 返回的错误而不从 KernelRun 返回；最终 success 可掩盖此局部失败，不能承诺全部归约错误上送。
        CHK_PRT(PostLocalReduce(param, templateResource.threads, tempAlgParams, sliceInfoVec));
    // [中文导读] [AllReduce逐行 S138] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
    }
    // [中文导读] [AllReduce逐行 S139] 开始 HCCL_INFO 诊断输出，记录 KernelRun 当前阶段的参数、候选或错误；日志本身不决定返回码。
    HCCL_INFO("[InsTempAllReduceMesh1DOneShot][KernelRun] AllReduceMesh1DOneShot finished: rank[%d] end", myRank_);
    // [中文导读] [AllReduce逐行 S140] 模板返回成功；前面的 CHK_PRT 只记录本地归约失败，不改变此返回值。
    return HCCL_SUCCESS;
// [中文导读] [AllReduce逐行 S141] 结束 KernelRun 实现；其返回状态或已写回字段由调用者接收。
}

// [中文导读] [AllReduce逐行 S143] 定义 RunAllReduce 入口：OneShot 先把自身输入复制到输出，再每个从 Thread 面向一个 Peer 全量交换；普通路径 Write 到对端 CCL 源 Rank 槽。
HcclResult InsTempAllReduceMesh1DOneShot::RunAllReduce(
    // [中文导读] [AllReduce逐行 S144] 接收每个 Peer 的通道资源与统一算子参数。
    const OpParam& param, const std::map<u32, std::vector<ChannelInfo>>& channels,
    // [中文导读] [AllReduce逐行 S145] 接收主/从 Thread 句柄及本轮输入输出偏移。
    const std::vector<ThreadHandle>& threads, const TemplateDataParams& tempAlgParams,
    // [中文导读] [AllReduce逐行 S146] 接收各发送源 Rank 的 CCL 槽布局。
    const RankSliceInfo& sliceInfoVec)
// [中文导读] [AllReduce逐行 S147] 进入 RunAllReduce 的实现作用域；OneShot 先把自身输入复制到输出，再每个从 Thread 面向一个 Peer 全量交换；普通路径 Write 到对端 CCL 源 Rank 槽。
{
    // [中文导读] [AllReduce逐行 S148] 开始 HCCL_INFO 诊断输出，记录 RunAllReduce 当前阶段的参数、候选或错误；日志本身不决定返回码。
    HCCL_INFO("[InsTempAllReduceMesh1DOneShot][RunAllReduce] send/recv: rank[%d]", myRank_);

    // [中文导读] [AllReduce逐行 S150] 建立本 Rank 当前轮的完整用户输入片。
    DataSlice usrInSlices
        // [中文导读] [AllReduce逐行 S151] 输入片使用用户基址、本轮输入字节偏移、整轮字节数和元素数。
        = DataSlice(tempAlgParams.buffInfo.inputPtr, tempAlgParams.buffInfo.inBuffBaseOff, processSize_, count_);
    // [中文导读] [AllReduce逐行 S152] 建立本 Rank 当前轮的用户输出累加片。
    DataSlice usrOutSlices
        // [中文导读] [AllReduce逐行 S153] 输出片使用对应输出字节偏移，长度与输入完全相同。
        = DataSlice(tempAlgParams.buffInfo.outputPtr, tempAlgParams.buffInfo.outBuffBaseOff, processSize_, count_);

    // 主流 - 本地拷贝
    // [中文导读] [AllReduce逐行 S156] 在主 Thread 将本 Rank 输入完整复制到本 Rank 输出，作为后续 SUM 的初始累加值。
    CHK_RET(static_cast<HcclResult>(LocalCopy(threads[0], usrInSlices, usrOutSlices)));

    // 特殊场景
    // [中文导读] [AllReduce逐行 S159] 通信域只有一个 Rank 时，本地复制已完成全部 AllReduce 工作，无需 Peer 网络交换。
    if (subCommRanks_[0].size() == 1) {
        // [中文导读] [AllReduce逐行 S160] 单 Rank 的本地复制已满足 AllReduce，结束当前轮。
        return HCCL_SUCCESS;
    // [中文导读] [AllReduce逐行 S161] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
    }

    // 从流
    // [中文导读] [AllReduce逐行 S164] 从 que=1 遍历到 Rank 数减 1，为每个 Peer 分配一个从 Thread；que=0 留给主 Thread。
    for (u32 queIdx = 1; queIdx < threadNum_; queIdx++) {
        // [中文导读] [AllReduce逐行 S165] 代码用本地用户 Rank 加从 Thread 序号再对通信域规模取模得到 Peer 索引；本例用户 Rank 为连续0..3。
        u32 nextRank = (myRank_ + queIdx) % templateRankSize_;
        // [中文导读] [AllReduce逐行 S166] 把 Peer 槽索引映射为接收方向的用户 Rank。
        u32 fromRank = subCommRanks_[0][nextRank];
        // [中文导读] [AllReduce逐行 S167] 发送与接收指向同一个 Peer 用户 Rank。
        u32 toRank = subCommRanks_[0][nextRank];

        // [中文导读] [AllReduce逐行 S169] 从资源映射中取得接收 Peer 的第一个 ChannelInfo；这里假定匹配 Peer 且至少有一个通道。
        const ChannelInfo& linkRecv = channels.at(fromRank)[0]; // linkRecv - 从fromRank接收的链路
        // [中文导读] [AllReduce逐行 S170] 本算法发送和接收面向同一个 Peer，分别取其首个 ChannelInfo。
        const ChannelInfo& linkSend = channels.at(toRank)[0];   // linkSend - 向toRank发送的链路

        // [中文导读] [AllReduce逐行 S172] 对称内存时从 Peer 用户输入直接读并归约到本端输出，不写普通 CCL 接收槽。
        if (supportSymmetricMemAccess_) {
            // [中文导读] [AllReduce逐行 S173] 对称内存使用直接远端输入 ReadReduce，完成该 Peer 后跳过普通 CCL Write 构造。
            CHK_RET(SymmetricReadReduce(linkSend, linkRecv, tempAlgParams, usrOutSlices, threads, queIdx));
            // [中文导读] [AllReduce逐行 S174] 该 Peer 已安排 ReadReduce，跳过普通 Write 描述与 CCL 槽构造。
            continue;
        // [中文导读] [AllReduce逐行 S175] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
        }

        // [中文导读] [AllReduce逐行 S177] 保存普通 Write 的本地源片描述。
        std::vector<DataSlice> txSrcSlices;
        // [中文导读] [AllReduce逐行 S178] 保存普通 Write 的 Peer 远端目标片描述。
        std::vector<DataSlice> txDstSlices;
        // [中文导读] [AllReduce逐行 S179] 取得发送 Peer 暴露的远端 CCL 基址。
        void* txCclBuffAddr = linkSend.remoteCclMem.addr;
        // [中文导读] [AllReduce逐行 S180] 用本地用户 Rank 直接索引切片表，计算 Peer 远端 CCL 中本 Rank 输入槽的偏移；本例0..3与槽编号一致。
        u64 txDstOffset = sliceInfoVec[myRank_][0].offset + tempAlgParams.buffInfo.hcclBuffBaseOff;
        // [中文导读] [AllReduce逐行 S181] 取得本 Rank 完整输入槽的字节长度。
        u64 txDstSize = sliceInfoVec[myRank_][0].size;
        // [中文导读] [AllReduce逐行 S182] 发送源直接引用本 Rank 当前轮的用户输入片。
        DataSlice txSrcSlice = usrInSlices;
        // [中文导读] [AllReduce逐行 S183] 目标为 Peer 远端 CCL 的本 Rank 槽，长度和元素数对应完整输入。
        DataSlice txDstSlice = DataSlice(txCclBuffAddr, txDstOffset, txDstSize, count_);
        // [中文导读] [AllReduce逐行 S184] 将用户输入片加入发送方向源列表。
        txSrcSlices.push_back(txSrcSlice);
        // [中文导读] [AllReduce逐行 S185] 将 Peer 远端 CCL 槽加入发送方向目标列表。
        txDstSlices.push_back(txDstSlice);

        // [中文导读] [AllReduce逐行 S187] 创建接收方向源描述列表；普通 Write 通用包装实际只消费发送片。
        std::vector<DataSlice> rxSrcSlices;
        // [中文导读] [AllReduce逐行 S188] 创建接收方向目标描述列表，协议接收完成由通知保证。
        std::vector<DataSlice> rxDstSlices;
        // [中文导读] [AllReduce逐行 S189] 取接收 Peer 的远端 CCL 基址，供接收方向描述组装。
        void* rxCclBuffAddr = linkRecv.remoteCclMem.addr;
        // [中文导读] [AllReduce逐行 S190] 代码以接收 Peer 的用户 Rank 直接索引槽表，再加本轮 CCL 基础偏移。
        u64 rxDstOffset = sliceInfoVec[fromRank][0].offset + tempAlgParams.buffInfo.hcclBuffBaseOff;
        // [中文导读] [AllReduce逐行 S191] 取接收 Peer 对应完整输入槽的字节长度。
        u64 rxDstSize = sliceInfoVec[fromRank][0].size;
        // [中文导读] [AllReduce逐行 S192] 接收方向源描述同样引用本 Rank 用户输入。
        DataSlice rxSrcSlice = usrInSlices;
        // [中文导读] [AllReduce逐行 S193] 组装接收方向远端 CCL 目标描述；它不触发额外的主动接收拷贝。
        DataSlice rxDstSlice = DataSlice(rxCclBuffAddr, rxDstOffset, rxDstSize, count_);
        // [中文导读] [AllReduce逐行 S194] 保存接收方向源片，供 SendRecvInfo 结构完整组装。
        rxSrcSlices.push_back(rxSrcSlice);
        // [中文导读] [AllReduce逐行 S195] 保存接收方向目标片，普通 Write 的接收依赖使用 Peer 通知。
        rxDstSlices.push_back(rxDstSlice);

        // [中文导读] [AllReduce逐行 S197] 建立双向协议对象，保存 send/recv 通道与数据片。
        SendRecvInfo sendRecvInfo{
            // [中文导读] [AllReduce逐行 S198] 填入普通 Write 的发送片、接收描述和数据类型。
            {linkSend, linkRecv}, {{txSrcSlices, txDstSlices}, {rxSrcSlices, rxDstSlices}}, dataType_};
        // [中文导读] [AllReduce逐行 S199] 检查双向 Write 包装返回码，非零结果打印错误并返回内部错误。
        CHK_PRT_RET(
            // [中文导读] [AllReduce逐行 S200] 提交普通内存双向批 Write；实际批能力缺失时包装回退到逐片 Write。
            SendRecvBatchWrite(sendRecvInfo, threads[queIdx]),
            // [中文导读] [AllReduce逐行 S201] Peer 全量输入交换提交失败时将错误映射为内部错误。
            HCCL_ERROR("[InsTempAllReduceMesh1DOneShot] RunAllReduce SendRecv failed"), HcclResult::HCCL_E_INTERNAL);
    // [中文导读] [AllReduce逐行 S202] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
    }

    // [中文导读] [AllReduce逐行 S204] 所有 Peer 的交换与握手提交成功，后续由主从后同步汇合。
    return HCCL_SUCCESS;
// [中文导读] [AllReduce逐行 S205] 结束 RunAllReduce 实现；其返回状态或已写回字段由调用者接收。
}

// [中文导读] [AllReduce逐行 S207] 定义 SymmetricReadReduce 入口：对称内存分支构造对端输入到本端输出的 ReadReduce 片，采用双向 ReadReduce 握手。
HcclResult InsTempAllReduceMesh1DOneShot::SymmetricReadReduce(
    // [中文导读] [AllReduce逐行 S208] 使用发送/接收 Peer 通道和本轮参数，定位对称用户输入。
    const ChannelInfo& linkSend, const ChannelInfo& linkRecv, const TemplateDataParams& tempAlgParams,
    // [中文导读] [AllReduce逐行 S209] 本地输出片作为归约目标，在当前 Peer 的从 Thread 上执行。
    const DataSlice& usrOutSlices, const std::vector<ThreadHandle>& threads, u32 queIdx)
// [中文导读] [AllReduce逐行 S210] 进入 SymmetricReadReduce 的实现作用域；对称内存分支构造对端输入到本端输出的 ReadReduce 片，采用双向 ReadReduce 握手。
{
    // 对称内存路径：从远端 input read+reduce 到本地 output
    // [中文导读] [AllReduce逐行 S212] 保存对端读取本端输入的发送方向源描述。
    std::vector<DataSlice> txSrcSlices;
    // [中文导读] [AllReduce逐行 S213] 保存发送方向对应的对称远端输入描述。
    std::vector<DataSlice> txDstSlices;
    // [中文导读] [AllReduce逐行 S214] 保存本端主动读取的 Peer 用户输入片。
    std::vector<DataSlice> rxSrcSlices;
    // [中文导读] [AllReduce逐行 S215] 保存本端 ReadReduce 的输出累加目标片。
    std::vector<DataSlice> rxDstSlices;
    // [中文导读] [AllReduce逐行 S216] 发送方向公布本端当前轮用户输入片。
    txSrcSlices.push_back(
        // [中文导读] [AllReduce逐行 S217] 本端公布的输入片含当前轮基址偏移、完整字节数和元素数。
        DataSlice(tempAlgParams.buffInfo.inputPtr, tempAlgParams.buffInfo.inBuffBaseOff, processSize_, count_));
    // [中文导读] [AllReduce逐行 S218] 发送方向建立对端对称输入描述。
    txDstSlices.push_back(
        // [中文导读] [AllReduce逐行 S219] 对端输入描述取 remoteInputGraphMode 地址并使用同一轮输入偏移。
        DataSlice(linkSend.remoteInputGraphMode.addr, tempAlgParams.buffInfo.inBuffBaseOff, processSize_, count_));
    // [中文导读] [AllReduce逐行 S220] 接收方向建立将主动读取的 Peer 用户输入源。
    rxSrcSlices.push_back(
        // [中文导读] [AllReduce逐行 S221] ReadReduce 从接收 Peer 的对称输入地址加本轮输入偏移读取完整输入。
        DataSlice(linkRecv.remoteInputGraphMode.addr, tempAlgParams.buffInfo.inBuffBaseOff, processSize_, count_));
    // [中文导读] [AllReduce逐行 S222] 读归约目标直接采用本端当前轮用户输出累加片。
    rxDstSlices.push_back(usrOutSlices);
    // [中文导读] [AllReduce逐行 S223] 建立双向 ReadReduce 协议对象。
    SendRecvReduceInfo sendRecvReduceInfo{
        // [中文导读] [AllReduce逐行 S224] 填入双向通道、源目标片、元素类型和归约运算。
        {linkSend, linkRecv}, {{txSrcSlices, txDstSlices}, {rxSrcSlices, rxDstSlices}}, dataType_, reduceOp_};
    // [中文导读] [AllReduce逐行 S225] 读归约包装失败时返回内部错误，不继续当前 Peer。
    CHK_PRT_RET(
        // [中文导读] [AllReduce逐行 S226] 对称内存分支把对端用户输入读并归约进本地输出，不经过普通 CCL 接收槽。
        SendRecvBatchReadReduce(sendRecvReduceInfo, threads[queIdx]),
        // [中文导读] [AllReduce逐行 S227] 开始 HCCL_ERROR 诊断输出，记录 SymmetricReadReduce 当前阶段的参数、候选或错误；日志本身不决定返回码。
        HCCL_ERROR("[InsTempAllReduceMesh1DOneShot] RunAllReduce SendRecvBatchReadReduce failed"),
        // [中文导读] [AllReduce逐行 S228] 双向 ReadReduce 返回非零时向模板调用者报告内部错误。
        HcclResult::HCCL_E_INTERNAL);
    // [中文导读] [AllReduce逐行 S229] 此 Peer 的读归约与握手已提交成功。
    return HCCL_SUCCESS;
// [中文导读] [AllReduce逐行 S230] 结束 SymmetricReadReduce 实现；其返回状态或已写回字段由调用者接收。
}

// [中文导读] [AllReduce逐行 S232] 定义 PostLocalReduce 入口：普通内存交换完成后从每个其它 Rank 的本端 CCL 槽归约到本端输出；特殊类型先提交并 Join 所有 Thread。
HcclResult InsTempAllReduceMesh1DOneShot::PostLocalReduce(
    // [中文导读] [AllReduce逐行 S233] 传入已申请 Thread 和当前轮输出参数，使用主 Thread 累加各接收槽。
    const OpParam& param, const std::vector<ThreadHandle>& threads, const TemplateDataParams& tempAlgParams,
    // [中文导读] [AllReduce逐行 S234] 传入普通 Write 使用的 Rank 槽布局以定位各来源。
    const RankSliceInfo& sliceInfoVec)
// [中文导读] [AllReduce逐行 S235] 进入 PostLocalReduce 的实现作用域；普通内存交换完成后从每个其它 Rank 的本端 CCL 槽归约到本端输出；特殊类型先提交并 Join 所有 Thread。
{
    // [中文导读] [AllReduce逐行 S236] 开始 HCCL_INFO 诊断输出，记录 PostLocalReduce 当前阶段的参数、候选或错误；日志本身不决定返回码。
    HCCL_INFO("[InsTempAllReduceMesh1DOneShot][RunAllReduce] reduce: rank[%d]", myRank_);
    // 增加thread synchronize以支持64类数据类型
    // [中文导读] [AllReduce逐行 S238] 64位类型或 PROD 需要 CPU 直接访问输入，先提交并等待全部交换完成。
    if (needAicpuReduce_) {
        // 启动任务并等待所有threads任务执行完成
        // [中文导读] [AllReduce逐行 S240] 软件归约需要真正可读的数据，先结束当前 Thread 的任务批次以便提交。
        CHK_RET(static_cast<HcclResult>(HcommBatchModeEnd(param.algTag)));
        // [中文导读] [AllReduce逐行 S241] 重新开启后续 Thread 的批次记录；下方 Join 显式等待已提交交换就绪。
        CHK_RET(static_cast<HcclResult>(HcommBatchModeStart(param.algTag)));
        // [中文导读] [AllReduce逐行 S242] 逐 Thread 显式等待已提交任务，确保软件归约读到完成的数据。
        for (const auto& thread : threads) {
            // [中文导读] [AllReduce逐行 S243] 逐 Thread Join 等待所有前序传输完成，避免 CPU 软件归约读取未到达的输入。
            CHK_RET(static_cast<HcclResult>(HcommThreadJoin(thread, CUSTOM_TIMEOUT)));
        // [中文导读] [AllReduce逐行 S244] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
        }
    // [中文导读] [AllReduce逐行 S245] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
    }

    // [中文导读] [AllReduce逐行 S247] 建立当前轮本端输出的完整累加片。
    DataSlice usrOutSlices
        // [中文导读] [AllReduce逐行 S248] 目标输出基址加当前轮输出偏移，容量与本轮完整输入一致。
        = DataSlice(tempAlgParams.buffInfo.outputPtr, tempAlgParams.buffInfo.outBuffBaseOff, processSize_, count_);

    // [中文导读] [AllReduce逐行 S250] 遍历每个源 Rank 的 CCL 槽，随后跳过自身并逐槽归约到用户输出。
    for (u32 rankIdx = 0; rankIdx < subCommRanks_[0].size(); rankIdx++) {
        // [中文导读] [AllReduce逐行 S251] 使用当前槽编号定位源 Rank 完整输入。
        u32 curRank = rankIdx;
        // 遍历除自身外所有rank，计算reduce(scratch, local-usrout)
        // [中文导读] [AllReduce逐行 S253] 当前槽编号等于本地用户 Rank 时跳过；本例0..3对应自身槽，本地输入已经复制进输出。
        if (curRank == myRank_) {
            // [中文导读] [AllReduce逐行 S254] 自身贡献已由输入复制到输出，跳过本 Rank 槽以免重复累加。
            continue;
        // [中文导读] [AllReduce逐行 S255] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
        }

        // 执行本地归约
        // [中文导读] [AllReduce逐行 S258] 变量名 remotePtr 对应本端 CCL 缓冲区地址，数据来源是 Peer 写入；此行没有新远端读取。
        void* RemotePtr = tempAlgParams.buffInfo.hcclBuff.addr;
        // [中文导读] [AllReduce逐行 S259] 当前来源在本端 CCL 的字节偏移等于其槽起点加本轮 CCL 基础偏移。
        u64 curSrcOffset = sliceInfoVec[curRank][0].offset + tempAlgParams.buffInfo.hcclBuffBaseOff;
        // [中文导读] [AllReduce逐行 S260] 从切片表读取该来源的完整输入字节长度。
        u64 curSrcSize = sliceInfoVec[curRank][0].size;
        // [中文导读] [AllReduce逐行 S261] 以本端 CCL 地址、源槽偏移和本轮 count 建立本地归约源片。
        DataSlice curSrcSlice = DataSlice(RemotePtr, curSrcOffset, curSrcSize, count_);
        // [中文导读] [AllReduce逐行 S262] 归约目标始终为当前轮用户输出片，逐来源累加到同一输出。
        DataSlice curDstSlice = usrOutSlices;

        // [中文导读] [AllReduce逐行 S264] 将一个其它 Rank 的本端 CCL 完整输入槽归约进本 Rank 输出，普通 FP32 SUM 使用本地 HCOMM 归约。
        CHK_RET(static_cast<HcclResult>(LocalReduce(threads[0], curSrcSlice, curDstSlice, dataType_, reduceOp_)));
    // [中文导读] [AllReduce逐行 S265] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
    }
    // [中文导读] [AllReduce逐行 S266] 其它来源已逐槽提交/执行归约；调用者仍应按各原语完成语义理解成功。
    return HCCL_SUCCESS;
// [中文导读] [AllReduce逐行 S267] 结束 PostLocalReduce 实现；其返回状态或已写回字段由调用者接收。
}

// [中文导读] [AllReduce逐行 S269] 生成主 Thread 放行各从 Thread 时使用的通知槽列表。
void InsTempAllReduceMesh1DOneShot::GetNotifyIdxMainToSub(std::vector<u32>& notifyIdxMainToSub)
// [中文导读] [AllReduce逐行 S270] 进入 GetNotifyIdxMainToSub 的实现作用域；所有从 Thread 使用各自通知槽 0 接收全局主 Thread 的放行通知。
{
    // [中文导读] [AllReduce逐行 S271] 清除此前通知映射，重新按当前 Rank 数生成。
    notifyIdxMainToSub.clear();
    // [中文导读] [AllReduce逐行 S272] 总 Thread 数至少1，通常与通信域 Rank 数相同。
    u32 threadNum = templateRankSize_ > 1 ? templateRankSize_ : 1;
    // [中文导读] [AllReduce逐行 S273] 排除主 Thread，得到需要前同步的从 Thread 数。
    u32 slaveThreadNum = threadNum - 1;
    // [中文导读] [AllReduce逐行 S274] 为每个从 Thread 建立一项前同步通知映射。
    for (u32 slaveThreadIdx = 0; slaveThreadIdx < slaveThreadNum; slaveThreadIdx++) {
        // [中文导读] [AllReduce逐行 S275] 每个从 Thread 都在自身通知槽0等待主 Thread，句柄不同无需不同索引。
        notifyIdxMainToSub.push_back(0);
    // [中文导读] [AllReduce逐行 S276] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
    }
// [中文导读] [AllReduce逐行 S277] 结束 GetNotifyIdxMainToSub 实现；其返回状态或已写回字段由调用者接收。
}

// [中文导读] [AllReduce逐行 S279] 生成各从 Thread 报告完成时使用的主 Thread 槽列表。
void InsTempAllReduceMesh1DOneShot::GetNotifyIdxSubToMain(std::vector<u32>& notifyIdxSubToMain)
// [中文导读] [AllReduce逐行 S280] 进入 GetNotifyIdxSubToMain 的实现作用域；主 Thread 为各从 Thread 分配不同完成通知索引，形成逐个汇合。
{
    // [中文导读] [AllReduce逐行 S281] 清除此前后同步映射，重新生成当前轮所需索引。
    notifyIdxSubToMain.clear();
    // [中文导读] [AllReduce逐行 S282] 计算总 Thread 数并保留单 Rank 至少一个主 Thread。
    u32 threadNum = templateRankSize_ > 1 ? templateRankSize_ : 1;
    // [中文导读] [AllReduce逐行 S283] 主 Thread 需要 N−1 个独立完成通知槽。
    u32 notifyNum = threadNum - 1;
    // [中文导读] [AllReduce逐行 S284] 按0到N−2依次分配从 Thread 完成汇合索引。
    for (u32 notifyIdx = 0; notifyIdx < notifyNum; notifyIdx++) {
        // [中文导读] [AllReduce逐行 S285] 为该从 Thread 追加唯一的主 Thread 完成槽，避免与其它来源复用同一等待条件。
        notifyIdxSubToMain.push_back(notifyIdx);
    // [中文导读] [AllReduce逐行 S286] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
    }
// [中文导读] [AllReduce逐行 S287] 结束 GetNotifyIdxSubToMain 实现；其返回状态或已写回字段由调用者接收。
}

} // namespace ops_hccl
