#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import os

import torch
import torch_npu  # noqa: F401  注册 npu 后端
import torch.distributed as dist

REPLAY_TIMES = 4

"""
AllReduce 算子 ACL Graph 捕获模式 demo（Python / torch_npu + torch.distributed）。

对应文档 docs/zh/aclgraph/aclgraph_introduction.md：
  - §2.1 捕获生命周期（NPUGraph 包装了 aclmdlRICaptureBegin/End/ExecuteAsync）
  - §2.3 捕获期约束（HCCL watchdog 排空、默认 stream 禁止、资源需保活）
  - §5   多 stream 拓扑（HCCL 内部多 stream 由框架自动纳入 model_ri）
  - 附录 B torch_npu 源码对接（NPUGraph / ProcessGroupHCCL）

核心思想：把"一次下发、多次执行"的 Host 调度开销摊销到一次 capture。
torch.npu.NPUGraph 把 aclmdlRICaptureBegin/End/ExecuteAsync 封装为
`with torch.npu.graph(g):` 上下文 + `g.replay()`，HCCL 集合通信在上下文内
被一并捕获。

运行前提：
  - 安装 torch + torch_npu，CANN >= 8.5
  - 多卡环境，通过 torchrun / mpirun 拉起，HCCL 后端
运行示例：
  torchrun --nproc_per_node=2 allreduce_aclgraph_demo.py

  参考：ACL Graph 基础 API 用法见 cann/runtime 仓示例
  https://gitcode.com/cann/runtime/blob/master/example/2_advanced_features/model_ri/0_simple_model/main.cpp
"""


def main() -> None:
    # ===== 0. 进程组初始化（hccl 后端）=====
    dist.init_process_group(backend="hccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.npu.set_device(rank)

    # ===== 1. 捕获前准备：静态输入显存 =====
    device = torch.device(f"npu:{rank}")
    static_input = torch.full((world_size,), float(rank), dtype=torch.float32, device=device)

    # ===== 2. 捕获：NPUGraph 上下文 =====
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        # 业务计算 + 集合通信，整段被捕获、暂存为可重放 DAG
        out = static_input * 2.0
        dist.all_reduce(out, op=dist.ReduceOp.SUM)

    # ===== 3. 重放：一次 capture，多次 1-syscall replay =====
    #    replay 操作同一块静态显存；如需逐次更换输入，走 §3 任务更新或重新捕获。
    for _ in range(REPLAY_TIMES):
        graph.replay()
    torch.npu.synchronize()

    # ===== 4. 取回结果并打印 =====
    #    期望：out = 2 * sum(0..world_size-1) = world_size*(world_size-1)
    result = out.cpu().tolist()

    # ===== 5. 释放（顺序约束见 §2.4）=====
    #    先 reset 图（释放 ACL mempool）→ 再 destroy_process_group（HCCL watchdog join）
    #    → 最后 HcclCommDestroy。颠倒顺序会触发 use-after-free。
    graph.reset()
    dist.destroy_process_group()


if __name__ == "__main__":
    os.environ.setdefault("HCCL_TIMEOUT", "1800")
    main()
