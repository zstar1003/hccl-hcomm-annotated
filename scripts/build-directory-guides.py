#!/usr/bin/env python3
"""Build two offline source catalogues from tracked paths and local content.

No network or dependencies. --check verifies byte-for-byte reproducibility.
File summaries are navigation aids, not claims of full algorithm review.
"""
import argparse
import collections
import html
import json
from pathlib import Path, PurePosixPath
import posixpath
import re
import subprocess
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "catalog"

# These roles follow the locked snapshots' architecture documents. The actual
# path inventory, including resources/ and compatibility directories, comes
# from git rather than the architecture documents' target directory trees.
ROLES = {
    "src": "实现源码", "ops": "集合通信算子", "op_common": "算子公共调度与算法组件",
    "common": "所在层的公共类型与工具", "algorithm": "通信算法组织",
    "selector": "根据算子参数、拓扑和引擎选择算法", "executor": "组织模板与执行资源",
    "template": "展开算法步骤与数据原语", "wrapper": "封装数据传输与同步协议",
    "topo_match": "匹配算法所需拓扑层级", "topo_info": "生成拓扑信息与Rank映射",
    "aicpu": "AICPU执行路径", "aiv": "AIV执行路径", "ccu": "CCU执行路径与资源",
    "base_comm": "L3基础通信：原语和基础资源", "coll_communicator_mgr": "L2通信域、拓扑和资源管理",
    "primitives": "数据搬运、归约与同步原语", "resources": "基础通信资源",
    "resource": "所在模块的资源适配", "resource_mgr": "通信域资源管理",
    "resource_manager": "所在流程的资源管理", "api_c_adpt": "C接口参数检查与实现分流",
    "communicator": "通信域创建、状态和生命周期", "rank_graph": "通信Rank关系与拓扑图",
    "rank_graphs": "Rank图资源", "rank_graph_builder": "构造通信Rank图",
    "phy_topo": "物理连接拓扑", "phy_topo_builder": "构造物理拓扑",
    "rank_table_info": "解析与管理Rank表", "rank_info_detect": "检测和交换Rank信息",
    "config_mgr": "配置解析与管理", "alg_env_config": "算子环境变量配置",
    "env_config": "环境变量解析", "dfx": "日志、诊断、异常与性能观测",
    "profiling": "性能采样与上报", "loggers": "日志输出适配", "debug": "调试与诊断支持",
    "cluster_monitor": "集群状态监控", "ns_recovery": "通信异常恢复",
    "taskException": "任务异常处理", "task_exception": "任务异常处理",
    "endpoint_pairs": "端点对与通道建立/复用", "endpoints": "通信端点创建与内存注册",
    "channels": "通信通道实现与状态", "channel": "通信通道适配",
    "reged_mems": "已注册内存与内存句柄", "comm_mems": "通信域内存登记",
    "comm_engine_res": "通信引擎资源", "comm_engine_reses": "通信引擎资源集合",
    "southbound_adpt": "底层驱动/运行时接口适配", "hccp": "通信网络代理与底层网络资源",
    "local": "本地Rank侧资源", "my_rank": "本Rank建链和资源协调",
    "remote": "远端Rank资源信息", "threads": "执行Thread资源", "thread": "执行Thread适配",
    "notify": "通知资源与同步", "socket": "Socket连接和握手", "connection": "连接状态和建链",
    "transport": "传输任务和协议", "rdma_agent": "RDMA代理", "rdma_service": "RDMA服务",
    "net_adapt": "网络接口适配", "network": "网络通信支持", "rma_buffer": "远程访问缓冲区管理",
    "symmetric_memory": "对称内存管理", "mem": "内存资源", "buffer": "缓冲区管理",
    "ccu_representation": "CCU中间表示构建", "reps": "CCU指令表示节点",
    "translator": "CCU表示到执行指令的转换", "ccu_microcode": "CCU微码组织",
    "ccu_microcode_opt": "CCU微码优化", "ccu_transport": "CCU传输适配",
    "ccu_instance": "CCU实例生命周期", "ccu_device": "CCU设备资源管理",
    "ccu_comp": "CCU设备组件", "ccu_channel": "CCU通道资源",
    "ccu_kernel": "CCU Kernel构建与发射", "ccu_pfe": "CCU PFE组件",
    "ccu_context": "CCU算法上下文", "ccu_executor": "CCU算法执行器",
    "ccu_alg_template": "CCU算法模板", "alg_ccu_context": "CCU算法上下文",
    "arithmetic": "算术表达", "control": "控制流表达", "loop": "循环表达",
    "sync": "同步表达", "data": "数据操作表达", "context": "上下文状态",
    "hcomm_dlsym": "动态加载HCOMM符号，维持跨仓解耦", "algo_plugin": "算法插件加载与接口",
    "tuner": "调优插件与参数选择", "op_graph": "图模式算子与图转换",
    "interface_graph_mode": "图模式公共算子入口", "hccl_mc2": "MC2通信计算融合支持",
    "framework": "框架适配与调度", "tf_plugin": "TensorFlow算子与图适配",
    "legacy": "历史流程兼容实现", "ascend910": "Ascend910系列兼容路径",
    "ascend950": "Ascend950旧流程兼容路径", "unified_platform": "旧流程统一平台适配",
    "experimental": "试验性扩展（接口与商用构建范围需单独确认）",
    "eco_system": "生态扩展与插件示例", "nic_plugin": "网络设备插件接口和示例",
    "cluster_link_diag": "集群链路诊断扩展", "recursive_executor": "递归执行器扩展与验证",
    "include": "接口头文件（稳定性取决于所在目录）", "pkg_inc": "包间接口，非稳定公开API",
    "inc": "模块头文件", "pub_inc": "该层共享头文件", "private": "内部实现头文件",
    "test": "验证代码与测试支撑", "ut": "单元测试", "st": "系统/算法验证",
    "stub": "测试替身与运行时模拟", "depends": "测试依赖接口或替身",
    "testcase": "测试用例", "checker": "结果与语义检查器", "runner": "测试执行器",
    "semantics_check": "算法语义检查", "hccl_vm": "通信算法虚拟执行与验证工具",
    "mem_conflict_check": "内存冲突检查", "task_graph_generator": "任务图生成",
    "task_graph_generator_v3": "任务图生成V3", "frontend": "可视化工具前端",
    "frontend_v3": "可视化工具前端V3", "ui": "工具界面", "components": "界面组件",
    "examples": "可运行接口/算子示例", "example": "模块示例",
    "docs": "项目文档", "zh": "中文资料", "en": "英文资料", "api_ref": "接口参考",
    "architecture": "架构说明", "user_guide": "用户使用指南", "error_codes": "错误码说明",
    "rfcs": "设计提案", "diagrams": "架构图源文件", "figures": "文档图片",
    "figures_cn": "中文文档图片", "figures_en": "英文文档图片",
    "cmake": "构建目标、依赖与编译配置", "scripts": "构建、打包或维护脚本",
    "package": "安装包构造", "third_party": "第三方依赖适配", "external_depends": "外部组件接口",
    "python": "Python绑定与工具", "comm_mgr_python": "通信管理Python接口",
    ".agents": "仓内Agent工作规则", "skills": "仓内工作技能", "references": "技能参考规则",
    ".gitcode": "GitCode协作与CI配置", "workflows": "CI工作流", "ISSUE_TEMPLATE": "Issue模板",
    "utils": "所属模块的辅助逻辑", "util": "所属模块的辅助逻辑", "types": "类型定义",
    "team": "通信Team接口与实现", "hccl": "HCCL兼容/上层接口适配", "hcomm": "HCOMM接口适配",
    "device": "设备侧实现", "host": "Host侧实现", "device_arm": "设备ARM执行路径",
    "aicpu_ts": "AICPU TS执行路径", "cpu-cpu_ts-aicpu_ts": "CPU/CPU_TS/AICPU_TS执行路径",
    "op_host": "Host侧算子逻辑", "op_kernel_aicpu": "AICPU算子Kernel",
    "op_kernel_ccu": "CCU算子Kernel", "operator": "通信操作组织", "processor": "任务处理",
    "dispatcher": "任务分发", "task_cache": "任务缓存与重放", "unfold_cache": "展开任务缓存",
    "stream": "流与执行队列", "init": "初始化流程", "recovery": "异常恢复",
}
OPS = {
    "all_gather": "AllGather等长收集", "all_gather_v": "AllGatherV变长收集",
    "all_reduce": "AllReduce全Rank归约", "all_to_all_v": "AllToAll/AllToAllV/AllToAllVC分片交换",
    "all_to_all": "AllToAll分片交换", "reduce_scatter": "ReduceScatter归约后分发",
    "reduce_scatter_v": "ReduceScatterV变长归约分发", "reduce": "Reduce归约到Root",
    "broadcast": "Broadcast从Root广播", "scatter": "Scatter从Root分发",
    "send": "点对点发送", "recv": "点对点接收", "batch_send_recv": "批量点对点收发",
    "barrier": "Rank间屏障同步",
}
ROLES.update(OPS)
SPECIAL = {
    "hccl/src/ops/op_common/op_common.cc": "公共执行枢纽：拓扑/算法资源查询与复用、Engine分流、设备任务下发",
    "hccl/src/ops/op_common/algorithm/template/aicpu/kernel_launch.cc": "AICPU设备入口：恢复参数/资源、任务缓存、执行器编排和完成通知",
    "hccl/src/ops/op_common/algorithm/template/wrapper/alg_data_trans_wrapper.cc": "Read/Write收发握手、本地复制/归约和主从Thread同步的协议包装",
    "hccl/src/ops/all_to_all_v/algorithm/template/aicpu/ins_temp_all_to_all_v_mesh_1D.cc": "Mesh1D AllToAll模板：Peer轮次、每Channel分片、CCL中转和Read/Write选择",
    "hccl/src/ops/all_to_all_v/all_to_all_v.cc": "AllToAll系列入口：校验不同计数/位移布局、组装OpParam并进入公共调度",
    "hccl/src/ops/all_gather/all_gather.cc": "AllGather入口：检查参数、推导收集后的输出字节数、缓存/单Rank/算法分流",
    "hccl/src/ops/all_reduce/all_reduce.cc": "AllReduce入口：校验类型与归约运算组合、设置参数并进入算法调度",
    "hcomm/src/coll_communicator_mgr/resource_mgr/local/my_rank/my_rank.cc": "本Rank协调器：准备Socket/Endpoint/注册内存，创建或复用Channel并交换资源信息",
    "hcomm/src/coll_communicator_mgr/resource_mgr/local/my_rank/endpoints/endpoint_mgr.cc": "缓存通信端点，按tag与版本管理注册内存，维护销毁和解注册顺序",
    "hcomm/src/base_comm/resources/endpoint_pairs/endpoint_pair.cc": "按端点对、Engine与槽位创建/复用Channel，更新所需内存和通知资源",
    "hcomm/src/base_comm/primitives/api_c_adpt/aicpu_ts_primitives_c_adpt.cc": "AICPU_TS数据面C接口：本地/远端搬运、归约、通知、任务批提交和通信域占用保护",
    "hcomm/src/base_comm/primitives/api_c_adpt/hcomm_channel_c_adpt.cc": "基础Channel创建/销毁/查询适配，区分集合通信内部入口与公开入口",
    "hcomm/src/coll_communicator_mgr/api_c_adpt/coll_comm_res_c_adpt.cc": "通信域资源接口：拓扑/引擎资源查询和Channel申请，分流新域与兼容域",
    "hcomm/src/coll_communicator_mgr/api_c_adpt/exchange_info_c_adpt.cc": "算子一致性元信息的登记、读取消费与重置接口",
    "hcomm/src/coll_communicator_mgr/api_c_adpt/resource/channel_c_adpt.cc": "查询Channel通知数量、远端CCL缓冲区和远端注册内存",
    "hcomm/src/coll_communicator_mgr/api_c_adpt/resource/thread_c_adpt.cc": "申请Thread、转换Engine/TS类型、把用户Stream包装为Thread资源",
    "hcomm/src/coll_communicator_mgr/api_c_adpt/resource/comm_mem_c_adpt.cc": "通信域CCL缓冲区获取与注册内存信息管理接口",
    "hccl/include/hccl.h": "L1公开集合通信与点对点算子API声明",
    "hccl/include/hccl_mc2.h": "MC2自定义通信算子相关参数与接口声明",
    "hcomm/include/hcomm_primitives.h": "L3公开数据搬运、归约、同步与任务提交原语声明",
    "hcomm/include/hcomm_res.h": "L3基础通信资源获取/管理API声明",
    "hcomm/include/hcomm_channel.h": "L3 Channel创建、销毁与属性查询API声明",
    "hcomm/include/hccl/hccl_comm.h": "L2通信域初始化、销毁与属性查询API声明",
    "hcomm/include/hccl/hccl_res.h": "L2通信域资源申请与查询API声明",
    "hcomm/include/hccl/hccl_rank_graph.h": "L2 Rank图、网络层级和拓扑查询API声明",
    "hccl/src/common/param_check.cc": "校验算子输入数量、数据类型、归约操作和相关参数，统一错误报告",
    "hccl/src/common/alg_env_config.cc": "读取算法环境配置，为选择器和展开引擎提供配置值",
    "hccl/src/common/alg_parse.cc": "解析算法选择配置，把配置字符串转换为内部算法选项",
    "hccl/src/common/alg_type.cc": "维护算法类型与表示转换",
    "hccl/src/common/adapter_acl.cc": "封装ACL运行时相关调用，供算子公共流程使用",
    "hccl/src/common/inconsistent_check.cc": "准备和检查跨Rank算子参数一致性描述",
    "hccl/src/common/hcomm_dlsym/hcomm_dlsym.cc": "HCOMM动态库与符号加载管理，保持两仓运行时解耦",
    "hccl/src/common/hcomm_dlsym/hcomm_primitives_dl.cc": "动态加载并包装HCOMM数据原语，提供搬运和同步调用边界",
    "hccl/src/ops/op_common/topo_info/topo_host.cc": "构造Host侧算法所需拓扑描述，整理Rank与网络层级信息",
    "hccl/src/ops/op_common/ccu_fallback.cc": "CCU资源不可用时的协商/回退支持",
    "hccl/src/ops/op_common/exec_timeout_manager.cc": "维护执行超时配置并供协议等待使用",
    "hccl/src/ops/op_common/order_launch.cc": "维护Host/Device任务发射的保序依赖",
    "hcomm/src/base_comm/primitives/launch_context.cc": "保存线程私有发射上下文：EAGER/BATCH模式、参与Thread和默认等待配置",
}

NAME_ROLES = [
    (r"(?:ins_temp|alg_template|(?:^|_)temp(?:_|$))", "算法指令模板"),
    (r"(?:executor|alg_exec)", "算法执行器"), (r"selector", "算法选择器"),
    (r"(?:^|_)mesh(?:_|\d|$)", "Mesh拓扑"), (r"(?:^|_)ring(?:_|\d|$)", "Ring环形拓扑"),
    (r"(?:^|_)nhr(?:_|$)", "NHR算法族"), (r"(?:^|_)bruck(?:_|$)", "Bruck算法族"),
    (r"pipeline|pipe_line", "流水线组织"), (r"(?:^|_)aicpu(?:_|$)", "AICPU执行"),
    (r"(?:^|_)aiv(?:_|$)", "AIV执行"), (r"(?:^|_)ccu(?:_|$)", "CCU执行/资源"),
    (r"(?:^|_)roce(?:_|$)", "RoCE网络"), (r"(?:^|_)rdma(?:_|$)", "RDMA访问"),
    (r"(?:^|_)ub(?:_|$)|urma", "UB/URMA通信"), (r"(?:^|_)socket(?:_|$)", "Socket连接"),
    (r"(?:^|_)endpoint(?:_|$)", "通信端点"), (r"(?:^|_)channel(?:_|$)", "通信通道"),
    (r"(?:^|_)notify(?:_|$)", "通知同步"), (r"(?:^|_)stream(?:_|$)", "执行流"),
    (r"(?:^|_)thread(?:_|$)", "执行Thread"), (r"topo|rank_graph", "拓扑/Rank图"),
    (r"rank_table|ranktable", "Rank表"), (r"(?:^|_)mem(?:_|$)|memory", "内存资源"),
    (r"buffer", "缓冲区"), (r"(?:^|_)cache(?:_|$)", "缓存复用"),
    (r"(?:^|_)mgr(?:_|$)|manager", "管理与生命周期"),
    (r"(?:^|_)parse(?:_|$)|parser", "解析与转换"), (r"adapter|adpt", "接口适配"),
    (r"(?:^|_)config(?:_|$)", "配置"), (r"(?:^|_)log(?:_|$)|logger", "日志"),
    (r"profiling|profiler", "性能观测"), (r"recover|retry", "恢复/重试"),
]


def file_topic(stem):
    normalized = stem.lower().replace("alltoall", "all_to_all").replace("allgather", "all_gather").replace("allreduce", "all_reduce").replace("reducescatter", "reduce_scatter")
    topics = []
    remainder = normalized
    # Prefer compound operation names: all_reduce must not also be labelled as
    # a separate Reduce-to-root operation. Short names may describe primitives.
    primitive_labels = {"reduce": "归约操作", "scatter": "数据分发", "send": "发送操作", "recv": "接收操作", "broadcast": "广播操作", "barrier": "屏障同步"}
    for key in sorted(OPS, key=len, reverse=True):
        if key in remainder:
            topics.append(primitive_labels.get(key, OPS[key]))
            remainder = remainder.replace(key, "")
    topics += [v for pattern, v in NAME_ROLES if re.search(pattern, normalized)]
    return "、".join(dict.fromkeys(topics)) + f"（{stem}）" if topics else stem


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT)


def module_roles(path):
    parts = PurePosixPath(path).parts[1:]
    found = [ROLES[p] for p in parts if p in ROLES]
    # Preserve a module-specific leaf name when its role has no curated mapping.
    if parts and parts[-1] not in ROLES:
        found.append(f"{parts[-1]}子模块（按上级职责定位）")
    return " → ".join(dict.fromkeys(found)) or "项目根目录与工程入口"


def symbols(text, suffix):
    """Extract evidence, never silently treat every call as a defined API."""
    found = []
    lines = text.splitlines()
    patterns = [
        (r"^\s*(?:class|struct|enum\s+class|enum)\s+([A-Za-z_]\w*)", "类型"),
        (r"^\s*(?:TEST|TEST_F|TEST_P)\s*\(\s*([^,]+)\s*,\s*([^\)]+)\)", "测试"),
        (r"^\s*(?:async\s+)?def\s+([A-Za-z_]\w*)\s*\(", "函数"),
        (r"^\s*(?:add_library|add_executable|add_subdirectory|target_sources)\s*\(\s*([^\s)]+)", "构建"),
    ]
    for i, line in enumerate(lines):
        for pattern, kind in patterns:
            match = re.search(pattern, line)
            if match:
                name = ".".join(x.strip() for x in match.groups())
                found.append({"name": name, "line": i + 1, "kind": kind})
        if suffix in {".cc", ".cpp", ".c", ".h", ".hpp", ".aicpu"}:
            # Global declarations / definitions; exclude indented call sites.
            match = re.match(r"^(?:(?:extern\s+\"C\"|static|inline|virtual)\s+)*"
                             r"(?:[A-Za-z_][\w:<>,*&]*\s+)+"
                             r"([A-Za-z_][\w:~]*)\s*\(", line)
            if match and match[1] not in {"if", "while", "switch", "for", "return"}:
                found.append({"name": match[1], "line": i + 1, "kind": "接口/实现"})
    unique = {}
    for item in found:
        unique.setdefault(item["name"], item)
    return list(unique.values())


def file_entry(path):
    local = ROOT / path
    suffix = local.suffix.lower()
    raw = local.read_bytes()
    binary = b"\0" in raw[:8192] or suffix in {".png", ".gif", ".jpg", ".jpeg", ".o", ".so", ".a", ".pdf", ".ttf", ".woff", ".zip"}
    text = "" if binary else raw.decode("utf-8", errors="replace")
    evidence = symbols(text, suffix)
    folder = str(PurePosixPath(path).parent)
    role = module_roles(folder)
    stem = local.stem
    name = local.name
    topic = file_topic(stem)
    engineering = {
        ".clang-format": "配置源码格式化规则，包括缩进、行宽和指针排版",
        ".gitattributes": "配置Git文件属性和Git LFS等文件处理规则",
        ".gitignore": "定义构建产物、缓存和本地文件的Git忽略规则",
        ".pre-commit-config.yaml": "配置提交前代码格式与合规检查钩子",
        "OAT.xml": "配置OAT开源合规检查规则与例外项",
        "classify_rule.yaml": "配置文件分类与打包归属规则",
        "blacklist.txt": "保存工程检查/打包相关排除清单；具体使用位置请追踪调用脚本",
        "Third_Party_Open_Source_Software_List.yaml": "登记第三方开源组件及其许可信息",
        "Third_Party_Open_Source_Software_Notice": "随附第三方开源组件版权与许可通知",
        "build.sh": "工程构建入口：解析打包、测试和构建选项，再组织CMake及安装包流程",
    }
    if name in engineering:
        desc, kind = engineering[name], "工程配置" if name != "build.sh" else "脚本"
    elif path in SPECIAL:
        desc, kind = SPECIAL[path], "核心实现" if suffix not in {".h", ".hpp"} else "接口"
    elif name == "CMakeLists.txt" or suffix == ".cmake":
        desc, kind = f"配置此目录的编译目标、源文件或子目录依赖；归属：{role}", "构建"
    elif suffix in {".cc", ".cpp", ".c", ".aicpu"}:
        if "/stub/" in path or "stub" in stem:
            desc, kind = f"{topic}的测试替身/依赖模拟实现；用于隔离运行时或底层接口", "测试支撑"
        elif "/test/" in path:
            desc, kind = f"验证或支撑{topic}相关行为；归属：{role}", "测试"
        elif "/examples/" in path or "/example/" in path:
            desc, kind = f"演示{topic}相关接口或算子流程；归属：{role}", "示例"
        else:
            desc, kind = f"实现{topic}相关逻辑；归属：{role}", "实现"
    elif suffix in {".h", ".hpp", ".cuh", ".inl"}:
        desc, kind = f"声明/内联实现{topic}相关类型与接口；归属：{role}", "头文件"
    elif suffix == ".md":
        title = re.search(r"^#\s+(.+)", text, re.M)
        desc, kind = (f"说明「{title[1].strip()}」" if title else f"{topic}的文档或规则") + f"；归属：{role}", "文档"
    elif suffix in {".sh", ".py", ".bat", ".ps1"}:
        desc, kind = f"执行{topic}相关构建/测试/维护自动化；归属：{role}", "脚本"
    elif suffix in {".svg", ".png", ".gif", ".jpg", ".jpeg", ".excalidraw", ".drawio"}:
        desc, kind = f"{topic}的图示、图片或绘图源文件；归属：{role}", "图示"
    elif suffix == ".map":
        desc, kind = f"配置{topic}相关动态库符号导出范围；以文件中的符号表为准", "符号表"
    elif suffix in {".json", ".yaml", ".yml", ".xml", ".ini", ".toml", ".cfg"}:
        desc, kind = f"{topic}的配置、数据或模板；归属：{role}", "配置/数据"
    elif suffix in {".js", ".ts", ".vue", ".html", ".css", ".scss"}:
        desc, kind = f"{topic}相关工具界面/逻辑；归属：{role}", "工具前端"
    elif suffix == ".o":
        desc, kind = "上游快照随附的目标文件；具体符号需用目标文件工具读取", "上游二进制"
    elif name.startswith("LICENSE") or name.startswith("COPYING"):
        desc, kind = "项目或依赖的许可条款", "许可证"
    elif name == "AGENTS.md":
        desc, kind = "定义本仓Agent工作规则、架构约束和构建/检视入口", "规则"
    elif name.startswith(".git"):
        desc, kind = f"Git协作配置：{name}", "工程配置"
    elif suffix == ".txt":
        desc, kind = f"{topic}相关文本清单/规则/数据；归属：{role}", "文本"
    else:
        desc, kind = f"{name}随附工程文件；归属：{role}，具体内容从源码链接核查", "工程资料"
    if binary:
        hint = "二进制或图片条目；目录索引不解析其内部逻辑。"
    elif evidence:
        hint = "先读下面列出的内容定位，再查看同名头文件/实现及调用者；符号仅为代表性抽取。"
    else:
        hint = "此文件没有抽取到代表性类型/接口；摘要依据目录、文件类型或文档标题，请打开原文件核查细节。"
    return {"path": path, "dir": folder, "name": name, "kind": kind, "role": role,
            "description": desc, "symbols": evidence, "hint": hint,
            "annotated": "// [中文导读] " in text, "bytes": len(raw)}


def catalogue(repo):
    paths = sorted(x for x in git("ls-files", "-z", "--", repo).decode().split("\0") if x)
    entries = [file_entry(p) for p in paths]
    dirs = {repo}
    for p in paths:
        parent = PurePosixPath(p).parent
        while str(parent) != ".":
            dirs.add(str(parent))
            parent = parent.parent
    counts = collections.Counter()
    direct = collections.Counter(e["dir"] for e in entries)
    for entry in entries:
        for d in PurePosixPath(entry["path"]).parents:
            if str(d) != ".":
                counts[str(d)] += 1
    directory_entries = [{"path": d, "parent": str(PurePosixPath(d).parent),
                          "role": module_roles(d), "files": counts[d], "direct": direct[d]}
                         for d in sorted(dirs)]
    lock = json.loads((ROOT / "sources.lock.json").read_text())
    return {"repo": repo, "upstream": lock[repo]["commit"], "files": entries, "dirs": directory_entries}


def inline_markdown(text):
    escaped = html.escape(text)

    def link(match):
        target = html.unescape(match[2])
        if not target.startswith(("https://", "http://", "#")):
            if target.startswith("catalog/"):
                target = target[len("catalog/"):]
            else:
                target = "https://github.com/zstar1003/hccl-hcomm-annotated/blob/main/" + quote(posixpath.normpath("docs/" + target), safe="/#")
        return f'<a href="{html.escape(target, quote=True)}">{match[1]}</a>'

    escaped = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", link, escaped)
    escaped = re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)
    return re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", escaped)


def render_markdown(text):
    """Render the deliberately small Markdown vocabulary used by our guides."""
    output, paragraph, code, table, items = [], [], None, [], []

    def flush():
        if paragraph:
            output.append("<p>" + inline_markdown(" ".join(paragraph)) + "</p>")
            paragraph.clear()
        if table:
            rows = [row for row in table if not re.match(r"^\|[\s:|\-]+\|$", row)]
            body = []
            for i, row in enumerate(rows):
                tag = "th" if i == 0 else "td"
                body.append("<tr>" + "".join(f"<{tag}>" + inline_markdown(cell.strip()) + f"</{tag}>" for cell in row.strip("|").split("|")) + "</tr>")
            output.append('<div class="table-wrap"><table>' + "".join(body) + "</table></div>")
            table.clear()
        if items:
            output.append("<ol>" + "".join("<li>" + inline_markdown(item) + "</li>" for item in items) + "</ol>")
            items.clear()

    for line in text.splitlines():
        if line.startswith("```"):
            flush()
            if code is None:
                code = []
            else:
                output.append("<pre><code>" + html.escape("\n".join(code)) + "</code></pre>")
                code = None
        elif code is not None:
            code.append(line)
        elif not line.strip():
            flush()
        elif line.startswith("#"):
            flush()
            level = min(len(line) - len(line.lstrip("#")), 6)
            output.append(f"<h{level}>" + inline_markdown(line.lstrip("# ")) + f"</h{level}>")
        elif line.startswith("|"):
            table.append(line)
        elif re.match(r"^\d+\. ", line):
            items.append(re.sub(r"^\d+\. ", "", line))
        else:
            paragraph.append(line)
    flush()
    return "\n".join(output)


def render(repo, data):
    intro = (ROOT / "docs" / f"{repo.upper()}_DIRECTORY_GUIDE.zh-CN.md").read_text()
    # Keep the narrative readable in plain GitHub Markdown and in the offline
    # catalogue. The catalogue never executes source text or external requests.
    guide = render_markdown(intro)
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    template = (ROOT / "scripts" / "directory-guide.template.html").read_text()
    return template.replace("@@TITLE@@", repo.upper()).replace("@@GUIDE@@", guide).replace("@@DATA@@", payload)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify generated HTML without modifying it")
    args = parser.parse_args()
    for repo in ("hccl", "hcomm"):
        data = catalogue(repo)
        output = OUTPUT / f"{repo.upper()}_DIRECTORY_INDEX.zh-CN.html"
        result = render(repo, data).encode()
        if args.check:
            if not output.exists() or output.read_bytes() != result:
                raise SystemExit(f"STALE: {output.relative_to(ROOT)}; run this script to rebuild")
        else:
            OUTPUT.mkdir(parents=True, exist_ok=True)
            output.write_bytes(result)
        print(f"{'CHECK' if args.check else 'BUILD'} {repo}: {len(data['files'])} files, {len(data['dirs'])} directories, {len(result)} bytes")


if __name__ == "__main__":
    main()
