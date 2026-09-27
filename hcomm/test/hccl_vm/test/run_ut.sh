#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# HCCL_VM UT 测试执行脚本
# 流程: 第一步-CMake配置+编译 -> 第二步-生成可执行文件 -> 第三步-执行用例 -> 第四步-生成lcov覆盖率报告
# 用法:
#   ./run_ut.sh                          全量编译+执行所有测试
#   ./run_ut.sh --cov                    全量编译+执行+生成gcov/lcov覆盖率HTML报告
#   ./run_ut.sh <目录>                   编译+执行指定目录下所有测试(递归)
#   ./run_ut.sh <二进制名>                 编译+执行指定二进制测试
#   ./run_ut.sh <测试文件名>               编译+执行指定测试文件对应的二进制
#   ./run_ut.sh <文件> <测试用例名>         编译+执行指定文件中的单个测试用例
#   ./run_ut.sh -l, --list                列出所有可用测试
#   ./run_ut.sh -h, --help                显示帮助信息

# ==================== 配置 ====================
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CODE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="$CODE_DIR/build"
BIN_DIR="$BUILD_DIR/output/bin"
TEST_DIR="$SCRIPT_DIR"
LOG_DIR="$CODE_DIR/ut_logs"
COV_DIR="$CODE_DIR/coverage_report"
ENABLE_COVERAGE=0
CMAKE_BUILD_TYPE="Debug"
MAKE_JOBS=8

# ==================== 测试执行优化配置 ====================
# 默认超时(秒)
DEFAULT_TIMEOUT=60
# 并行执行的最大并发数 (1=串行, >1=并行)
PARALLEL_JOBS=1

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

check_env() {
    local env_error=0

    if [ -z "${HCOMM_CODE_HOME:-}" ]; then
        echo -e "${RED}✗ HCOMM_CODE_HOME 未设置${NC}"
        echo -e "${YELLOW}  请在控制台执行: export HCOMM_CODE_HOME=<你的hcomm路径>${NC}"
        env_error=1
    else
        echo -e "${GREEN}✓ HCOMM_CODE_HOME=$HCOMM_CODE_HOME${NC}"
    fi

    if [ -z "${HCCL_CODE_HOME:-}" ]; then
        echo -e "${RED}✗ HCCL_CODE_HOME 未设置${NC}"
        echo -e "${YELLOW}  请在控制台执行: export HCCL_CODE_HOME=<你的hccl路径>${NC}"
        env_error=1
    else
        echo -e "${GREEN}✓ HCCL_CODE_HOME=$HCCL_CODE_HOME${NC}"
    fi

    if [ -n "${ASCEND_HOME_PATH:-}" ] || [ -n "${ASCEND_TOOLKIT_HOME:-}" ]; then
        echo -e "${GREEN}✓ CANN 环境已加载 (ASCEND_HOME_PATH=${ASCEND_HOME_PATH:-未设置}, ASCEND_TOOLKIT_HOME=${ASCEND_TOOLKIT_HOME:-未设置})${NC}"
    elif [ -z "${CANN_SET_ENV:-}" ]; then
        echo -e "${RED}✗ CANN_SET_ENV 未设置，且 CANN 环境未加载${NC}"
        echo -e "${YELLOW}  请在控制台执行以下任一方式:${NC}"
        echo -e "${YELLOW}    方式1: export CANN_SET_ENV=/path/to/set_env.sh${NC}"
        echo -e "${YELLOW}    方式2: source /path/to/set_env.sh${NC}"
        env_error=1
    elif [ ! -f "$CANN_SET_ENV" ]; then
        echo -e "${RED}✗ CANN_SET_ENV 指向的文件不存在: $CANN_SET_ENV${NC}"
        echo -e "${YELLOW}  请检查路径是否正确${NC}"
        env_error=1
    else
        source "$CANN_SET_ENV" 2>/dev/null
        if [ $? -ne 0 ]; then
            echo -e "${RED}✗ source $CANN_SET_ENV 执行失败${NC}"
            env_error=1
        else
            echo -e "${GREEN}✓ 已加载 CANN 环境: $CANN_SET_ENV${NC}"
        fi
    fi

    if [ $env_error -eq 1 ]; then
        echo -e "${RED}✗ 必要环境变量未完整设置，请设置后重新执行脚本${NC}"
        echo -e "${YELLOW}  示例:${NC}"
        echo -e "${YELLOW}    export HCOMM_CODE_HOME=/path/to/hcomm${NC}"
        echo -e "${YELLOW}    export HCCL_CODE_HOME=/path/to/hccl${NC}"
        echo -e "${YELLOW}    export CANN_SET_ENV=/path/to/set_env.sh"
        echo -e "${YELLOW}    然后重新运行: $0 $*${NC}"
        return 1
    fi
    return 0
}

check_env "$@" || exit 1

# ==================== 日志系统 ====================
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
mkdir -p "$LOG_DIR/$TIMESTAMP"
BUILD_LOG="$LOG_DIR/$TIMESTAMP/build.log"
RUN_LOG="$LOG_DIR/$TIMESTAMP/run.log"
SUMMARY_LOG="$LOG_DIR/$TIMESTAMP/summary.log"

log_build() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$BUILD_LOG"; }
log_run()   { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$RUN_LOG"; }
log_summary(){ echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$SUMMARY_LOG"; }

# ==================== 第一步: CMake 配置 + 编译 ====================
step_cmake() {
    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN}  第一步: CMake 配置${NC}"
    echo -e "${CYAN}========================================${NC}"
    log_build "========== CMake 配置开始 =========="
    log_build "源码目录: $CODE_DIR | 构建目录: $BUILD_DIR | 类型: $CMAKE_BUILD_TYPE"

    if [ -d "$BUILD_DIR" ]; then
        echo -e "${YELLOW}清空构建目录: $BUILD_DIR${NC}"
        # rm -rf "${BUILD_DIR:?}"/*
    else
        mkdir -p "$BUILD_DIR"
    fi
    local start=$(date +%s)
    cd "$BUILD_DIR"
set -o pipefail
    local cmake_extra_flags=""
    if [ "$ENABLE_COVERAGE" -eq 1 ]; then
        cmake_extra_flags="-DENABLE_COVERAGE=ON"
        echo -e "${YELLOW}覆盖率模式: 已启用gcov/lcov插桩 (--coverage)${NC}"
        log_build "覆盖率模式: ENABLE_COVERAGE=ON"
        rm -f "$BUILD_DIR/CMakeCache.txt"
    fi
    if cmake .. -DCMAKE_BUILD_TYPE=$CMAKE_BUILD_TYPE -DBUILD_TESTS=ON $cmake_extra_flags 2>&1 | tee -a "$BUILD_LOG"; then
        local dur=$(($(date +%s) - start))
        echo -e "${GREEN}✓ CMake 配置成功 (耗时: ${dur}s)${NC}"
        log_build "CMake 配置成功, 耗时: ${dur}s"
        log_summary "CMake 配置: 成功, 耗时: ${dur}s"
        return 0
    else
        local dur=$(($(date +%s) - start))
        echo -e "${RED}✗ CMake 配置失败 (耗时: ${dur}s)${NC}"
        log_build "CMake 配置失败, 耗时: ${dur}s"
        log_summary "CMake 配置: 失败, 耗时: ${dur}s"
        return 1
    fi
}

step_make() {
    local targets=("$@")
    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN}  第一步: 编译目标${NC}"
    echo -e "${CYAN}========================================${NC}"

    if [ ${#targets[@]} -eq 0 ]; then
        echo -e "${YELLOW}编译所有测试...${NC}"
        log_build "========== 编译所有目标 =========="
    else
        echo -e "${YELLOW}编译目标: ${targets[*]}${NC}"
        log_build "========== 编译目标: ${targets[*]} =========="
    fi

    local start=$(date +%s)
    cd "$BUILD_DIR"
    local make_cmd="make -j${MAKE_JOBS} ${targets[*]}"
    log_build "执行: $make_cmd"

    set -o pipefail
    if $make_cmd 2>&1 | tee -a "$BUILD_LOG"; then
        local dur=$(($(date +%s) - start))
        echo -e "${GREEN}✓ 编译成功 (耗时: ${dur}s)${NC}"
        log_build "编译成功, 耗时: ${dur}s"
        log_summary "编译: 成功, 耗时: ${dur}s, 目标: ${targets[*]}"
        return 0
    else
        local dur=$(($(date +%s) - start))
        echo -e "${RED}✗ 编译失败 (耗时: ${dur}s)${NC}"
        log_build "编译失败, 耗时: ${dur}s"
        log_summary "编译: 失败, 耗时: ${dur}s, 目标: ${targets[*]}"

        echo ""
        echo -e "${RED}========================================${NC}"
        echo -e "${RED}  编译错误详情 (从构建日志提取)${NC}"
        echo -e "${RED}========================================${NC}"
        local error_lines=$(grep -E '(error:|错误|fatal error|no matching function|no rule to make target|undefined reference|cannot find|No such file)' "$BUILD_LOG" | grep -v '^--' | tail -50)
        if [ -n "$error_lines" ]; then
            echo "$error_lines" | while IFS= read -r line; do
                echo -e "${RED}  $line${NC}"
            done
        else
            echo -e "${YELLOW}  未匹配到标准错误模式，显示最后30行日志:${NC}"
            tail -30 "$BUILD_LOG" | while IFS= read -r line; do
                echo -e "${YELLOW}  $line${NC}"
            done
        fi
        echo -e "${RED}========================================${NC}"
        echo -e "${YELLOW}完整构建日志: $BUILD_LOG${NC}"
        echo ""

        return 1
    fi
}

do_build() {
    local targets=("$@")
    step_cmake || return 1
    step_make "${targets[@]}" || return 1
    step_show_binaries
}

# ==================== 第二步: 生成可执行文件 ====================
step_show_binaries() {
    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN}  第二步: 生成可执行文件${NC}"
    echo -e "${CYAN}========================================${NC}"

    local count=$(find "$BIN_DIR" -name "test_*" -executable -type f 2>/dev/null | wc -l)
    echo -e "可执行测试文件目录: ${YELLOW}$BIN_DIR${NC}"
    echo -e "可执行测试文件数量: ${GREEN}$count${NC}"

    log_build "========== 生成可执行文件 =========="
    log_build "输出目录: $BIN_DIR"
    log_build "可执行文件数量: $count"

    for bin in "$BIN_DIR"/test_*; do
        if [ -x "$bin" ]; then
            local name=$(basename "$bin")
            local size=$(ls -lh "$bin" | awk '{print $5}')
            local mtime=$(stat -c '%y' "$bin" 2>/dev/null | cut -d'.' -f1)
            echo "  $name  (${size}, ${mtime})"
            log_build "  生成: $name (${size}, ${mtime})"
        fi
    done

    log_summary "生成可执行文件: $count 个"
    echo ""
}

# ==================== 第三步: 执行用例 + 显示结果 ====================
get_binary_name() {
    local test_file="$1"
    local base=$(basename "$test_file" .cc)
    local dir=$(dirname "$test_file" | sed 's|.*/test/||')

    local patterns=("test_${base%_test}" "test_${base%_test}_test" "$base")

    for p in "${patterns[@]}"; do
        [ -x "$BIN_DIR/$p" ] && echo "$p" && return 0
    done

    local filepart=$(echo "$base" | sed 's/_test$//')
    for bin in "$BIN_DIR"/test_*; do
        echo "$(basename "$bin")" | grep -q "$filepart" && echo "$(basename "$bin")" && return 0
    done

    # Clean build directory may not have produced the binary yet. Fall back to
    # the repository convention so directory mode can still build only the
    # targets that belong to that directory.
    echo "test_${base%_test}"
    return 0
}

get_dir_binaries() {
    local dir="$1"
    find "$TEST_DIR/$dir" -name "*_test.cc" 2>/dev/null | sort | while read f; do
        local bin_name=$(get_binary_name "$f" 2>/dev/null)
        if [ -n "$bin_name" ]; then
            echo "$bin_name"
        fi
    done | sort -u
}

run_one_binary() {
    local bin_name="$1"
    local gtest_filter="$2"
    local timeout_sec="$DEFAULT_TIMEOUT"

    if [ ! -x "$BIN_DIR/$bin_name" ]; then
        echo -e "${RED}✗ 二进制不存在: $bin_name${NC}"
        log_run "二进制不存在: $bin_name"
        echo "STATUS_LINE:NOBIN:0:0"
        return 1
    fi

    local filter_info=""
    [ -n "$gtest_filter" ] && filter_info=" 过滤: $gtest_filter"

    echo -e "${BLUE}--- 执行: $bin_name${filter_info} ---${NC}"
    log_run "执行: $bin_name${filter_info}"

    rm -f /dev/shm/Hccl* 2>/dev/null

    local start=$(date +%s)
    local output
    local ret=0
    output=$(cd "$BIN_DIR" && timeout $timeout_sec ./$bin_name $([ -n "$gtest_filter" ] && echo "--gtest_filter=$gtest_filter") 2>&1) || ret=$?
    local dur=$(($(date +%s) - start))

    echo "$output"
    echo "$output" >> "$RUN_LOG"

    local passed=0 failed=0
    local p_line=$(echo "$output" | grep -oP '\[  PASSED  \] \K\d+' | tail -1)
    local f_line=$(echo "$output" | grep -oP '\d+(?= FAILED TEST)' | tail -1)
    [ -n "$p_line" ] && passed=$p_line
    [ -n "$f_line" ] && failed=$f_line

    local status=""
    if [ $ret -eq 124 ]; then
        echo -e "${YELLOW}⚠ 超时 (>${timeout_sec}s)${NC}"
        log_run "超时: $bin_name, 耗时: ${dur}s"
        log_summary "  $bin_name: 超时"
        status="TIMEOUT:$passed:$failed"
    elif echo "$output" | grep -q "dumped core"; then
        echo -e "${RED}✗ 崩溃${NC}"
        log_run "崩溃: $bin_name"
        log_summary "  $bin_name: 崩溃"
        status="CRASH:$passed:$failed"
    elif [ $ret -ne 0 ]; then
        echo -e "${RED}✗ 失败 (exit: $ret)${NC}"
        log_run "失败: $bin_name, exit: $ret, PASSED: $passed, FAILED: $failed, 耗时: ${dur}s"
        log_summary "  $bin_name: 失败, PASSED: $passed, FAILED: $failed, 耗时: ${dur}s"
        status="FAIL:$passed:$failed"
    else
        echo -e "${GREEN}✓ 通过 (PASSED: $passed, 耗时: ${dur}s)${NC}"
        log_run "成功: $bin_name, PASSED: $passed, 耗时: ${dur}s"
        log_summary "  $bin_name: 成功, PASSED: $passed, FAILED: 0, 耗时: ${dur}s"
        status="PASS:$passed:$failed"
    fi

    echo "STATUS_LINE,$status"
}

step_run_single() {
    local target="$1"
    local test_case="$2"
    local bin_name=""

    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN}  第三步: 执行测试${NC}"
    echo -e "${CYAN}========================================${NC}"

    if [[ "$target" == test_* ]]; then
        bin_name="$target"
    elif [[ "$target" == *_test.cc ]]; then
        bin_name=$(get_binary_name "$target")
        [ -z "$bin_name" ] && echo -e "${RED}错误: 找不到 $target 对应的二进制${NC}" && return 1
    else
        local found=$(find "$TEST_DIR" -name "$target" 2>/dev/null | head -1)
        if [ -n "$found" ]; then
            bin_name=$(get_binary_name "$found")
        else
            bin_name="test_$target"
        fi
    fi

    log_run "========== 执行单个测试: $bin_name =========="
    log_summary "========== 执行单个测试: $bin_name =========="

    local result=$(run_one_binary "$bin_name" "$test_case")
    local status_line=$(echo "$result" | grep "^STATUS_LINE," | tail -1 | sed 's/STATUS_LINE,//')
    echo ""
    echo -e "${CYAN}========== 执行结果 ==========${NC}"
    local stype=$(echo "$status_line" | cut -d: -f1)
    local sp=$(echo "$status_line" | cut -d: -f2)
    local sf=$(echo "$status_line" | cut -d: -f3)
    echo -e "状态: ${GREEN}$stype${NC}, PASSED: ${GREEN}${sp}${NC}, FAILED: ${RED}${sf}${NC}"
}

step_run_directory() {
    local dir="$1"

    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN}  第三步: 执行目录测试: $dir${NC}"
    echo -e "${CYAN}========================================${NC}"

    log_run "========== 执行目录测试: $dir =========="
    log_summary "========== 执行目录测试: $dir =========="

    local total_pass=0 total_fail=0 total_crash=0 total_timeout=0 count=0

    local bins=$(get_dir_binaries "$dir")
    if [ -z "$bins" ]; then
        echo -e "${YELLOW}目录下无可执行测试${NC}"
        log_run "目录下无可执行测试: $dir"
        return 0
    fi

    for bin_name in $bins; do
        count=$((count + 1))
        echo -e "\n[${count}] $bin_name"
        local result=$(run_one_binary "$bin_name" "")
        local status_line=$(echo "$result" | grep "^STATUS_LINE," | tail -1 | sed 's/STATUS_LINE,//')
        local stype=$(echo "$status_line" | cut -d: -f1)
        local sp=$(echo "$status_line" | cut -d: -f2)
        local sf=$(echo "$status_line" | cut -d: -f3)
        [ -n "$sp" ] && total_pass=$((total_pass + sp))
        [ -n "$sf" ] && total_fail=$((total_fail + sf))
        [ "$stype" = "CRASH" ] && total_crash=$((total_crash + 1))
        [ "$stype" = "TIMEOUT" ] && total_timeout=$((total_timeout + 1))
        echo ""
    done

    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN}  目录测试结果汇总: $dir${NC}"
    echo -e "${CYAN}========================================${NC}"
    echo -e "  执行数量:   $count"
    echo -e "  ${GREEN}PASSED:  $total_pass${NC}"
    echo -e "  ${RED}FAILED:  $total_fail${NC}"
    echo -e "  ${RED}CRASHED: $total_crash${NC}"
    echo -e "  ${YELLOW}TIMEOUT: $total_timeout${NC}"

    log_summary "目录 $dir 汇总: 执行=$count, PASSED=$total_pass, FAILED=$total_fail, CRASHED=$total_crash, TIMEOUT=$total_timeout"
}

step_run_all() {
    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN}  第三步: 执行所有测试${NC}"
    echo -e "${CYAN}========================================${NC}"

    if [ "$ENABLE_COVERAGE" -eq 1 ] && command -v lcov &>/dev/null; then
        echo -e "${YELLOW}清零覆盖率计数器...${NC}"
        lcov --zerocounters --directory "$BUILD_DIR" --quiet 2>&1 | tee -a "$RUN_LOG"
    fi

    log_run "========== 执行所有测试 =========="
    log_summary "========== 执行所有测试 =========="

    # 收集所有需要执行的测试
    local test_bins=()
    for bin in "$BIN_DIR"/test_*; do
        [ ! -x "$bin" ] && continue
        local bin_name=$(basename "$bin")
        test_bins+=("$bin_name")
    done

    local total_pass=0 total_fail=0 total_crash=0 total_timeout=0 count=0

    local effective_parallel_jobs="$PARALLEL_JOBS"
    if [ "$ENABLE_COVERAGE" -eq 1 ] && [ "$effective_parallel_jobs" -gt 1 ]; then
        effective_parallel_jobs=1
        echo -e "${YELLOW}覆盖率模式: 禁用并行执行，避免gcov数据竞争${NC}"
    fi

    # 并行执行模式
    if [ "$effective_parallel_jobs" -gt 1 ] && [ "${#test_bins[@]}" -gt 0 ]; then
        echo -e "${YELLOW}并行执行模式: ${effective_parallel_jobs}并发${NC}"
        local tmpdir=$(mktemp -d)
        local idx=0
        for bin_name in "${test_bins[@]}"; do
            ((idx++))
            count=$((count + 1))
            # 后台执行，结果写入临时文件
            (
                local result=$(run_one_binary "$bin_name" "")
                echo "$result" > "$tmpdir/result_$idx.txt"
            ) &
            # 控制并发数
            if (( idx % effective_parallel_jobs == 0 )); then
                wait
            fi
        done
        wait

        # 收集结果
        for i in $(seq 1 $idx); do
            if [ -f "$tmpdir/result_$i.txt" ]; then
                local result=$(cat "$tmpdir/result_$i.txt")
                local status_line=$(echo "$result" | grep "^STATUS_LINE," | tail -1 | sed 's/STATUS_LINE,//')
                local stype=$(echo "$status_line" | cut -d: -f1)
                local sp=$(echo "$status_line" | cut -d: -f2)
                local sf=$(echo "$status_line" | cut -d: -f3)
                [ -n "$sp" ] && total_pass=$((total_pass + sp))
                [ -n "$sf" ] && total_fail=$((total_fail + sf))
                [ "$stype" = "CRASH" ] && total_crash=$((total_crash + 1))
                [ "$stype" = "TIMEOUT" ] && total_timeout=$((total_timeout + 1))
            fi
        done
        rm -rf "$tmpdir"
    else
        # 串行执行模式
        for bin_name in "${test_bins[@]}"; do
            count=$((count + 1))
            echo -e "\n[${count}] $bin_name"
            local result=$(run_one_binary "$bin_name" "")
            local status_line=$(echo "$result" | grep "^STATUS_LINE," | tail -1 | sed 's/STATUS_LINE,//')
            local stype=$(echo "$status_line" | cut -d: -f1)
            local sp=$(echo "$status_line" | cut -d: -f2)
            local sf=$(echo "$status_line" | cut -d: -f3)
            [ -n "$sp" ] && total_pass=$((total_pass + sp))
            [ -n "$sf" ] && total_fail=$((total_fail + sf))
            [ "$stype" = "CRASH" ] && total_crash=$((total_crash + 1))
            [ "$stype" = "TIMEOUT" ] && total_timeout=$((total_timeout + 1))
            echo ""
        done
    fi

    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN}  全部测试结果汇总${NC}"
    echo -e "${CYAN}========================================${NC}"
    echo -e "  执行数量:   $count"
    echo -e "  ${GREEN}PASSED:  $total_pass${NC}"
    echo -e "  ${RED}FAILED:  $total_fail${NC}"
    echo -e "  ${RED}CRASHED: $total_crash${NC}"
    echo -e "  ${YELLOW}TIMEOUT: $total_timeout${NC}"

    log_summary "全部测试汇总: 执行=$count, PASSED=$total_pass, FAILED=$total_fail, CRASHED=$total_crash, TIMEOUT=$total_timeout"

    G_TEST_COUNT=$count
    G_TEST_PASS=$total_pass
    G_TEST_FAIL=$total_fail
    G_TEST_CRASH=$total_crash
    G_TEST_TIMEOUT=$total_timeout
}

# ==================== 第四步: 生成覆盖率报告 ====================
step_gen_coverage() {
    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN}  第四步: 生成gcov/lcov覆盖率报告${NC}"
    echo -e "${CYAN}========================================${NC}"

    if ! command -v lcov &>/dev/null; then
        echo -e "${RED}✗ lcov 未安装, 请执行: sudo apt install lcov${NC}"
        log_summary "覆盖率报告: lcov未安装, 跳过"
        return 1
    fi

    if ! command -v genhtml &>/dev/null; then
        echo -e "${RED}✗ genhtml 未安装, 请执行: sudo apt install lcov${NC}"
        log_summary "覆盖率报告: genhtml未安装, 跳过"
        return 1
    fi

    mkdir -p "$COV_DIR"
    local cov_info="$COV_DIR/coverage.info"
    local cov_info_filtered="$COV_DIR/coverage_filtered.info"

    echo -e "${YELLOW}1. 收集原始覆盖率数据...${NC}"
    lcov --capture --directory "$BUILD_DIR" --base-directory "$CODE_DIR" --output-file "$cov_info" --rc geninfo_unexecuted_blocks=1 --rc lcov_merge_coverage=1 --ignore-errors mismatch,gcov 2>&1 | tee -a "$RUN_LOG"
    if [ ! -f "$cov_info" ] || [ ! -s "$cov_info" ]; then
        echo -e "${RED}✗ 收集覆盖率数据失败${NC}"
        log_summary "覆盖率报告: 收集数据失败"
        return 1
    fi

    local has_source=$(grep -c "^SF:" "$cov_info" 2>/dev/null || echo "0")
    if [ "$has_source" -eq 0 ]; then
        echo -e "${RED}✗ 覆盖率数据中无源文件记录 (可能所有文件被过滤)${NC}"
        log_summary "覆盖率报告: 无源文件记录"
        return 1
    fi

    echo -e "${YELLOW}2. 过滤第三方/系统头文件...${NC}"
    lcov --remove "$cov_info" \
        '*/third_party/*' \
        '*/googletest/*' \
        '*/googlemock/*' \
        '*/build/*' \
        '*/usr/include/*' \
        '*/checker/build/*' \
        '*/checker/header/external/spdlog/*' \
        '*/checker/header/external/nlohmann/*' \
        '*/checker/header/internal/spdlog/*' \
        '*/test/*' \
        --output-file "$cov_info_filtered" 2>&1 | tee -a "$RUN_LOG"

    echo -e "${YELLOW}3. 生成HTML报告...${NC}"
    genhtml "$cov_info_filtered" --output-directory "$COV_DIR/html" --title "HCCL_VM UT Coverage" --legend --num-spaces 4 2>&1 | tee -a "$RUN_LOG"

    local lcov_summary=$(lcov --summary "$cov_info_filtered" 2>&1)
    G_LINE_RATE=$(echo "$lcov_summary" | grep "lines" | grep -oP '\d+\.\d+%' | head -1)
    G_FUNC_RATE=$(echo "$lcov_summary" | grep "functions" | grep -oP '\d+\.\d+%' | head -1)
    G_BRANCH_RATE=$(echo "$lcov_summary" | grep "branches" | grep -oP '\d+\.\d+%' | head -1)

    log_summary "gcov/lcov覆盖率: lines=$G_LINE_RATE, functions=$G_FUNC_RATE, branches=$G_BRANCH_RATE"
    log_summary "覆盖率HTML报告: $COV_DIR/html/index.html"
}

# ==================== 最终汇总: 测试结果 + 真实覆盖率合并 ====================
show_final_summary() {
    echo ""
    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN}  最终汇总: 测试结果 + gcov/lcov 真实覆盖率${NC}"
    echo -e "${CYAN}========================================${NC}"
    echo -e "  执行数量:   ${G_TEST_COUNT:-0}"
    echo -e "  ${GREEN}PASSED:    ${G_TEST_PASS:-0}${NC}"
    echo -e "  ${RED}FAILED:    ${G_TEST_FAIL:-0}${NC}"
    echo -e "  ${RED}CRASHED:   ${G_TEST_CRASH:-0}${NC}"
    echo -e "  ${YELLOW}TIMEOUT:   ${G_TEST_TIMEOUT:-0}${NC}"
    echo ""
    echo -e "  ${GREEN}行覆盖率:   ${G_LINE_RATE:-N/A}${NC}"
    echo -e "  ${GREEN}函数覆盖率: ${G_FUNC_RATE:-N/A}${NC}"
    echo ""
}
list_all_tests() {
    echo -e "${BLUE}=== 可执行的测试 ===${NC}"
    echo ""
    printf "%-55s %-8s %s\n" "测试文件" "用例数" "二进制名"
    printf "%-55s %-8s %s\n" "--------" "------" "--------"

    find "$TEST_DIR" -name "*_test.cc" | sort | while read f; do
        local relpath=$(echo "$f" | sed "s|$TEST_DIR/||")
        local count=$(grep -c "^TEST_F" "$f" 2>/dev/null || echo 0)
        local bin_name=$(get_binary_name "$f" 2>/dev/null)
        if [ -n "$bin_name" ] && [ -x "$BIN_DIR/$bin_name" ]; then
            echo -e "${GREEN}✓${NC} $relpath ($count) -> $bin_name"
        else
            echo -e "${RED}✗${NC} $relpath ($count) -> ${YELLOW}未编译${NC}"
        fi
    done
    echo ""
    echo -e "可执行测试数: $(find "$BIN_DIR" -name "test_*" -executable -type f 2>/dev/null | wc -l)"
}

# ==================== 帮助 ====================
print_help() {
    echo "HCCL_VM UT 测试执行脚本"
    echo ""
    echo "流程: 第一步-CMake配置+编译 -> 第二步-生成可执行文件 -> 第三步-执行用例+显示结果"
    echo ""
    echo "用法:"
    echo "  $0                                    全量编译+执行所有测试"
    echo "  $0 --cov                              全量编译+执行+生成gcov/lcov覆盖率HTML报告"
    echo "  $0 <目录> [--cov]                     编译+执行指定目录下所有测试(递归)"
    echo "  $0 <二进制名> [--cov]                  编译+执行指定二进制测试"
    echo "  $0 <测试文件名> [--cov]                编译+执行指定测试文件对应的二进制"
    echo "  $0 <文件> <测试用例名> [--cov]          编译+执行指定文件中的单个测试用例"
    echo "  $0 -l, --list                         列出所有可用测试"
    echo "  $0 -h, --help                         显示帮助信息"
    echo ""
    echo "选项:"
    echo "  --cov                                 启用覆盖率插桩(--coverage), 执行后生成lcov HTML报告"
    echo ""
    echo "示例:"
    echo "  $0                                    # 全量编译+执行所有测试"
    echo "  $0 --cov                              # 全量编译+执行+生成覆盖率报告"
    echo "  $0 plugin/checker                     # 编译+执行 plugin/checker 目录下所有测试(递归)"
    echo "  $0 test_checker --cov                 # 编译+执行 test_checker 并生成覆盖率"
    echo "  $0 checker_test.cc CheckerTest.Basic  # 编译+执行单个测试用例"
    echo "  $0 -l                                 # 列出所有测试"
    echo ""
    echo "日志目录: $LOG_DIR/<时间戳>/"
    echo "  build.log    - 编译日志"
    echo "  run.log      - 执行日志"
    echo "  summary.log  - 汇总日志"
    echo "  覆盖率报告   - $COV_DIR/html/index.html (需--cov)"
}

# ==================== 判断参数类型 ====================
is_directory() {
    local arg="$1"
    [ -d "$TEST_DIR/$arg" ] && return 0
    return 1
}

is_test_file() {
    local arg="$1"
    [[ "$arg" == *_test.cc ]] && return 0
    return 1
}

is_binary_name() {
    local arg="$1"
    [[ "$arg" == test_* ]] && return 0
    return 1
}

# ==================== 主逻辑 ====================
main() {
    local args=()
    for arg in "$@"; do
        if [ "$arg" = "--cov" ]; then
            ENABLE_COVERAGE=1
        else
            args+=("$arg")
        fi
    done

    echo -e "${CYAN}HCCL_VM UT 测试执行脚本${NC}"
    echo -e "日志目录: ${YELLOW}$LOG_DIR/$TIMESTAMP/${NC}"
    if [ "$ENABLE_COVERAGE" -eq 1 ]; then
        echo -e "${YELLOW}覆盖率模式: 已启用 (--cov)${NC}"
    fi
    echo ""

    log_summary "========== 脚本启动 =========="
    log_summary "参数: $*"
    [ "$ENABLE_COVERAGE" -eq 1 ] && log_summary "覆盖率模式: ENABLE_COVERAGE=ON"

    case "${args[0]:-}" in
        -h|--help)
            print_help
            ;;
        -l|--list)
            list_all_tests
            ;;
        "")
            log_summary "模式: 全量编译+执行"
            do_build || exit 1
            step_run_all
            if [ "$ENABLE_COVERAGE" -eq 1 ]; then
                step_gen_coverage
                show_final_summary
            fi
            ;;
        *)
            if is_directory "${args[0]}"; then
                local dir="${args[0]}"
                log_summary "模式: 目录编译+执行, 目录: $dir"
                local bins=$(get_dir_binaries "$dir")
                if [ -z "$bins" ]; then
                    echo -e "${YELLOW}目录下无可执行测试，尝试编译所有${NC}"
                    do_build || exit 1
                else
                    do_build $bins || exit 1
                fi
                step_run_directory "$dir"
            elif is_test_file "${args[0]}"; then
                log_summary "模式: 文件编译+执行, 文件: ${args[0]}"
                local bin_name=$(get_binary_name "${args[0]}")
                if [ -z "$bin_name" ]; then
                    echo -e "${RED}错误: 找不到 ${args[0]} 对应的二进制${NC}"
                    exit 1
                fi
                do_build "$bin_name" || exit 1
                step_run_single "${args[0]}" "${args[1]:-}"
            elif is_binary_name "${args[0]}"; then
                log_summary "模式: 二进制编译+执行, 目标: ${args[0]}"
                do_build "${args[0]}" || exit 1
                step_run_single "${args[0]}" "${args[1]:-}"
            else
                local found=$(find "$TEST_DIR" -name "${args[0]}" -o -name "${args[0]}_test.cc" 2>/dev/null | head -1)
                if [ -n "$found" ]; then
                    local bin_name=$(get_binary_name "$found")
                    if [ -n "$bin_name" ]; then
                        log_summary "模式: 文件编译+执行, 文件: $found"
                        do_build "$bin_name" || exit 1
                        step_run_single "$found" "${args[1]:-}"
                    else
                        echo -e "${RED}错误: 找不到 ${args[0]} 对应的二进制${NC}"
                        exit 1
                    fi
                else
                    local bin_name="test_${args[0]}"
                    log_summary "模式: 二进制编译+执行, 目标: $bin_name"
                    do_build "$bin_name" || exit 1
                    step_run_single "$bin_name" "${args[1]:-}"
                fi
            fi
            if [ "$ENABLE_COVERAGE" -eq 1 ]; then
                step_gen_coverage
                show_final_summary
            fi
            ;;
    esac

    log_summary "========== 脚本结束 =========="

    # 清空构建目录，释放磁盘空间
    if [ -d "$BUILD_DIR" ]; then
        echo -e "${YELLOW}清空构建目录: $BUILD_DIR${NC}"
        rm -rf "${BUILD_DIR:?}"/*
        log_summary "构建目录已清空"
    fi

    echo ""
    echo -e "日志目录: ${YELLOW}$LOG_DIR/$TIMESTAMP/${NC}"
    echo -e "  build.log    - 编译日志"
    echo -e "  run.log      - 执行日志"
    echo -e "  summary.log  - 汇总日志"
    if [ "$ENABLE_COVERAGE" -eq 1 ]; then
        echo -e "  覆盖率报告:  ${YELLOW}$COV_DIR/html/index.html${NC}"
    fi
}

main "$@"
