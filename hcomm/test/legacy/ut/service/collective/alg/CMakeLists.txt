# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# 添加 hccl_next 算法的 cpp 文件
FILE(GLOB_RECURSE hccl_next_alg_files
    ${HCCL_V2_CODE_ROOT}/service/collective/*.cc
    ${HCCL_V2_CODE_ROOT}/service/collective/*.cpp
)
FILE(GLOB_RECURSE hccl_next_alg_files_exclude
    ${HCCL_V2_CODE_ROOT}/service/collective/alg/coll_alg_factory/utils/host/*.cc
)
LIST (REMOVE_ITEM hccl_next_alg_files ${hccl_next_alg_files_exclude})

# 添加 hccl_orion 算法的 cpp
FILE (GLOB_RECURSE hccl_orion_alg_files
    ${HCCL_V2_CODE_ROOT}/service/collective/alg/coll_alg_factory/alg_ccu_context/*.cc
    ${HCCL_V2_CODE_ROOT}/service/collective/alg/coll_alg_factory/alg_template/ccu_alg_template/*.cc
    ${HCCL_V2_CODE_ROOT}/service/collective/alg/coll_alg_factory/alg_executor/prim_alg_executor/*.cc
    ${HCCL_V2_CODE_ROOT}/service/collective/alg/coll_alg_factory/alg_template/prim_alg_template/*.cc
    ${HCCL_V2_CODE_ROOT}/service/collective/alg/coll_alg_factory/alg_template/aiv_alg_template/*.cc
    ${HCCL_V2_CODE_ROOT}/service/collective/primitive/*.cc
)

target_sources(hccl_v2_utest
        PUBLIC
        ut_ccu_alg_2d.cc
        ${hccl_next_alg_files}
        ${hccl_orion_alg_files}
)
