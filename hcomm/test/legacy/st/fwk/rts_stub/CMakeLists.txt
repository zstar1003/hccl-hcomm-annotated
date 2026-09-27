# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

target_include_directories(hccl_st_fwk PUBLIC
        ${TOP_DIR}/runtime/include/external
        ${TOP_DIR}/runtime/pkg_inc/runtime
        ${TOP_DIR}/runtime/pkg_inc/
        ${TOP_DIR}/runtime/pkg_inc/profiling
        ${TOP_DIR}/ace/npuruntime/inc
        ${TOP_DIR}/ace/npuruntime/acl/inc/external/
        ${TOP_DIR}/ace/npuruntime/acl/inc/external/acl
        ${TOP_DIR}/ace/comop/inc/
        ${TOP_DIR}/ace/comop/inc/hccl/
        ${TOP_DIR}/ace/comop/inc/external/
        ${TOP_DIR}/ace/comop/inc/external/hccl
        ${TOP_DIR}/inc/external
        ${TOP_DIR}/runtime/src/dfx/adump/inc/adump
        ${TOP_DIR}/metadef/inc/external
        ${TOP_DIR}/metadef/pkg_inc
        ${TOP_DIR}/abl/mmpa/inc/mmpa
        ${TOP_DIR}/abl/mmpa/inc/mmpa/sub_inc
)

target_sources(hccl_st_fwk
        PRIVATE
        rts_stub.cc
        FakeStreamMgr.cc
        mmpa_stub_st.cc
)