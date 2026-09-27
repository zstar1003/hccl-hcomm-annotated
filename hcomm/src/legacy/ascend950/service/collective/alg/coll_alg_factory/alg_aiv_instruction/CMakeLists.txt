# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set_source_files_properties(
    aiv_communication_v2.h PROPERTIES LANGUAGE CXX
)
if(BUILD_OPEN_PROJECT)
    set(CCE_DIR ${ASCEND_CANN_PACKAGE_PATH}/tools/ccec_compiler/bin)
    
    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/hccl_aiv_op_910_95.o
        COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_communication_v2.h 
                -o ${CMAKE_CURRENT_BINARY_DIR}/aiv_communication_910_95_tmp.o 
                --cce-aicore-arch=dav-c310-vec
                --cce-aicore-only 
                -mllvm -cce-aicore-function-stack-size=0x8000 
                -mllvm -cce-aicore-stack-size=0x8000
                -D__CCE_AICORE__=310 -D__DAV_C310__ -DASCENDC_DUMP
                "-I$<JOIN:$<TARGET_PROPERTY:asc_kernel_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
                "-I$<JOIN:$<TARGET_PROPERTY:kernel_tiling_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
                -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
                -mllvm -cce-aicore-record-overflow=false 
                -mllvm -cce-aicore-addr-transform 
                -mllvm --cce-aicore-jump-expand=true 
                -std=c++17 -fstack-protector-all -DONE_CORE_DUMP_SIZE=1024
        
        COMMAND ${CCE_DIR}/ld.lld -m aicorelinux -Ttext=0 
                ${CMAKE_CURRENT_BINARY_DIR}/aiv_communication_910_95_tmp.o 
                -static -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_aiv_op_910_95.o
        
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/aiv_communication_v2.h
        COMMENT "Compiling and Linking AIV kernel"
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMAND_EXPAND_LISTS VERBATIM
        DEPENDS kernel_tiling_headers
    )

    add_custom_target(hccl_aiv_op_910_95 ALL
        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/hccl_aiv_op_910_95.o)

    add_custom_target(hccl_ar_superkernel_mesh_1d_oneshot_int8 ALL
        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/hccl_ar_superkernel_mesh_1d_oneshot_int8.o)
    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/hccl_ar_superkernel_mesh_1d_oneshot_int8.o
            COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_ar_mesh_1d_oneshot_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_ar_superkernel_mesh_1d_oneshot_int8.o
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0
            -DASCENDC_DUMP --cce-aicore-arch=dav-c310-vec
            --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-dcci-insert-for-scalar=false
            -mllvm -cce-aicore-stack-size=0x8000 -D__CCE_AICORE__=310 -D__DAV_C310__
            -DHCCL_DTYPE_INT8 -Dsk_allreduce_mesh_1d_oneshot=sk_allreduce_mesh_1d_oneshot_int8
            "-I$<JOIN:$<TARGET_PROPERTY:asc_kernel_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
            "-I$<JOIN:$<TARGET_PROPERTY:kernel_tiling_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
            -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMAND_EXPAND_LISTS VERBATIM
            DEPENDS kernel_tiling_headers
    )

    add_custom_target(hccl_ar_superkernel_mesh_1d_oneshot_int16 ALL
        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/hccl_ar_superkernel_mesh_1d_oneshot_int16.o)
    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/hccl_ar_superkernel_mesh_1d_oneshot_int16.o
            COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_ar_mesh_1d_oneshot_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_ar_superkernel_mesh_1d_oneshot_int16.o
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0
            -DASCENDC_DUMP --cce-aicore-arch=dav-c310-vec
            --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-dcci-insert-for-scalar=false
            -mllvm -cce-aicore-stack-size=0x8000 -D__CCE_AICORE__=310 -D__DAV_C310__
            -DHCCL_DTYPE_INT16 -Dsk_allreduce_mesh_1d_oneshot=sk_allreduce_mesh_1d_oneshot_int16
            "-I$<JOIN:$<TARGET_PROPERTY:asc_kernel_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
            "-I$<JOIN:$<TARGET_PROPERTY:kernel_tiling_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
            -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMAND_EXPAND_LISTS VERBATIM
            DEPENDS kernel_tiling_headers
    )

    add_custom_target(hccl_ar_superkernel_mesh_1d_oneshot_int32 ALL
        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/hccl_ar_superkernel_mesh_1d_oneshot_int32.o)
    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/hccl_ar_superkernel_mesh_1d_oneshot_int32.o
            COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_ar_mesh_1d_oneshot_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_ar_superkernel_mesh_1d_oneshot_int32.o
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0
            -DASCENDC_DUMP --cce-aicore-arch=dav-c310-vec
            --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-dcci-insert-for-scalar=false
            -mllvm -cce-aicore-stack-size=0x8000 -D__CCE_AICORE__=310 -D__DAV_C310__
            -DHCCL_DTYPE_INT32 -Dsk_allreduce_mesh_1d_oneshot=sk_allreduce_mesh_1d_oneshot_int32
            "-I$<JOIN:$<TARGET_PROPERTY:asc_kernel_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
            "-I$<JOIN:$<TARGET_PROPERTY:kernel_tiling_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
            -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMAND_EXPAND_LISTS VERBATIM
            DEPENDS kernel_tiling_headers
    )

    add_custom_target(hccl_ar_superkernel_mesh_1d_oneshot_float16 ALL
        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/hccl_ar_superkernel_mesh_1d_oneshot_float16.o)
    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/hccl_ar_superkernel_mesh_1d_oneshot_float16.o
            COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_ar_mesh_1d_oneshot_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_ar_superkernel_mesh_1d_oneshot_float16.o
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0
            -DASCENDC_DUMP --cce-aicore-arch=dav-c310-vec
            --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-dcci-insert-for-scalar=false
            -mllvm -cce-aicore-stack-size=0x8000 -D__CCE_AICORE__=310 -D__DAV_C310__
            -DHCCL_DTYPE_FP16 -Dsk_allreduce_mesh_1d_oneshot=sk_allreduce_mesh_1d_oneshot_float16
            "-I$<JOIN:$<TARGET_PROPERTY:asc_kernel_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
            "-I$<JOIN:$<TARGET_PROPERTY:kernel_tiling_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
            -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMAND_EXPAND_LISTS VERBATIM
            DEPENDS kernel_tiling_headers
    )

    add_custom_target(hccl_ar_superkernel_mesh_1d_oneshot_float32 ALL
        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/hccl_ar_superkernel_mesh_1d_oneshot_float32.o)
    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/hccl_ar_superkernel_mesh_1d_oneshot_float32.o
            COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_ar_mesh_1d_oneshot_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_ar_superkernel_mesh_1d_oneshot_float32.o
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0
            -DASCENDC_DUMP --cce-aicore-arch=dav-c310-vec
            --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-dcci-insert-for-scalar=false
            -mllvm -cce-aicore-stack-size=0x8000 -D__CCE_AICORE__=310 -D__DAV_C310__
            -DHCCL_DTYPE_FP32 -Dsk_allreduce_mesh_1d_oneshot=sk_allreduce_mesh_1d_oneshot_float32
            "-I$<JOIN:$<TARGET_PROPERTY:asc_kernel_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
            "-I$<JOIN:$<TARGET_PROPERTY:kernel_tiling_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
            -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMAND_EXPAND_LISTS VERBATIM
            DEPENDS kernel_tiling_headers
    )

    add_custom_target(hccl_ar_superkernel_mesh_1d_oneshot_bfloat16 ALL
        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/hccl_ar_superkernel_mesh_1d_oneshot_bfloat16.o)
    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/hccl_ar_superkernel_mesh_1d_oneshot_bfloat16.o
            COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_ar_mesh_1d_oneshot_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_ar_superkernel_mesh_1d_oneshot_bfloat16.o
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0
            -DASCENDC_DUMP --cce-aicore-arch=dav-c310-vec
            --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-dcci-insert-for-scalar=false
            -mllvm -cce-aicore-stack-size=0x8000 -D__CCE_AICORE__=310 -D__DAV_C310__
            -DHCCL_DTYPE_BFP16 -Dsk_allreduce_mesh_1d_oneshot=sk_allreduce_mesh_1d_oneshot_bfloat16
            "-I$<JOIN:$<TARGET_PROPERTY:asc_kernel_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
            "-I$<JOIN:$<TARGET_PROPERTY:kernel_tiling_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
            -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMAND_EXPAND_LISTS VERBATIM
            DEPENDS kernel_tiling_headers
    )

    add_custom_target(hccl_ar_superkernel_mesh_1d_twoshot_int8 ALL
        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/hccl_ar_superkernel_mesh_1d_twoshot_int8.o)
    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/hccl_ar_superkernel_mesh_1d_twoshot_int8.o
            COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_ar_mesh_1d_twoshot_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_ar_superkernel_mesh_1d_twoshot_int8.o
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0
            -DASCENDC_DUMP --cce-aicore-arch=dav-c310-vec
            --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-dcci-insert-for-scalar=false
            -mllvm -cce-aicore-stack-size=0x8000 -D__CCE_AICORE__=310 -D__DAV_C310__
            -DHCCL_DTYPE_INT8 -Dsk_allreduce_mesh_1d_twoshot=sk_allreduce_mesh_1d_twoshot_int8
            "-I$<JOIN:$<TARGET_PROPERTY:asc_kernel_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
            "-I$<JOIN:$<TARGET_PROPERTY:kernel_tiling_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
            -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMAND_EXPAND_LISTS VERBATIM
            DEPENDS kernel_tiling_headers
    )

    add_custom_target(hccl_ar_superkernel_mesh_1d_twoshot_int16 ALL
        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/hccl_ar_superkernel_mesh_1d_twoshot_int16.o)
    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/hccl_ar_superkernel_mesh_1d_twoshot_int16.o
            COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_ar_mesh_1d_twoshot_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_ar_superkernel_mesh_1d_twoshot_int16.o
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0
            -DASCENDC_DUMP --cce-aicore-arch=dav-c310-vec
            --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-dcci-insert-for-scalar=false
            -mllvm -cce-aicore-stack-size=0x8000 -D__CCE_AICORE__=310 -D__DAV_C310__
            -DHCCL_DTYPE_INT16 -Dsk_allreduce_mesh_1d_twoshot=sk_allreduce_mesh_1d_twoshot_int16
            "-I$<JOIN:$<TARGET_PROPERTY:asc_kernel_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
            "-I$<JOIN:$<TARGET_PROPERTY:kernel_tiling_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
            -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMAND_EXPAND_LISTS VERBATIM
            DEPENDS kernel_tiling_headers
    )

    add_custom_target(hccl_ar_superkernel_mesh_1d_twoshot_int32 ALL
        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/hccl_ar_superkernel_mesh_1d_twoshot_int32.o)
    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/hccl_ar_superkernel_mesh_1d_twoshot_int32.o
            COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_ar_mesh_1d_twoshot_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_ar_superkernel_mesh_1d_twoshot_int32.o
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0
            -DASCENDC_DUMP --cce-aicore-arch=dav-c310-vec
            --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-dcci-insert-for-scalar=false
            -mllvm -cce-aicore-stack-size=0x8000 -D__CCE_AICORE__=310 -D__DAV_C310__
            -DHCCL_DTYPE_INT32 -Dsk_allreduce_mesh_1d_twoshot=sk_allreduce_mesh_1d_twoshot_int32
            "-I$<JOIN:$<TARGET_PROPERTY:asc_kernel_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
            "-I$<JOIN:$<TARGET_PROPERTY:kernel_tiling_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
            -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMAND_EXPAND_LISTS VERBATIM
            DEPENDS kernel_tiling_headers
    )

    add_custom_target(hccl_ar_superkernel_mesh_1d_twoshot_float16 ALL
        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/hccl_ar_superkernel_mesh_1d_twoshot_float16.o)
    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/hccl_ar_superkernel_mesh_1d_twoshot_float16.o
            COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_ar_mesh_1d_twoshot_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_ar_superkernel_mesh_1d_twoshot_float16.o
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0
            -DASCENDC_DUMP --cce-aicore-arch=dav-c310-vec
            --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-dcci-insert-for-scalar=false
            -mllvm -cce-aicore-stack-size=0x8000 -D__CCE_AICORE__=310 -D__DAV_C310__
            -DHCCL_DTYPE_FP16 -Dsk_allreduce_mesh_1d_twoshot=sk_allreduce_mesh_1d_twoshot_float16
            "-I$<JOIN:$<TARGET_PROPERTY:asc_kernel_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
            "-I$<JOIN:$<TARGET_PROPERTY:kernel_tiling_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
            -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMAND_EXPAND_LISTS VERBATIM
            DEPENDS kernel_tiling_headers
    )

    add_custom_target(hccl_ar_superkernel_mesh_1d_twoshot_float32 ALL
        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/hccl_ar_superkernel_mesh_1d_twoshot_float32.o)
    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/hccl_ar_superkernel_mesh_1d_twoshot_float32.o
            COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_ar_mesh_1d_twoshot_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_ar_superkernel_mesh_1d_twoshot_float32.o
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0
            -DASCENDC_DUMP --cce-aicore-arch=dav-c310-vec
            --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-dcci-insert-for-scalar=false
            -mllvm -cce-aicore-stack-size=0x8000 -D__CCE_AICORE__=310 -D__DAV_C310__
            -DHCCL_DTYPE_FP32 -Dsk_allreduce_mesh_1d_twoshot=sk_allreduce_mesh_1d_twoshot_float32
            "-I$<JOIN:$<TARGET_PROPERTY:asc_kernel_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
            "-I$<JOIN:$<TARGET_PROPERTY:kernel_tiling_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
            -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMAND_EXPAND_LISTS VERBATIM
            DEPENDS kernel_tiling_headers
    )

    add_custom_target(hccl_ar_superkernel_mesh_1d_twoshot_bfloat16 ALL
        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/hccl_ar_superkernel_mesh_1d_twoshot_bfloat16.o)
    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/hccl_ar_superkernel_mesh_1d_twoshot_bfloat16.o
            COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_ar_mesh_1d_twoshot_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_ar_superkernel_mesh_1d_twoshot_bfloat16.o
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0
            -DASCENDC_DUMP --cce-aicore-arch=dav-c310-vec
            --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-dcci-insert-for-scalar=false
            -mllvm -cce-aicore-stack-size=0x8000 -D__CCE_AICORE__=310 -D__DAV_C310__
            -DHCCL_DTYPE_BFP16 -Dsk_allreduce_mesh_1d_twoshot=sk_allreduce_mesh_1d_twoshot_bfloat16
            "-I$<JOIN:$<TARGET_PROPERTY:asc_kernel_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
            "-I$<JOIN:$<TARGET_PROPERTY:kernel_tiling_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
            -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMAND_EXPAND_LISTS VERBATIM
            DEPENDS kernel_tiling_headers
    )

    add_custom_target(hccl_rs_superkernel_mesh_1d_int8 ALL
        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/hccl_rs_superkernel_mesh_1d_int8.o)
    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/hccl_rs_superkernel_mesh_1d_int8.o
            COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_rs_mesh_1d_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_rs_superkernel_mesh_1d_int8.o
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0
            -DASCENDC_DUMP --cce-aicore-arch=dav-c310-vec
            --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-dcci-insert-for-scalar=false
            -mllvm -cce-aicore-stack-size=0x8000 -D__CCE_AICORE__=310 -D__DAV_C310__
            -DHCCL_DTYPE_INT8 -Dsk_reducescatter_mesh_1d=sk_reducescatter_mesh_1d_int8
            "-I$<JOIN:$<TARGET_PROPERTY:asc_kernel_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
            "-I$<JOIN:$<TARGET_PROPERTY:kernel_tiling_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
            -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMAND_EXPAND_LISTS VERBATIM
            DEPENDS kernel_tiling_headers
    )

    add_custom_target(hccl_rs_superkernel_mesh_1d_int16 ALL
        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/hccl_rs_superkernel_mesh_1d_int16.o)
    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/hccl_rs_superkernel_mesh_1d_int16.o
            COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_rs_mesh_1d_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_rs_superkernel_mesh_1d_int16.o
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0
            -DASCENDC_DUMP --cce-aicore-arch=dav-c310-vec
            --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-dcci-insert-for-scalar=false
            -mllvm -cce-aicore-stack-size=0x8000 -D__CCE_AICORE__=310 -D__DAV_C310__
            -DHCCL_DTYPE_INT16 -Dsk_reducescatter_mesh_1d=sk_reducescatter_mesh_1d_int16
            "-I$<JOIN:$<TARGET_PROPERTY:asc_kernel_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
            "-I$<JOIN:$<TARGET_PROPERTY:kernel_tiling_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
            -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMAND_EXPAND_LISTS VERBATIM
            DEPENDS kernel_tiling_headers
    )

    add_custom_target(hccl_rs_superkernel_mesh_1d_int32 ALL
        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/hccl_rs_superkernel_mesh_1d_int32.o)
    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/hccl_rs_superkernel_mesh_1d_int32.o
            COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_rs_mesh_1d_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_rs_superkernel_mesh_1d_int32.o
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0
            -DASCENDC_DUMP --cce-aicore-arch=dav-c310-vec
            --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-dcci-insert-for-scalar=false
            -mllvm -cce-aicore-stack-size=0x8000 -D__CCE_AICORE__=310 -D__DAV_C310__
            -DHCCL_DTYPE_INT32 -Dsk_reducescatter_mesh_1d=sk_reducescatter_mesh_1d_int32
            "-I$<JOIN:$<TARGET_PROPERTY:asc_kernel_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
            "-I$<JOIN:$<TARGET_PROPERTY:kernel_tiling_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
            -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMAND_EXPAND_LISTS VERBATIM
            DEPENDS kernel_tiling_headers
    )

    add_custom_target(hccl_rs_superkernel_mesh_1d_float16 ALL
        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/hccl_rs_superkernel_mesh_1d_float16.o)
    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/hccl_rs_superkernel_mesh_1d_float16.o
            COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_rs_mesh_1d_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_rs_superkernel_mesh_1d_float16.o
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0
            -DASCENDC_DUMP --cce-aicore-arch=dav-c310-vec
            --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-dcci-insert-for-scalar=false
            -mllvm -cce-aicore-stack-size=0x8000 -D__CCE_AICORE__=310 -D__DAV_C310__
            -DHCCL_DTYPE_FP16 -Dsk_reducescatter_mesh_1d=sk_reducescatter_mesh_1d_float16
            "-I$<JOIN:$<TARGET_PROPERTY:asc_kernel_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
            "-I$<JOIN:$<TARGET_PROPERTY:kernel_tiling_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
            -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMAND_EXPAND_LISTS VERBATIM
            DEPENDS kernel_tiling_headers
    )

    add_custom_target(hccl_rs_superkernel_mesh_1d_float32 ALL
        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/hccl_rs_superkernel_mesh_1d_float32.o)
    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/hccl_rs_superkernel_mesh_1d_float32.o
            COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_rs_mesh_1d_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_rs_superkernel_mesh_1d_float32.o
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0
            -DASCENDC_DUMP --cce-aicore-arch=dav-c310-vec
            --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-dcci-insert-for-scalar=false
            -mllvm -cce-aicore-stack-size=0x8000 -D__CCE_AICORE__=310 -D__DAV_C310__
            -DHCCL_DTYPE_FP32 -Dsk_reducescatter_mesh_1d=sk_reducescatter_mesh_1d_float32
            "-I$<JOIN:$<TARGET_PROPERTY:asc_kernel_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
            "-I$<JOIN:$<TARGET_PROPERTY:kernel_tiling_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
            -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMAND_EXPAND_LISTS VERBATIM
            DEPENDS kernel_tiling_headers
    )

    add_custom_target(hccl_rs_superkernel_mesh_1d_bfloat16 ALL
        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/hccl_rs_superkernel_mesh_1d_bfloat16.o)
    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/hccl_rs_superkernel_mesh_1d_bfloat16.o
            COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_rs_mesh_1d_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_rs_superkernel_mesh_1d_bfloat16.o
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0
            -DASCENDC_DUMP --cce-aicore-arch=dav-c310-vec
            --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-dcci-insert-for-scalar=false
            -mllvm -cce-aicore-stack-size=0x8000 -D__CCE_AICORE__=310 -D__DAV_C310__
            -DHCCL_DTYPE_BFP16 -Dsk_reducescatter_mesh_1d=sk_reducescatter_mesh_1d_bfloat16
            "-I$<JOIN:$<TARGET_PROPERTY:asc_kernel_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
            "-I$<JOIN:$<TARGET_PROPERTY:kernel_tiling_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
            -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMAND_EXPAND_LISTS VERBATIM
            DEPENDS kernel_tiling_headers
    )

    add_custom_target(hccl_ag_superkernel_mesh_1d_int8 ALL
        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_mesh_1d_int8.o)
    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_mesh_1d_int8.o
            COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_ag_mesh_1d_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_mesh_1d_int8.o
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0
            -DASCENDC_DUMP --cce-aicore-arch=dav-c310-vec
            --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-dcci-insert-for-scalar=false
            -mllvm -cce-aicore-stack-size=0x8000 -D__CCE_AICORE__=310 -D__DAV_C310__
            -DHCCL_DTYPE_INT8 -Dsk_allgather_mesh_1d=sk_allgather_mesh_1d_int8
            "-I$<JOIN:$<TARGET_PROPERTY:asc_kernel_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
            "-I$<JOIN:$<TARGET_PROPERTY:kernel_tiling_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
            -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMAND_EXPAND_LISTS VERBATIM
            DEPENDS kernel_tiling_headers
    )

    add_custom_target(hccl_ag_superkernel_mesh_1d_int16 ALL
        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_mesh_1d_int16.o)
    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_mesh_1d_int16.o
            COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_ag_mesh_1d_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_mesh_1d_int16.o
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0
            -DASCENDC_DUMP --cce-aicore-arch=dav-c310-vec
            --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-dcci-insert-for-scalar=false
            -mllvm -cce-aicore-stack-size=0x8000 -D__CCE_AICORE__=310 -D__DAV_C310__
            -DHCCL_DTYPE_INT16 -Dsk_allgather_mesh_1d=sk_allgather_mesh_1d_int16
            "-I$<JOIN:$<TARGET_PROPERTY:asc_kernel_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
            "-I$<JOIN:$<TARGET_PROPERTY:kernel_tiling_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
            -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMAND_EXPAND_LISTS VERBATIM
            DEPENDS kernel_tiling_headers
    )

    add_custom_target(hccl_ag_superkernel_mesh_1d_int32 ALL
        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_mesh_1d_int32.o)
    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_mesh_1d_int32.o
            COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_ag_mesh_1d_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_mesh_1d_int32.o
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0
            -DASCENDC_DUMP --cce-aicore-arch=dav-c310-vec
            --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-dcci-insert-for-scalar=false
            -mllvm -cce-aicore-stack-size=0x8000 -D__CCE_AICORE__=310 -D__DAV_C310__
            -DHCCL_DTYPE_INT32 -Dsk_allgather_mesh_1d=sk_allgather_mesh_1d_int32
            "-I$<JOIN:$<TARGET_PROPERTY:asc_kernel_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
            "-I$<JOIN:$<TARGET_PROPERTY:kernel_tiling_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
            -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMAND_EXPAND_LISTS VERBATIM
            DEPENDS kernel_tiling_headers
    )

    add_custom_target(hccl_ag_superkernel_mesh_1d_float16 ALL
        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_mesh_1d_float16.o)
    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_mesh_1d_float16.o
            COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_ag_mesh_1d_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_mesh_1d_float16.o
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0
            -DASCENDC_DUMP --cce-aicore-arch=dav-c310-vec
            --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-dcci-insert-for-scalar=false
            -mllvm -cce-aicore-stack-size=0x8000 -D__CCE_AICORE__=310 -D__DAV_C310__
          -DHCCL_DTYPE_FP16 -Dsk_allgather_mesh_1d=sk_allgather_mesh_1d_float16
            "-I$<JOIN:$<TARGET_PROPERTY:asc_kernel_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
            "-I$<JOIN:$<TARGET_PROPERTY:kernel_tiling_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
            -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMAND_EXPAND_LISTS VERBATIM
            DEPENDS kernel_tiling_headers
    )

    add_custom_target(hccl_ag_superkernel_mesh_1d_float32 ALL
        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_mesh_1d_float32.o)
    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_mesh_1d_float32.o
            COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_ag_mesh_1d_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_mesh_1d_float32.o
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0
            -DASCENDC_DUMP --cce-aicore-arch=dav-c310-vec
            --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-dcci-insert-for-scalar=false
            -mllvm -cce-aicore-stack-size=0x8000 -D__CCE_AICORE__=310 -D__DAV_C310__
            -DHCCL_DTYPE_FP32 -Dsk_allgather_mesh_1d=sk_allgather_mesh_1d_float32
            "-I$<JOIN:$<TARGET_PROPERTY:asc_kernel_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
            "-I$<JOIN:$<TARGET_PROPERTY:kernel_tiling_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
            -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMAND_EXPAND_LISTS VERBATIM
            DEPENDS kernel_tiling_headers
    )

    add_custom_target(hccl_ag_superkernel_mesh_1d_bfloat16 ALL
        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_mesh_1d_bfloat16.o)
    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_mesh_1d_bfloat16.o
            COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_ag_mesh_1d_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_mesh_1d_bfloat16.o
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0
            -DASCENDC_DUMP --cce-aicore-arch=dav-c310-vec
            --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-dcci-insert-for-scalar=false
            -mllvm -cce-aicore-stack-size=0x8000 -D__CCE_AICORE__=310 -D__DAV_C310__
            -DHCCL_DTYPE_BFP16 -Dsk_allgather_mesh_1d=sk_allgather_mesh_1d_bfloat16
            "-I$<JOIN:$<TARGET_PROPERTY:asc_kernel_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
            "-I$<JOIN:$<TARGET_PROPERTY:kernel_tiling_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
            -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMAND_EXPAND_LISTS VERBATIM
            DEPENDS kernel_tiling_headers
    )

    add_custom_target(hccl_ag_superkernel_mesh_1d_uint8 ALL
        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_mesh_1d_uint8.o)
    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_mesh_1d_uint8.o
            COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_ag_mesh_1d_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_mesh_1d_uint8.o
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0
            -DASCENDC_DUMP --cce-aicore-arch=dav-c310-vec
            --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-dcci-insert-for-scalar=false
            -mllvm -cce-aicore-stack-size=0x8000 -D__CCE_AICORE__=310 -D__DAV_C310__
            -DHCCL_DTYPE_UINT8 -Dsk_allgather_mesh_1d=sk_allgather_mesh_1d_uint8
            "-I$<JOIN:$<TARGET_PROPERTY:asc_kernel_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
            "-I$<JOIN:$<TARGET_PROPERTY:kernel_tiling_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
            -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMAND_EXPAND_LISTS VERBATIM
            DEPENDS kernel_tiling_headers
    )

    add_custom_target(hccl_ag_superkernel_mesh_1d_uint16 ALL
        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_mesh_1d_uint16.o)
    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_mesh_1d_uint16.o
            COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_ag_mesh_1d_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_mesh_1d_uint16.o
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0
            -DASCENDC_DUMP --cce-aicore-arch=dav-c310-vec
            --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-dcci-insert-for-scalar=false
            -mllvm -cce-aicore-stack-size=0x8000 -D__CCE_AICORE__=310 -D__DAV_C310__
            -DHCCL_DTYPE_UINT16 -Dsk_allgather_mesh_1d=sk_allgather_mesh_1d_uint16
            "-I$<JOIN:$<TARGET_PROPERTY:asc_kernel_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
            "-I$<JOIN:$<TARGET_PROPERTY:kernel_tiling_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
            -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMAND_EXPAND_LISTS VERBATIM
            DEPENDS kernel_tiling_headers
    )

    add_custom_target(hccl_ag_superkernel_mesh_1d_uint32 ALL
        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_mesh_1d_uint32.o)
    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_mesh_1d_uint32.o
            COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_ag_mesh_1d_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_mesh_1d_uint32.o
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0
            -DASCENDC_DUMP --cce-aicore-arch=dav-c310-vec
            --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-dcci-insert-for-scalar=false
            -mllvm -cce-aicore-stack-size=0x8000 -D__CCE_AICORE__=310 -D__DAV_C310__
            -DHCCL_DTYPE_UINT32 -Dsk_allgather_mesh_1d=sk_allgather_mesh_1d_uint32
            "-I$<JOIN:$<TARGET_PROPERTY:asc_kernel_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
            "-I$<JOIN:$<TARGET_PROPERTY:kernel_tiling_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
            -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMAND_EXPAND_LISTS VERBATIM
            DEPENDS kernel_tiling_headers
    )

    add_custom_target(hccl_ag_superkernel_mesh_1d_int64 ALL
        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_mesh_1d_int64.o)
    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_mesh_1d_int64.o
            COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_ag_mesh_1d_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_mesh_1d_int64.o
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0
            -DASCENDC_DUMP --cce-aicore-arch=dav-c310-vec
            --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-dcci-insert-for-scalar=false
            -mllvm -cce-aicore-stack-size=0x8000 -D__CCE_AICORE__=310 -D__DAV_C310__
            -DHCCL_DTYPE_INT64 -Dsk_allgather_mesh_1d=sk_allgather_mesh_1d_int64
            "-I$<JOIN:$<TARGET_PROPERTY:asc_kernel_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
            "-I$<JOIN:$<TARGET_PROPERTY:kernel_tiling_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
            -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMAND_EXPAND_LISTS VERBATIM
            DEPENDS kernel_tiling_headers
    )

    add_custom_target(hccl_ag_superkernel_mesh_1d_uint64 ALL
        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_mesh_1d_uint64.o)
    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_mesh_1d_uint64.o
            COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_ag_mesh_1d_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_mesh_1d_uint64.o
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0
            -DASCENDC_DUMP --cce-aicore-arch=dav-c310-vec
            --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-dcci-insert-for-scalar=false
            -mllvm -cce-aicore-stack-size=0x8000 -D__CCE_AICORE__=310 -D__DAV_C310__
            -DHCCL_DTYPE_UINT64 -Dsk_allgather_mesh_1d=sk_allgather_mesh_1d_uint64
            "-I$<JOIN:$<TARGET_PROPERTY:asc_kernel_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
            "-I$<JOIN:$<TARGET_PROPERTY:kernel_tiling_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
            -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMAND_EXPAND_LISTS VERBATIM
            DEPENDS kernel_tiling_headers
    )

    add_custom_target(hccl_a2a_superkernel_mesh_1d_int8 ALL
        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/hccl_a2a_superkernel_mesh_1d_int8.o)
    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/hccl_a2a_superkernel_mesh_1d_int8.o
            COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_a2a_mesh_1d_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_a2a_superkernel_mesh_1d_int8.o
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0
            -DASCENDC_DUMP --cce-aicore-arch=dav-c310-vec
            --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-dcci-insert-for-scalar=false
            -mllvm -cce-aicore-stack-size=0x8000 -D__CCE_AICORE__=310 -D__DAV_C310__
            -DHCCL_DTYPE_INT8 -Dsk_alltoall_mesh_1d=sk_alltoall_mesh_1d_int8
            "-I$<JOIN:$<TARGET_PROPERTY:asc_kernel_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
            "-I$<JOIN:$<TARGET_PROPERTY:kernel_tiling_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
            -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMAND_EXPAND_LISTS VERBATIM
            DEPENDS kernel_tiling_headers
    )

    add_custom_target(hccl_a2a_superkernel_mesh_1d_int16 ALL
        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/hccl_a2a_superkernel_mesh_1d_int16.o)
    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/hccl_a2a_superkernel_mesh_1d_int16.o
            COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_a2a_mesh_1d_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_a2a_superkernel_mesh_1d_int16.o
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0
            -DASCENDC_DUMP --cce-aicore-arch=dav-c310-vec
            --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-dcci-insert-for-scalar=false
            -mllvm -cce-aicore-stack-size=0x8000 -D__CCE_AICORE__=310 -D__DAV_C310__
            -DHCCL_DTYPE_INT16 -Dsk_alltoall_mesh_1d=sk_alltoall_mesh_1d_int16
            "-I$<JOIN:$<TARGET_PROPERTY:asc_kernel_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
            "-I$<JOIN:$<TARGET_PROPERTY:kernel_tiling_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
            -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMAND_EXPAND_LISTS VERBATIM
            DEPENDS kernel_tiling_headers
    )

    add_custom_target(hccl_a2a_superkernel_mesh_1d_int32 ALL
        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/hccl_a2a_superkernel_mesh_1d_int32.o)
    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/hccl_a2a_superkernel_mesh_1d_int32.o
            COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_a2a_mesh_1d_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_a2a_superkernel_mesh_1d_int32.o
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0
            -DASCENDC_DUMP --cce-aicore-arch=dav-c310-vec
            --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-dcci-insert-for-scalar=false
            -mllvm -cce-aicore-stack-size=0x8000 -D__CCE_AICORE__=310 -D__DAV_C310__
            -DHCCL_DTYPE_INT32 -Dsk_alltoall_mesh_1d=sk_alltoall_mesh_1d_int32
            "-I$<JOIN:$<TARGET_PROPERTY:asc_kernel_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
            "-I$<JOIN:$<TARGET_PROPERTY:kernel_tiling_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
            -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMAND_EXPAND_LISTS VERBATIM
            DEPENDS kernel_tiling_headers
    )

    add_custom_target(hccl_a2a_superkernel_mesh_1d_float16 ALL
        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/hccl_a2a_superkernel_mesh_1d_float16.o)
    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/hccl_a2a_superkernel_mesh_1d_float16.o
            COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_a2a_mesh_1d_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_a2a_superkernel_mesh_1d_float16.o
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0
            -DASCENDC_DUMP --cce-aicore-arch=dav-c310-vec
            --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-dcci-insert-for-scalar=false
            -mllvm -cce-aicore-stack-size=0x8000 -D__CCE_AICORE__=310 -D__DAV_C310__
            -DHCCL_DTYPE_FP16 -Dsk_alltoall_mesh_1d=sk_alltoall_mesh_1d_float16
            "-I$<JOIN:$<TARGET_PROPERTY:asc_kernel_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
            "-I$<JOIN:$<TARGET_PROPERTY:kernel_tiling_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
            -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMAND_EXPAND_LISTS VERBATIM
            DEPENDS kernel_tiling_headers
    )

    add_custom_target(hccl_a2a_superkernel_mesh_1d_float32 ALL
        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/hccl_a2a_superkernel_mesh_1d_float32.o)
    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/hccl_a2a_superkernel_mesh_1d_float32.o
            COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_a2a_mesh_1d_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_a2a_superkernel_mesh_1d_float32.o
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0
            -DASCENDC_DUMP --cce-aicore-arch=dav-c310-vec
            --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-dcci-insert-for-scalar=false
            -mllvm -cce-aicore-stack-size=0x8000 -D__CCE_AICORE__=310 -D__DAV_C310__
            -DHCCL_DTYPE_FP32 -Dsk_alltoall_mesh_1d=sk_alltoall_mesh_1d_float32
            "-I$<JOIN:$<TARGET_PROPERTY:asc_kernel_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
            "-I$<JOIN:$<TARGET_PROPERTY:kernel_tiling_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
            -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMAND_EXPAND_LISTS VERBATIM
            DEPENDS kernel_tiling_headers
    )

    add_custom_target(hccl_a2a_superkernel_mesh_1d_bfloat16 ALL
        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/hccl_a2a_superkernel_mesh_1d_bfloat16.o)
    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/hccl_a2a_superkernel_mesh_1d_bfloat16.o
            COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_a2a_mesh_1d_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_a2a_superkernel_mesh_1d_bfloat16.o
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0
            -DASCENDC_DUMP --cce-aicore-arch=dav-c310-vec
            --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-dcci-insert-for-scalar=false
            -mllvm -cce-aicore-stack-size=0x8000 -D__CCE_AICORE__=310 -D__DAV_C310__
            -DHCCL_DTYPE_BFP16 -Dsk_alltoall_mesh_1d=sk_alltoall_mesh_1d_bfloat16
            "-I$<JOIN:$<TARGET_PROPERTY:asc_kernel_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
            "-I$<JOIN:$<TARGET_PROPERTY:kernel_tiling_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
            -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMAND_EXPAND_LISTS VERBATIM
            DEPENDS kernel_tiling_headers
    )

    add_custom_target(hccl_a2a_superkernel_mesh_1d_uint8 ALL
        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/hccl_a2a_superkernel_mesh_1d_uint8.o)
    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/hccl_a2a_superkernel_mesh_1d_uint8.o
            COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_a2a_mesh_1d_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_a2a_superkernel_mesh_1d_uint8.o
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0
            -DASCENDC_DUMP --cce-aicore-arch=dav-c310-vec
            --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-dcci-insert-for-scalar=false
            -mllvm -cce-aicore-stack-size=0x8000 -D__CCE_AICORE__=310 -D__DAV_C310__
            -DHCCL_DTYPE_UINT8 -Dsk_alltoall_mesh_1d=sk_alltoall_mesh_1d_uint8
            "-I$<JOIN:$<TARGET_PROPERTY:asc_kernel_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
            "-I$<JOIN:$<TARGET_PROPERTY:kernel_tiling_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
            -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMAND_EXPAND_LISTS VERBATIM
            DEPENDS kernel_tiling_headers
    )

    add_custom_target(hccl_a2a_superkernel_mesh_1d_uint16 ALL
        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/hccl_a2a_superkernel_mesh_1d_uint16.o)
    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/hccl_a2a_superkernel_mesh_1d_uint16.o
            COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_a2a_mesh_1d_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_a2a_superkernel_mesh_1d_uint16.o
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0
            -DASCENDC_DUMP --cce-aicore-arch=dav-c310-vec
            --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-dcci-insert-for-scalar=false
            -mllvm -cce-aicore-stack-size=0x8000 -D__CCE_AICORE__=310 -D__DAV_C310__
            -DHCCL_DTYPE_UINT16 -Dsk_alltoall_mesh_1d=sk_alltoall_mesh_1d_uint16
            "-I$<JOIN:$<TARGET_PROPERTY:asc_kernel_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
            "-I$<JOIN:$<TARGET_PROPERTY:kernel_tiling_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
            -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMAND_EXPAND_LISTS VERBATIM
            DEPENDS kernel_tiling_headers
    )

    add_custom_target(hccl_a2a_superkernel_mesh_1d_uint32 ALL
        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/hccl_a2a_superkernel_mesh_1d_uint32.o)
    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/hccl_a2a_superkernel_mesh_1d_uint32.o
            COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_a2a_mesh_1d_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_a2a_superkernel_mesh_1d_uint32.o
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0
            -DASCENDC_DUMP --cce-aicore-arch=dav-c310-vec
            --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-dcci-insert-for-scalar=false
            -mllvm -cce-aicore-stack-size=0x8000 -D__CCE_AICORE__=310 -D__DAV_C310__
            -DHCCL_DTYPE_UINT32 -Dsk_alltoall_mesh_1d=sk_alltoall_mesh_1d_uint32
            "-I$<JOIN:$<TARGET_PROPERTY:asc_kernel_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
            "-I$<JOIN:$<TARGET_PROPERTY:kernel_tiling_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
            -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMAND_EXPAND_LISTS VERBATIM
            DEPENDS kernel_tiling_headers
    )

    add_custom_target(hccl_a2a_superkernel_mesh_1d_int64 ALL
        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/hccl_a2a_superkernel_mesh_1d_int64.o)
    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/hccl_a2a_superkernel_mesh_1d_int64.o
            COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_a2a_mesh_1d_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_a2a_superkernel_mesh_1d_int64.o
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0
            -DASCENDC_DUMP --cce-aicore-arch=dav-c310-vec
            --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-dcci-insert-for-scalar=false
            -mllvm -cce-aicore-stack-size=0x8000 -D__CCE_AICORE__=310 -D__DAV_C310__
            -DHCCL_DTYPE_INT64 -Dsk_alltoall_mesh_1d=sk_alltoall_mesh_1d_int64
            "-I$<JOIN:$<TARGET_PROPERTY:asc_kernel_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
            "-I$<JOIN:$<TARGET_PROPERTY:kernel_tiling_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
            -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMAND_EXPAND_LISTS VERBATIM
            DEPENDS kernel_tiling_headers
    )

    add_custom_target(hccl_a2a_superkernel_mesh_1d_uint64 ALL
        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/hccl_a2a_superkernel_mesh_1d_uint64.o)
    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/hccl_a2a_superkernel_mesh_1d_uint64.o
            COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_a2a_mesh_1d_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_a2a_superkernel_mesh_1d_uint64.o
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0
            -DASCENDC_DUMP --cce-aicore-arch=dav-c310-vec
            --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-dcci-insert-for-scalar=false
            -mllvm -cce-aicore-stack-size=0x8000 -D__CCE_AICORE__=310 -D__DAV_C310__
            -DHCCL_DTYPE_UINT64 -Dsk_alltoall_mesh_1d=sk_alltoall_mesh_1d_uint64
            "-I$<JOIN:$<TARGET_PROPERTY:asc_kernel_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
            "-I$<JOIN:$<TARGET_PROPERTY:kernel_tiling_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
            -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMAND_EXPAND_LISTS VERBATIM
            DEPENDS kernel_tiling_headers
    )

else()
    get_filename_component(CCE_DIR "${CMAKE_LINKER}" PATH)

    set(TILING_GEN_DIR ${CMAKE_BINARY_DIR}/asc/asc-devkit)

    add_custom_target(hccl_aiv_op_910_95
            COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_communication_v2.h -o ${CMAKE_CURRENT_BINARY_DIR}/aiv_communication_910_95_tmp.o
            -DASCENDC_DUMP --cce-aicore-arch=dav-c310-vec
            --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-dcci-insert-for-scalar=false
            -mllvm -cce-aicore-stack-size=0x8000 -D__CCE_AICORE__=310 -D__DAV_C310__
            -I${TOP_DIR}/asc/asc-devkit/
            -I${TOP_DIR}/asc/asc-devkit/include
            -I${TOP_DIR}/asc/asc-devkit/impl
            -I${TOP_DIR}/asc/asc-devkit/include/basic_api/
            -I${TOP_DIR}/asc/asc-devkit/include/micro_api/
            -I${TOP_DIR}/asc/asc-devkit/include/simt_api/
            -I${TOP_DIR}/asc/asc-devkit/impl/basic_api/
            -I${TOP_DIR}/asc/asc-devkit/impl/micro_api/
            -I${TOP_DIR}/asc/asc-devkit/impl/simt_api/
            -I${TOP_DIR}/asc/asc-tools/cpudebug/include/
            -I${TOP_DIR}/asc/asc-tools/cpudebug/model/cpulib/
            -I${TOP_DIR}/ace/comop/inc/external/hccl/
            -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
            -I${TILING_GEN_DIR}
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all -DONE_CORE_DUMP_SIZE=1024
            COMMAND ${CCE_DIR}/ld.lld -m aicorelinux -Ttext=0 ${CMAKE_CURRENT_BINARY_DIR}/aiv_communication_910_95_tmp.o -static -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_aiv_op_910_95.o
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMAND_EXPAND_LISTS VERBATIM
            DEPENDS kernel_tiling_headers
    )

    add_custom_target(hccl_ar_superkernel_mesh_1d_oneshot_int8
            COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_ar_mesh_1d_oneshot_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_ar_superkernel_mesh_1d_oneshot_int8.o
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0
            -DASCENDC_DUMP --cce-aicore-arch=dav-c310-vec
            --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-dcci-insert-for-scalar=false
            -mllvm -cce-aicore-stack-size=0x8000 -D__CCE_AICORE__=310 -D__DAV_C310__
            -DHCCL_DTYPE_INT8 -Dsk_allreduce_mesh_1d_oneshot=sk_allreduce_mesh_1d_oneshot_int8
            -I${TOP_DIR}/asc/asc-devkit/
            -I${TOP_DIR}/asc/asc-devkit/include
            -I${TOP_DIR}/asc/asc-devkit/impl
            -I${TOP_DIR}/asc/asc-devkit/include/basic_api/
            -I${TOP_DIR}/asc/asc-devkit/include/micro_api/
            -I${TOP_DIR}/asc/asc-devkit/include/simt_api/
            -I${TOP_DIR}/asc/asc-devkit/impl/basic_api/
            -I${TOP_DIR}/asc/asc-devkit/impl/micro_api/
            -I${TOP_DIR}/asc/asc-devkit/impl/simt_api/
            -I${TOP_DIR}/asc/asc-tools/cpudebug/include/
            -I${TOP_DIR}/asc/asc-tools/cpudebug/model/cpulib/
            -I${TOP_DIR}/ace/comop/inc/external/hccl/
            -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
            -I${TILING_GEN_DIR}
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all -DONE_CORE_DUMP_SIZE=1024
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMAND_EXPAND_LISTS VERBATIM
            DEPENDS kernel_tiling_headers
    )

    add_custom_target(hccl_ar_superkernel_mesh_1d_oneshot_int16
            COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_ar_mesh_1d_oneshot_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_ar_superkernel_mesh_1d_oneshot_int16.o
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0
            -DASCENDC_DUMP --cce-aicore-arch=dav-c310-vec
            --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-dcci-insert-for-scalar=false
            -mllvm -cce-aicore-stack-size=0x8000 -D__CCE_AICORE__=310 -D__DAV_C310__
            -DHCCL_DTYPE_INT16 -Dsk_allreduce_mesh_1d_oneshot=sk_allreduce_mesh_1d_oneshot_int16
            -I${TOP_DIR}/asc/asc-devkit/
            -I${TOP_DIR}/asc/asc-devkit/include
            -I${TOP_DIR}/asc/asc-devkit/impl
            -I${TOP_DIR}/asc/asc-devkit/include/basic_api/
            -I${TOP_DIR}/asc/asc-devkit/include/micro_api/
            -I${TOP_DIR}/asc/asc-devkit/include/simt_api/
            -I${TOP_DIR}/asc/asc-devkit/impl/basic_api/
            -I${TOP_DIR}/asc/asc-devkit/impl/micro_api/
            -I${TOP_DIR}/asc/asc-devkit/impl/simt_api/
            -I${TOP_DIR}/asc/asc-tools/cpudebug/include/
            -I${TOP_DIR}/asc/asc-tools/cpudebug/model/cpulib/
            -I${TOP_DIR}/ace/comop/inc/external/hccl/
            -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
            -I${TILING_GEN_DIR}
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all -DONE_CORE_DUMP_SIZE=1024
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMAND_EXPAND_LISTS VERBATIM
            DEPENDS kernel_tiling_headers
    )

    add_custom_target(hccl_ar_superkernel_mesh_1d_oneshot_int32
            COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_ar_mesh_1d_oneshot_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_ar_superkernel_mesh_1d_oneshot_int32.o
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0
            -DASCENDC_DUMP --cce-aicore-arch=dav-c310-vec
            --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-dcci-insert-for-scalar=false
            -mllvm -cce-aicore-stack-size=0x8000 -D__CCE_AICORE__=310 -D__DAV_C310__
            -DHCCL_DTYPE_INT32 -Dsk_allreduce_mesh_1d_oneshot=sk_allreduce_mesh_1d_oneshot_int32
            -I${TOP_DIR}/asc/asc-devkit/
            -I${TOP_DIR}/asc/asc-devkit/include
            -I${TOP_DIR}/asc/asc-devkit/impl
            -I${TOP_DIR}/asc/asc-devkit/include/basic_api/
            -I${TOP_DIR}/asc/asc-devkit/include/micro_api/
            -I${TOP_DIR}/asc/asc-devkit/include/simt_api/
            -I${TOP_DIR}/asc/asc-devkit/impl/basic_api/
            -I${TOP_DIR}/asc/asc-devkit/impl/micro_api/
            -I${TOP_DIR}/asc/asc-devkit/impl/simt_api/
            -I${TOP_DIR}/asc/asc-tools/cpudebug/include/
            -I${TOP_DIR}/asc/asc-tools/cpudebug/model/cpulib/
            -I${TOP_DIR}/ace/comop/inc/external/hccl/
            -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
            -I${TILING_GEN_DIR}
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all -DONE_CORE_DUMP_SIZE=1024
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMAND_EXPAND_LISTS VERBATIM
            DEPENDS kernel_tiling_headers
    )

    add_custom_target(hccl_ar_superkernel_mesh_1d_oneshot_float16
            COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_ar_mesh_1d_oneshot_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_ar_superkernel_mesh_1d_oneshot_float16.o
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0
            -DASCENDC_DUMP --cce-aicore-arch=dav-c310-vec
            --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-dcci-insert-for-scalar=false
            -mllvm -cce-aicore-stack-size=0x8000 -D__CCE_AICORE__=310 -D__DAV_C310__
            -DHCCL_DTYPE_FP16 -Dsk_allreduce_mesh_1d_oneshot=sk_allreduce_mesh_1d_oneshot_float16
            -I${TOP_DIR}/asc/asc-devkit/
            -I${TOP_DIR}/asc/asc-devkit/include
            -I${TOP_DIR}/asc/asc-devkit/impl
            -I${TOP_DIR}/asc/asc-devkit/include/basic_api/
            -I${TOP_DIR}/asc/asc-devkit/include/micro_api/
            -I${TOP_DIR}/asc/asc-devkit/include/simt_api/
            -I${TOP_DIR}/asc/asc-devkit/impl/basic_api/
            -I${TOP_DIR}/asc/asc-devkit/impl/micro_api/
            -I${TOP_DIR}/asc/asc-devkit/impl/simt_api/
            -I${TOP_DIR}/asc/asc-tools/cpudebug/include/
            -I${TOP_DIR}/asc/asc-tools/cpudebug/model/cpulib/
            -I${TOP_DIR}/ace/comop/inc/external/hccl/
            -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
            -I${TILING_GEN_DIR}
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all -DONE_CORE_DUMP_SIZE=1024
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMAND_EXPAND_LISTS VERBATIM
            DEPENDS kernel_tiling_headers
    )

    add_custom_target(hccl_ar_superkernel_mesh_1d_oneshot_float32
            COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_ar_mesh_1d_oneshot_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_ar_superkernel_mesh_1d_oneshot_float32.o
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0
            -DASCENDC_DUMP --cce-aicore-arch=dav-c310-vec
            --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-dcci-insert-for-scalar=false
            -mllvm -cce-aicore-stack-size=0x8000 -D__CCE_AICORE__=310 -D__DAV_C310__
            -DHCCL_DTYPE_FP32 -Dsk_allreduce_mesh_1d_oneshot=sk_allreduce_mesh_1d_oneshot_float32
            -I${TOP_DIR}/asc/asc-devkit/
            -I${TOP_DIR}/asc/asc-devkit/include
            -I${TOP_DIR}/asc/asc-devkit/impl
            -I${TOP_DIR}/asc/asc-devkit/include/basic_api/
            -I${TOP_DIR}/asc/asc-devkit/include/micro_api/
            -I${TOP_DIR}/asc/asc-devkit/include/simt_api/
            -I${TOP_DIR}/asc/asc-devkit/impl/basic_api/
            -I${TOP_DIR}/asc/asc-devkit/impl/micro_api/
            -I${TOP_DIR}/asc/asc-devkit/impl/simt_api/
            -I${TOP_DIR}/asc/asc-tools/cpudebug/include/
            -I${TOP_DIR}/asc/asc-tools/cpudebug/model/cpulib/
            -I${TOP_DIR}/ace/comop/inc/external/hccl/
            -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
            -I${TILING_GEN_DIR}
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all -DONE_CORE_DUMP_SIZE=1024
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMAND_EXPAND_LISTS VERBATIM
            DEPENDS kernel_tiling_headers
    )

    add_custom_target(hccl_ar_superkernel_mesh_1d_oneshot_bfloat16
            COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_ar_mesh_1d_oneshot_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_ar_superkernel_mesh_1d_oneshot_bfloat16.o
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0
            -DASCENDC_DUMP --cce-aicore-arch=dav-c310-vec
            --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-dcci-insert-for-scalar=false
            -mllvm -cce-aicore-stack-size=0x8000 -D__CCE_AICORE__=310 -D__DAV_C310__
            -DHCCL_DTYPE_BFP16 -Dsk_allreduce_mesh_1d_oneshot=sk_allreduce_mesh_1d_oneshot_bfloat16
            -I${TOP_DIR}/asc/asc-devkit/
            -I${TOP_DIR}/asc/asc-devkit/include
            -I${TOP_DIR}/asc/asc-devkit/impl
            -I${TOP_DIR}/asc/asc-devkit/include/basic_api/
            -I${TOP_DIR}/asc/asc-devkit/include/micro_api/
            -I${TOP_DIR}/asc/asc-devkit/include/simt_api/
            -I${TOP_DIR}/asc/asc-devkit/impl/basic_api/
            -I${TOP_DIR}/asc/asc-devkit/impl/micro_api/
            -I${TOP_DIR}/asc/asc-devkit/impl/simt_api/
            -I${TOP_DIR}/asc/asc-tools/cpudebug/include/
            -I${TOP_DIR}/asc/asc-tools/cpudebug/model/cpulib/
            -I${TOP_DIR}/ace/comop/inc/external/hccl/
            -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
            -I${TILING_GEN_DIR}
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all -DONE_CORE_DUMP_SIZE=1024
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMAND_EXPAND_LISTS VERBATIM
            DEPENDS kernel_tiling_headers
    )

    add_custom_target(hccl_ar_superkernel_mesh_1d_twoshot_int8
            COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_ar_mesh_1d_twoshot_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_ar_superkernel_mesh_1d_twoshot_int8.o
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0
            -DASCENDC_DUMP --cce-aicore-arch=dav-c310-vec
            --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-dcci-insert-for-scalar=false
            -mllvm -cce-aicore-stack-size=0x8000 -D__CCE_AICORE__=310 -D__DAV_C310__
            -DHCCL_DTYPE_INT8 -Dsk_allreduce_mesh_1d_twoshot=sk_allreduce_mesh_1d_twoshot_int8
            -I${TOP_DIR}/asc/asc-devkit/
            -I${TOP_DIR}/asc/asc-devkit/include
            -I${TOP_DIR}/asc/asc-devkit/impl
            -I${TOP_DIR}/asc/asc-devkit/include/basic_api/
            -I${TOP_DIR}/asc/asc-devkit/include/micro_api/
            -I${TOP_DIR}/asc/asc-devkit/include/simt_api/
            -I${TOP_DIR}/asc/asc-devkit/impl/basic_api/
            -I${TOP_DIR}/asc/asc-devkit/impl/micro_api/
            -I${TOP_DIR}/asc/asc-devkit/impl/simt_api/
            -I${TOP_DIR}/asc/asc-tools/cpudebug/include/
            -I${TOP_DIR}/asc/asc-tools/cpudebug/model/cpulib/
            -I${TOP_DIR}/ace/comop/inc/external/hccl/
            -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
            -I${TILING_GEN_DIR}
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all -DONE_CORE_DUMP_SIZE=1024
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMAND_EXPAND_LISTS VERBATIM
            DEPENDS kernel_tiling_headers
    )

    add_custom_target(hccl_ar_superkernel_mesh_1d_twoshot_int16
            COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_ar_mesh_1d_twoshot_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_ar_superkernel_mesh_1d_twoshot_int16.o
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0
            -DASCENDC_DUMP --cce-aicore-arch=dav-c310-vec
            --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-dcci-insert-for-scalar=false
            -mllvm -cce-aicore-stack-size=0x8000 -D__CCE_AICORE__=310 -D__DAV_C310__
            -DHCCL_DTYPE_INT16 -Dsk_allreduce_mesh_1d_twoshot=sk_allreduce_mesh_1d_twoshot_int16
            -I${TOP_DIR}/asc/asc-devkit/
            -I${TOP_DIR}/asc/asc-devkit/include
            -I${TOP_DIR}/asc/asc-devkit/impl
            -I${TOP_DIR}/asc/asc-devkit/include/basic_api/
            -I${TOP_DIR}/asc/asc-devkit/include/micro_api/
            -I${TOP_DIR}/asc/asc-devkit/include/simt_api/
            -I${TOP_DIR}/asc/asc-devkit/impl/basic_api/
            -I${TOP_DIR}/asc/asc-devkit/impl/micro_api/
            -I${TOP_DIR}/asc/asc-devkit/impl/simt_api/
            -I${TOP_DIR}/asc/asc-tools/cpudebug/include/
            -I${TOP_DIR}/asc/asc-tools/cpudebug/model/cpulib/
            -I${TOP_DIR}/ace/comop/inc/external/hccl/
            -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
            -I${TILING_GEN_DIR}
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all -DONE_CORE_DUMP_SIZE=1024
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMAND_EXPAND_LISTS VERBATIM
            DEPENDS kernel_tiling_headers
    )

    add_custom_target(hccl_ar_superkernel_mesh_1d_twoshot_int32
            COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_ar_mesh_1d_twoshot_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_ar_superkernel_mesh_1d_twoshot_int32.o
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0
            -DASCENDC_DUMP --cce-aicore-arch=dav-c310-vec
            --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-dcci-insert-for-scalar=false
            -mllvm -cce-aicore-stack-size=0x8000 -D__CCE_AICORE__=310 -D__DAV_C310__
            -DHCCL_DTYPE_INT32 -Dsk_allreduce_mesh_1d_twoshot=sk_allreduce_mesh_1d_twoshot_int32
            -I${TOP_DIR}/asc/asc-devkit/
            -I${TOP_DIR}/asc/asc-devkit/include
            -I${TOP_DIR}/asc/asc-devkit/impl
            -I${TOP_DIR}/asc/asc-devkit/include/basic_api/
            -I${TOP_DIR}/asc/asc-devkit/include/micro_api/
            -I${TOP_DIR}/asc/asc-devkit/include/simt_api/
            -I${TOP_DIR}/asc/asc-devkit/impl/basic_api/
            -I${TOP_DIR}/asc/asc-devkit/impl/micro_api/
            -I${TOP_DIR}/asc/asc-devkit/impl/simt_api/
            -I${TOP_DIR}/asc/asc-tools/cpudebug/include/
            -I${TOP_DIR}/asc/asc-tools/cpudebug/model/cpulib/
            -I${TOP_DIR}/ace/comop/inc/external/hccl/
            -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
            -I${TILING_GEN_DIR}
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all -DONE_CORE_DUMP_SIZE=1024
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMAND_EXPAND_LISTS VERBATIM
            DEPENDS kernel_tiling_headers
    )

    add_custom_target(hccl_ar_superkernel_mesh_1d_twoshot_float16
            COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_ar_mesh_1d_twoshot_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_ar_superkernel_mesh_1d_twoshot_float16.o
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0
            -DASCENDC_DUMP --cce-aicore-arch=dav-c310-vec
            --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-dcci-insert-for-scalar=false
            -mllvm -cce-aicore-stack-size=0x8000 -D__CCE_AICORE__=310 -D__DAV_C310__
            -DHCCL_DTYPE_FP16 -Dsk_allreduce_mesh_1d_twoshot=sk_allreduce_mesh_1d_twoshot_float16
            -I${TOP_DIR}/asc/asc-devkit/
            -I${TOP_DIR}/asc/asc-devkit/include
            -I${TOP_DIR}/asc/asc-devkit/impl
            -I${TOP_DIR}/asc/asc-devkit/include/basic_api/
            -I${TOP_DIR}/asc/asc-devkit/include/micro_api/
            -I${TOP_DIR}/asc/asc-devkit/include/simt_api/
            -I${TOP_DIR}/asc/asc-devkit/impl/basic_api/
            -I${TOP_DIR}/asc/asc-devkit/impl/micro_api/
            -I${TOP_DIR}/asc/asc-devkit/impl/simt_api/
            -I${TOP_DIR}/asc/asc-tools/cpudebug/include/
            -I${TOP_DIR}/asc/asc-tools/cpudebug/model/cpulib/
            -I${TOP_DIR}/ace/comop/inc/external/hccl/
            -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
            -I${TILING_GEN_DIR}
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all -DONE_CORE_DUMP_SIZE=1024
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMAND_EXPAND_LISTS VERBATIM
            DEPENDS kernel_tiling_headers
    )

    add_custom_target(hccl_ar_superkernel_mesh_1d_twoshot_float32
            COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_ar_mesh_1d_twoshot_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_ar_superkernel_mesh_1d_twoshot_float32.o
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0
            -DASCENDC_DUMP --cce-aicore-arch=dav-c310-vec
            --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-dcci-insert-for-scalar=false
            -mllvm -cce-aicore-stack-size=0x8000 -D__CCE_AICORE__=310 -D__DAV_C310__
            -DHCCL_DTYPE_FP32 -Dsk_allreduce_mesh_1d_twoshot=sk_allreduce_mesh_1d_twoshot_float32
            -I${TOP_DIR}/asc/asc-devkit/
            -I${TOP_DIR}/asc/asc-devkit/include
            -I${TOP_DIR}/asc/asc-devkit/impl
            -I${TOP_DIR}/asc/asc-devkit/include/basic_api/
            -I${TOP_DIR}/asc/asc-devkit/include/micro_api/
            -I${TOP_DIR}/asc/asc-devkit/include/simt_api/
            -I${TOP_DIR}/asc/asc-devkit/impl/basic_api/
            -I${TOP_DIR}/asc/asc-devkit/impl/micro_api/
            -I${TOP_DIR}/asc/asc-devkit/impl/simt_api/
            -I${TOP_DIR}/asc/asc-tools/cpudebug/include/
            -I${TOP_DIR}/asc/asc-tools/cpudebug/model/cpulib/
            -I${TOP_DIR}/ace/comop/inc/external/hccl/
            -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
            -I${TILING_GEN_DIR}
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all -DONE_CORE_DUMP_SIZE=1024
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMAND_EXPAND_LISTS VERBATIM
            DEPENDS kernel_tiling_headers
    )

    add_custom_target(hccl_ar_superkernel_mesh_1d_twoshot_bfloat16
            COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_ar_mesh_1d_twoshot_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_ar_superkernel_mesh_1d_twoshot_bfloat16.o
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0
            -DASCENDC_DUMP --cce-aicore-arch=dav-c310-vec
            --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-dcci-insert-for-scalar=false
            -mllvm -cce-aicore-stack-size=0x8000 -D__CCE_AICORE__=310 -D__DAV_C310__
            -DHCCL_DTYPE_BFP16 -Dsk_allreduce_mesh_1d_twoshot=sk_allreduce_mesh_1d_twoshot_bfloat16
            -I${TOP_DIR}/asc/asc-devkit/
            -I${TOP_DIR}/asc/asc-devkit/include
            -I${TOP_DIR}/asc/asc-devkit/impl
            -I${TOP_DIR}/asc/asc-devkit/include/basic_api/
            -I${TOP_DIR}/asc/asc-devkit/include/micro_api/
            -I${TOP_DIR}/asc/asc-devkit/include/simt_api/
            -I${TOP_DIR}/asc/asc-devkit/impl/basic_api/
            -I${TOP_DIR}/asc/asc-devkit/impl/micro_api/
            -I${TOP_DIR}/asc/asc-devkit/impl/simt_api/
            -I${TOP_DIR}/asc/asc-tools/cpudebug/include/
            -I${TOP_DIR}/asc/asc-tools/cpudebug/model/cpulib/
            -I${TOP_DIR}/ace/comop/inc/external/hccl/
            -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
            -I${TILING_GEN_DIR}
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all -DONE_CORE_DUMP_SIZE=1024
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMAND_EXPAND_LISTS VERBATIM
            DEPENDS kernel_tiling_headers
    )

    add_custom_target(hccl_rs_superkernel_mesh_1d_int8
            COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_rs_mesh_1d_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_rs_superkernel_mesh_1d_int8.o
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0
            -DASCENDC_DUMP --cce-aicore-arch=dav-c310-vec
            --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-dcci-insert-for-scalar=false
            -mllvm -cce-aicore-stack-size=0x8000 -D__CCE_AICORE__=310 -D__DAV_C310__
            -DHCCL_DTYPE_INT8 -Dsk_reducescatter_mesh_1d=sk_reducescatter_mesh_1d_int8
            -I${TOP_DIR}/asc/asc-devkit/
            -I${TOP_DIR}/asc/asc-devkit/include
            -I${TOP_DIR}/asc/asc-devkit/impl
            -I${TOP_DIR}/asc/asc-devkit/include/basic_api/
            -I${TOP_DIR}/asc/asc-devkit/include/micro_api/
            -I${TOP_DIR}/asc/asc-devkit/include/simt_api/
            -I${TOP_DIR}/asc/asc-devkit/impl/basic_api/
            -I${TOP_DIR}/asc/asc-devkit/impl/micro_api/
            -I${TOP_DIR}/asc/asc-devkit/impl/simt_api/
            -I${TOP_DIR}/asc/asc-tools/cpudebug/include/
            -I${TOP_DIR}/asc/asc-tools/cpudebug/model/cpulib/
            -I${TOP_DIR}/ace/comop/inc/external/hccl/
            -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
            -I${TILING_GEN_DIR}
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all -DONE_CORE_DUMP_SIZE=1024
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMAND_EXPAND_LISTS VERBATIM
            DEPENDS kernel_tiling_headers
    )

    add_custom_target(hccl_rs_superkernel_mesh_1d_int16
            COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_rs_mesh_1d_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_rs_superkernel_mesh_1d_int16.o
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0
            -DASCENDC_DUMP --cce-aicore-arch=dav-c310-vec
            --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-dcci-insert-for-scalar=false
            -mllvm -cce-aicore-stack-size=0x8000 -D__CCE_AICORE__=310 -D__DAV_C310__
            -DHCCL_DTYPE_INT16 -Dsk_reducescatter_mesh_1d=sk_reducescatter_mesh_1d_int16
            -I${TOP_DIR}/asc/asc-devkit/
            -I${TOP_DIR}/asc/asc-devkit/include
            -I${TOP_DIR}/asc/asc-devkit/impl
            -I${TOP_DIR}/asc/asc-devkit/include/basic_api/
            -I${TOP_DIR}/asc/asc-devkit/include/micro_api/
            -I${TOP_DIR}/asc/asc-devkit/include/simt_api/
            -I${TOP_DIR}/asc/asc-devkit/impl/basic_api/
            -I${TOP_DIR}/asc/asc-devkit/impl/micro_api/
            -I${TOP_DIR}/asc/asc-devkit/impl/simt_api/
            -I${TOP_DIR}/asc/asc-tools/cpudebug/include/
            -I${TOP_DIR}/asc/asc-tools/cpudebug/model/cpulib/
            -I${TOP_DIR}/ace/comop/inc/external/hccl/
            -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
            -I${TILING_GEN_DIR}
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all -DONE_CORE_DUMP_SIZE=1024
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMAND_EXPAND_LISTS VERBATIM
            DEPENDS kernel_tiling_headers
    )

    add_custom_target(hccl_rs_superkernel_mesh_1d_int32
            COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_rs_mesh_1d_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_rs_superkernel_mesh_1d_int32.o
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0
            -DASCENDC_DUMP --cce-aicore-arch=dav-c310-vec
            --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-dcci-insert-for-scalar=false
            -mllvm -cce-aicore-stack-size=0x8000 -D__CCE_AICORE__=310 -D__DAV_C310__
            -DHCCL_DTYPE_INT32 -Dsk_reducescatter_mesh_1d=sk_reducescatter_mesh_1d_int32
            -I${TOP_DIR}/asc/asc-devkit/
            -I${TOP_DIR}/asc/asc-devkit/include
            -I${TOP_DIR}/asc/asc-devkit/impl
            -I${TOP_DIR}/asc/asc-devkit/include/basic_api/
            -I${TOP_DIR}/asc/asc-devkit/include/micro_api/
            -I${TOP_DIR}/asc/asc-devkit/include/simt_api/
            -I${TOP_DIR}/asc/asc-devkit/impl/basic_api/
            -I${TOP_DIR}/asc/asc-devkit/impl/micro_api/
            -I${TOP_DIR}/asc/asc-devkit/impl/simt_api/
            -I${TOP_DIR}/asc/asc-tools/cpudebug/include/
            -I${TOP_DIR}/asc/asc-tools/cpudebug/model/cpulib/
            -I${TOP_DIR}/ace/comop/inc/external/hccl/
            -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
            -I${TILING_GEN_DIR}
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all -DONE_CORE_DUMP_SIZE=1024
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMAND_EXPAND_LISTS VERBATIM
            DEPENDS kernel_tiling_headers
    )

    add_custom_target(hccl_rs_superkernel_mesh_1d_float16
            COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_rs_mesh_1d_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_rs_superkernel_mesh_1d_float16.o
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0
            -DASCENDC_DUMP --cce-aicore-arch=dav-c310-vec
            --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-dcci-insert-for-scalar=false
            -mllvm -cce-aicore-stack-size=0x8000 -D__CCE_AICORE__=310 -D__DAV_C310__
            -DHCCL_DTYPE_FP16 -Dsk_reducescatter_mesh_1d=sk_reducescatter_mesh_1d_float16
            -I${TOP_DIR}/asc/asc-devkit/
            -I${TOP_DIR}/asc/asc-devkit/include
            -I${TOP_DIR}/asc/asc-devkit/impl
            -I${TOP_DIR}/asc/asc-devkit/include/basic_api/
            -I${TOP_DIR}/asc/asc-devkit/include/micro_api/
            -I${TOP_DIR}/asc/asc-devkit/include/simt_api/
            -I${TOP_DIR}/asc/asc-devkit/impl/basic_api/
            -I${TOP_DIR}/asc/asc-devkit/impl/micro_api/
            -I${TOP_DIR}/asc/asc-devkit/impl/simt_api/
            -I${TOP_DIR}/asc/asc-tools/cpudebug/include/
            -I${TOP_DIR}/asc/asc-tools/cpudebug/model/cpulib/
            -I${TOP_DIR}/ace/comop/inc/external/hccl/
            -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
            -I${TILING_GEN_DIR}
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all -DONE_CORE_DUMP_SIZE=1024
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMAND_EXPAND_LISTS VERBATIM
            DEPENDS kernel_tiling_headers
    )

    add_custom_target(hccl_rs_superkernel_mesh_1d_float32
            COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_rs_mesh_1d_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_rs_superkernel_mesh_1d_float32.o
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0
            -DASCENDC_DUMP --cce-aicore-arch=dav-c310-vec
            --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-dcci-insert-for-scalar=false
            -mllvm -cce-aicore-stack-size=0x8000 -D__CCE_AICORE__=310 -D__DAV_C310__
            -DHCCL_DTYPE_FP32 -Dsk_reducescatter_mesh_1d=sk_reducescatter_mesh_1d_float32
            -I${TOP_DIR}/asc/asc-devkit/
            -I${TOP_DIR}/asc/asc-devkit/include
            -I${TOP_DIR}/asc/asc-devkit/impl
            -I${TOP_DIR}/asc/asc-devkit/include/basic_api/
            -I${TOP_DIR}/asc/asc-devkit/include/micro_api/
            -I${TOP_DIR}/asc/asc-devkit/include/simt_api/
            -I${TOP_DIR}/asc/asc-devkit/impl/basic_api/
            -I${TOP_DIR}/asc/asc-devkit/impl/micro_api/
            -I${TOP_DIR}/asc/asc-devkit/impl/simt_api/
            -I${TOP_DIR}/asc/asc-tools/cpudebug/include/
            -I${TOP_DIR}/asc/asc-tools/cpudebug/model/cpulib/
            -I${TOP_DIR}/ace/comop/inc/external/hccl/
            -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
            -I${TILING_GEN_DIR}
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all -DONE_CORE_DUMP_SIZE=1024
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMAND_EXPAND_LISTS VERBATIM
            DEPENDS kernel_tiling_headers
    )

    add_custom_target(hccl_rs_superkernel_mesh_1d_bfloat16
            COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_rs_mesh_1d_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_rs_superkernel_mesh_1d_bfloat16.o
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0
            -DASCENDC_DUMP --cce-aicore-arch=dav-c310-vec
            --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-dcci-insert-for-scalar=false
            -mllvm -cce-aicore-stack-size=0x8000 -D__CCE_AICORE__=310 -D__DAV_C310__
            -DHCCL_DTYPE_BFP16 -Dsk_reducescatter_mesh_1d=sk_reducescatter_mesh_1d_bfloat16
            -I${TOP_DIR}/asc/asc-devkit/
            -I${TOP_DIR}/asc/asc-devkit/include
            -I${TOP_DIR}/asc/asc-devkit/impl
            -I${TOP_DIR}/asc/asc-devkit/include/basic_api/
            -I${TOP_DIR}/asc/asc-devkit/include/micro_api/
            -I${TOP_DIR}/asc/asc-devkit/include/simt_api/
            -I${TOP_DIR}/asc/asc-devkit/impl/basic_api/
            -I${TOP_DIR}/asc/asc-devkit/impl/micro_api/
            -I${TOP_DIR}/asc/asc-devkit/impl/simt_api/
            -I${TOP_DIR}/asc/asc-tools/cpudebug/include/
            -I${TOP_DIR}/asc/asc-tools/cpudebug/model/cpulib/
            -I${TOP_DIR}/ace/comop/inc/external/hccl/
            -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
            -I${TILING_GEN_DIR}
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all -DONE_CORE_DUMP_SIZE=1024
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMAND_EXPAND_LISTS VERBATIM
            DEPENDS kernel_tiling_headers
    )

    add_custom_target(hccl_ag_superkernel_mesh_1d_int8
            COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_ag_mesh_1d_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_mesh_1d_int8.o
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0
            -DASCENDC_DUMP --cce-aicore-arch=dav-c310-vec
            --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-dcci-insert-for-scalar=false
            -mllvm -cce-aicore-stack-size=0x8000 -D__CCE_AICORE__=310 -D__DAV_C310__
            -DHCCL_DTYPE_INT8 -Dsk_allgather_mesh_1d=sk_allgather_mesh_1d_int8
            -I${TOP_DIR}/asc/asc-devkit/
            -I${TOP_DIR}/asc/asc-devkit/include
            -I${TOP_DIR}/asc/asc-devkit/impl
            -I${TOP_DIR}/asc/asc-devkit/include/basic_api/
            -I${TOP_DIR}/asc/asc-devkit/include/micro_api/
            -I${TOP_DIR}/asc/asc-devkit/include/simt_api/
            -I${TOP_DIR}/asc/asc-devkit/impl/basic_api/
            -I${TOP_DIR}/asc/asc-devkit/impl/micro_api/
            -I${TOP_DIR}/asc/asc-devkit/impl/simt_api/
            -I${TOP_DIR}/asc/asc-tools/cpudebug/include/
            -I${TOP_DIR}/asc/asc-tools/cpudebug/model/cpulib/
            -I${TOP_DIR}/ace/comop/inc/external/hccl/
            -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
            -I${TILING_GEN_DIR}
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all -DONE_CORE_DUMP_SIZE=1024
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMAND_EXPAND_LISTS VERBATIM
            DEPENDS kernel_tiling_headers
    )

    add_custom_target(hccl_ag_superkernel_mesh_1d_int16
            COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_ag_mesh_1d_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_mesh_1d_int16.o
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0
            -DASCENDC_DUMP --cce-aicore-arch=dav-c310-vec
            --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-dcci-insert-for-scalar=false
            -mllvm -cce-aicore-stack-size=0x8000 -D__CCE_AICORE__=310 -D__DAV_C310__
            -DHCCL_DTYPE_INT16 -Dsk_allgather_mesh_1d=sk_allgather_mesh_1d_int16
            -I${TOP_DIR}/asc/asc-devkit/
            -I${TOP_DIR}/asc/asc-devkit/include
            -I${TOP_DIR}/asc/asc-devkit/impl
            -I${TOP_DIR}/asc/asc-devkit/include/basic_api/
            -I${TOP_DIR}/asc/asc-devkit/include/micro_api/
            -I${TOP_DIR}/asc/asc-devkit/include/simt_api/
            -I${TOP_DIR}/asc/asc-devkit/impl/basic_api/
            -I${TOP_DIR}/asc/asc-devkit/impl/micro_api/
            -I${TOP_DIR}/asc/asc-devkit/impl/simt_api/
            -I${TOP_DIR}/asc/asc-tools/cpudebug/include/
            -I${TOP_DIR}/asc/asc-tools/cpudebug/model/cpulib/
            -I${TOP_DIR}/ace/comop/inc/external/hccl/
            -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
            -I${TILING_GEN_DIR}
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all -DONE_CORE_DUMP_SIZE=1024
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMAND_EXPAND_LISTS VERBATIM
            DEPENDS kernel_tiling_headers
    )

    add_custom_target(hccl_ag_superkernel_mesh_1d_int32
            COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_ag_mesh_1d_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_mesh_1d_int32.o
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0
            -DASCENDC_DUMP --cce-aicore-arch=dav-c310-vec
            --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-dcci-insert-for-scalar=false
            -mllvm -cce-aicore-stack-size=0x8000 -D__CCE_AICORE__=310 -D__DAV_C310__
            -DHCCL_DTYPE_INT32 -Dsk_allgather_mesh_1d=sk_allgather_mesh_1d_int32
            -I${TOP_DIR}/asc/asc-devkit/
            -I${TOP_DIR}/asc/asc-devkit/include
            -I${TOP_DIR}/asc/asc-devkit/impl
            -I${TOP_DIR}/asc/asc-devkit/include/basic_api/
            -I${TOP_DIR}/asc/asc-devkit/include/micro_api/
            -I${TOP_DIR}/asc/asc-devkit/include/simt_api/
            -I${TOP_DIR}/asc/asc-devkit/impl/basic_api/
            -I${TOP_DIR}/asc/asc-devkit/impl/micro_api/
            -I${TOP_DIR}/asc/asc-devkit/impl/simt_api/
            -I${TOP_DIR}/asc/asc-tools/cpudebug/include/
            -I${TOP_DIR}/asc/asc-tools/cpudebug/model/cpulib/
            -I${TOP_DIR}/ace/comop/inc/external/hccl/
            -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
            -I${TILING_GEN_DIR}
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all -DONE_CORE_DUMP_SIZE=1024
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMAND_EXPAND_LISTS VERBATIM
            DEPENDS kernel_tiling_headers
    )

    add_custom_target(hccl_ag_superkernel_mesh_1d_float16
            COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_ag_mesh_1d_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_mesh_1d_float16.o
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0
            -DASCENDC_DUMP --cce-aicore-arch=dav-c310-vec
            --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-dcci-insert-for-scalar=false
            -mllvm -cce-aicore-stack-size=0x8000 -D__CCE_AICORE__=310 -D__DAV_C310__
            -DHCCL_DTYPE_FP16 -Dsk_allgather_mesh_1d=sk_allgather_mesh_1d_float16
            -I${TOP_DIR}/asc/asc-devkit/
            -I${TOP_DIR}/asc/asc-devkit/include
            -I${TOP_DIR}/asc/asc-devkit/impl
            -I${TOP_DIR}/asc/asc-devkit/include/basic_api/
            -I${TOP_DIR}/asc/asc-devkit/include/micro_api/
            -I${TOP_DIR}/asc/asc-devkit/include/simt_api/
            -I${TOP_DIR}/asc/asc-devkit/impl/basic_api/
            -I${TOP_DIR}/asc/asc-devkit/impl/micro_api/
            -I${TOP_DIR}/asc/asc-devkit/impl/simt_api/
            -I${TOP_DIR}/asc/asc-tools/cpudebug/include/
            -I${TOP_DIR}/asc/asc-tools/cpudebug/model/cpulib/
            -I${TOP_DIR}/ace/comop/inc/external/hccl/
            -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
            -I${TILING_GEN_DIR}
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all -DONE_CORE_DUMP_SIZE=1024
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMAND_EXPAND_LISTS VERBATIM
            DEPENDS kernel_tiling_headers
    )

    add_custom_target(hccl_ag_superkernel_mesh_1d_float32
            COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_ag_mesh_1d_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_mesh_1d_float32.o
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0
            -DASCENDC_DUMP --cce-aicore-arch=dav-c310-vec
            --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-dcci-insert-for-scalar=false
            -mllvm -cce-aicore-stack-size=0x8000 -D__CCE_AICORE__=310 -D__DAV_C310__
            -DHCCL_DTYPE_FP32 -Dsk_allgather_mesh_1d=sk_allgather_mesh_1d_float32
            -I${TOP_DIR}/asc/asc-devkit/
            -I${TOP_DIR}/asc/asc-devkit/include
            -I${TOP_DIR}/asc/asc-devkit/impl
            -I${TOP_DIR}/asc/asc-devkit/include/basic_api/
            -I${TOP_DIR}/asc/asc-devkit/include/micro_api/
            -I${TOP_DIR}/asc/asc-devkit/include/simt_api/
            -I${TOP_DIR}/asc/asc-devkit/impl/basic_api/
            -I${TOP_DIR}/asc/asc-devkit/impl/micro_api/
            -I${TOP_DIR}/asc/asc-devkit/impl/simt_api/
            -I${TOP_DIR}/asc/asc-tools/cpudebug/include/
            -I${TOP_DIR}/asc/asc-tools/cpudebug/model/cpulib/
            -I${TOP_DIR}/ace/comop/inc/external/hccl/
            -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
            -I${TILING_GEN_DIR}
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all -DONE_CORE_DUMP_SIZE=1024
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMAND_EXPAND_LISTS VERBATIM
            DEPENDS kernel_tiling_headers
    )

    add_custom_target(hccl_ag_superkernel_mesh_1d_bfloat16
            COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_ag_mesh_1d_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_mesh_1d_bfloat16.o
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0
            -DASCENDC_DUMP --cce-aicore-arch=dav-c310-vec
            --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-dcci-insert-for-scalar=false
            -mllvm -cce-aicore-stack-size=0x8000 -D__CCE_AICORE__=310 -D__DAV_C310__
            -DHCCL_DTYPE_BFP16 -Dsk_allgather_mesh_1d=sk_allgather_mesh_1d_bfloat16
            -I${TOP_DIR}/asc/asc-devkit/
            -I${TOP_DIR}/asc/asc-devkit/include
            -I${TOP_DIR}/asc/asc-devkit/impl
            -I${TOP_DIR}/asc/asc-devkit/include/basic_api/
            -I${TOP_DIR}/asc/asc-devkit/include/micro_api/
            -I${TOP_DIR}/asc/asc-devkit/include/simt_api/
            -I${TOP_DIR}/asc/asc-devkit/impl/basic_api/
            -I${TOP_DIR}/asc/asc-devkit/impl/micro_api/
            -I${TOP_DIR}/asc/asc-devkit/impl/simt_api/
            -I${TOP_DIR}/asc/asc-tools/cpudebug/include/
            -I${TOP_DIR}/asc/asc-tools/cpudebug/model/cpulib/
            -I${TOP_DIR}/ace/comop/inc/external/hccl/
            -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
            -I${TILING_GEN_DIR}
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all -DONE_CORE_DUMP_SIZE=1024
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMAND_EXPAND_LISTS VERBATIM
            DEPENDS kernel_tiling_headers
    )

    add_custom_target(hccl_ag_superkernel_mesh_1d_uint8
            COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_ag_mesh_1d_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_mesh_1d_uint8.o
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0
            -DASCENDC_DUMP --cce-aicore-arch=dav-c310-vec
            --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-dcci-insert-for-scalar=false
            -mllvm -cce-aicore-stack-size=0x8000 -D__CCE_AICORE__=310 -D__DAV_C310__
            -DHCCL_DTYPE_UINT8 -Dsk_allgather_mesh_1d=sk_allgather_mesh_1d_uint8
            -I${TOP_DIR}/asc/asc-devkit/
            -I${TOP_DIR}/asc/asc-devkit/include
            -I${TOP_DIR}/asc/asc-devkit/impl
            -I${TOP_DIR}/asc/asc-devkit/include/basic_api/
            -I${TOP_DIR}/asc/asc-devkit/include/micro_api/
            -I${TOP_DIR}/asc/asc-devkit/include/simt_api/
            -I${TOP_DIR}/asc/asc-devkit/impl/basic_api/
            -I${TOP_DIR}/asc/asc-devkit/impl/micro_api/
            -I${TOP_DIR}/asc/asc-devkit/impl/simt_api/
            -I${TOP_DIR}/asc/asc-tools/cpudebug/include/
            -I${TOP_DIR}/asc/asc-tools/cpudebug/model/cpulib/
            -I${TOP_DIR}/ace/comop/inc/external/hccl/
            -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
            -I${TILING_GEN_DIR}
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all -DONE_CORE_DUMP_SIZE=1024
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMAND_EXPAND_LISTS VERBATIM
            DEPENDS kernel_tiling_headers
    )

    add_custom_target(hccl_ag_superkernel_mesh_1d_uint16
            COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_ag_mesh_1d_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_mesh_1d_uint16.o
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0
            -DASCENDC_DUMP --cce-aicore-arch=dav-c310-vec
            --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-dcci-insert-for-scalar=false
            -mllvm -cce-aicore-stack-size=0x8000 -D__CCE_AICORE__=310 -D__DAV_C310__
            -DHCCL_DTYPE_UINT16 -Dsk_allgather_mesh_1d=sk_allgather_mesh_1d_uint16
            -I${TOP_DIR}/asc/asc-devkit/
            -I${TOP_DIR}/asc/asc-devkit/include
            -I${TOP_DIR}/asc/asc-devkit/impl
            -I${TOP_DIR}/asc/asc-devkit/include/basic_api/
            -I${TOP_DIR}/asc/asc-devkit/include/micro_api/
            -I${TOP_DIR}/asc/asc-devkit/include/simt_api/
            -I${TOP_DIR}/asc/asc-devkit/impl/basic_api/
            -I${TOP_DIR}/asc/asc-devkit/impl/micro_api/
            -I${TOP_DIR}/asc/asc-devkit/impl/simt_api/
            -I${TOP_DIR}/asc/asc-tools/cpudebug/include/
            -I${TOP_DIR}/asc/asc-tools/cpudebug/model/cpulib/
            -I${TOP_DIR}/ace/comop/inc/external/hccl/
            -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
            -I${TILING_GEN_DIR}
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all -DONE_CORE_DUMP_SIZE=1024
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMAND_EXPAND_LISTS VERBATIM
            DEPENDS kernel_tiling_headers
    )

    add_custom_target(hccl_ag_superkernel_mesh_1d_uint32
            COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_ag_mesh_1d_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_mesh_1d_uint32.o
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0
            -DASCENDC_DUMP --cce-aicore-arch=dav-c310-vec
            --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-dcci-insert-for-scalar=false
            -mllvm -cce-aicore-stack-size=0x8000 -D__CCE_AICORE__=310 -D__DAV_C310__
            -DHCCL_DTYPE_UINT32 -Dsk_allgather_mesh_1d=sk_allgather_mesh_1d_uint32
            -I${TOP_DIR}/asc/asc-devkit/
            -I${TOP_DIR}/asc/asc-devkit/include
            -I${TOP_DIR}/asc/asc-devkit/impl
            -I${TOP_DIR}/asc/asc-devkit/include/basic_api/
            -I${TOP_DIR}/asc/asc-devkit/include/micro_api/
            -I${TOP_DIR}/asc/asc-devkit/include/simt_api/
            -I${TOP_DIR}/asc/asc-devkit/impl/basic_api/
            -I${TOP_DIR}/asc/asc-devkit/impl/micro_api/
            -I${TOP_DIR}/asc/asc-devkit/impl/simt_api/
            -I${TOP_DIR}/asc/asc-tools/cpudebug/include/
            -I${TOP_DIR}/asc/asc-tools/cpudebug/model/cpulib/
            -I${TOP_DIR}/ace/comop/inc/external/hccl/
            -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
            -I${TILING_GEN_DIR}
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all -DONE_CORE_DUMP_SIZE=1024
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMAND_EXPAND_LISTS VERBATIM
            DEPENDS kernel_tiling_headers
    )

    add_custom_target(hccl_ag_superkernel_mesh_1d_int64
            COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_ag_mesh_1d_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_mesh_1d_int64.o
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0
            -DASCENDC_DUMP --cce-aicore-arch=dav-c310-vec
            --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-dcci-insert-for-scalar=false
            -mllvm -cce-aicore-stack-size=0x8000 -D__CCE_AICORE__=310 -D__DAV_C310__
            -DHCCL_DTYPE_INT64 -Dsk_allgather_mesh_1d=sk_allgather_mesh_1d_int64
            -I${TOP_DIR}/asc/asc-devkit/
            -I${TOP_DIR}/asc/asc-devkit/include
            -I${TOP_DIR}/asc/asc-devkit/impl
            -I${TOP_DIR}/asc/asc-devkit/include/basic_api/
            -I${TOP_DIR}/asc/asc-devkit/include/micro_api/
            -I${TOP_DIR}/asc/asc-devkit/include/simt_api/
            -I${TOP_DIR}/asc/asc-devkit/impl/basic_api/
            -I${TOP_DIR}/asc/asc-devkit/impl/micro_api/
            -I${TOP_DIR}/asc/asc-devkit/impl/simt_api/
            -I${TOP_DIR}/asc/asc-tools/cpudebug/include/
            -I${TOP_DIR}/asc/asc-tools/cpudebug/model/cpulib/
            -I${TOP_DIR}/ace/comop/inc/external/hccl/
            -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
            -I${TILING_GEN_DIR}
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all -DONE_CORE_DUMP_SIZE=1024
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMAND_EXPAND_LISTS VERBATIM
            DEPENDS kernel_tiling_headers
    )

    add_custom_target(hccl_ag_superkernel_mesh_1d_uint64
            COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_ag_mesh_1d_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_mesh_1d_uint64.o
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0
            -DASCENDC_DUMP --cce-aicore-arch=dav-c310-vec
            --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-dcci-insert-for-scalar=false
            -mllvm -cce-aicore-stack-size=0x8000 -D__CCE_AICORE__=310 -D__DAV_C310__
            -DHCCL_DTYPE_UINT64 -Dsk_allgather_mesh_1d=sk_allgather_mesh_1d_uint64
            -I${TOP_DIR}/asc/asc-devkit/
            -I${TOP_DIR}/asc/asc-devkit/include
            -I${TOP_DIR}/asc/asc-devkit/impl
            -I${TOP_DIR}/asc/asc-devkit/include/basic_api/
            -I${TOP_DIR}/asc/asc-devkit/include/micro_api/
            -I${TOP_DIR}/asc/asc-devkit/include/simt_api/
            -I${TOP_DIR}/asc/asc-devkit/impl/basic_api/
            -I${TOP_DIR}/asc/asc-devkit/impl/micro_api/
            -I${TOP_DIR}/asc/asc-devkit/impl/simt_api/
            -I${TOP_DIR}/asc/asc-tools/cpudebug/include/
            -I${TOP_DIR}/asc/asc-tools/cpudebug/model/cpulib/
            -I${TOP_DIR}/ace/comop/inc/external/hccl/
            -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
            -I${TILING_GEN_DIR}
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all -DONE_CORE_DUMP_SIZE=1024
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMAND_EXPAND_LISTS VERBATIM
            DEPENDS kernel_tiling_headers
    )

    add_custom_target(hccl_a2a_superkernel_mesh_1d_int8
            COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_a2a_mesh_1d_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_a2a_superkernel_mesh_1d_int8.o
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0
            -DASCENDC_DUMP --cce-aicore-arch=dav-c310-vec
            --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-dcci-insert-for-scalar=false
            -mllvm -cce-aicore-stack-size=0x8000 -D__CCE_AICORE__=310 -D__DAV_C310__
            -DHCCL_DTYPE_INT8 -Dsk_alltoall_mesh_1d=sk_alltoall_mesh_1d_int8
            -I${TOP_DIR}/asc/asc-devkit/
            -I${TOP_DIR}/asc/asc-devkit/include
            -I${TOP_DIR}/asc/asc-devkit/impl
            -I${TOP_DIR}/asc/asc-devkit/include/basic_api/
            -I${TOP_DIR}/asc/asc-devkit/include/micro_api/
            -I${TOP_DIR}/asc/asc-devkit/include/simt_api/
            -I${TOP_DIR}/asc/asc-devkit/impl/basic_api/
            -I${TOP_DIR}/asc/asc-devkit/impl/micro_api/
            -I${TOP_DIR}/asc/asc-devkit/impl/simt_api/
            -I${TOP_DIR}/asc/asc-tools/cpudebug/include/
            -I${TOP_DIR}/asc/asc-tools/cpudebug/model/cpulib/
            -I${TOP_DIR}/ace/comop/inc/external/hccl/
            -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
            -I${TILING_GEN_DIR}
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all -DONE_CORE_DUMP_SIZE=1024
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMAND_EXPAND_LISTS VERBATIM
            DEPENDS kernel_tiling_headers
    )

    add_custom_target(hccl_a2a_superkernel_mesh_1d_int16
            COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_a2a_mesh_1d_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_a2a_superkernel_mesh_1d_int16.o
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0
            -DASCENDC_DUMP --cce-aicore-arch=dav-c310-vec
            --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-dcci-insert-for-scalar=false
            -mllvm -cce-aicore-stack-size=0x8000 -D__CCE_AICORE__=310 -D__DAV_C310__
            -DHCCL_DTYPE_INT16 -Dsk_alltoall_mesh_1d=sk_alltoall_mesh_1d_int16
            -I${TOP_DIR}/asc/asc-devkit/
            -I${TOP_DIR}/asc/asc-devkit/include
            -I${TOP_DIR}/asc/asc-devkit/impl
            -I${TOP_DIR}/asc/asc-devkit/include/basic_api/
            -I${TOP_DIR}/asc/asc-devkit/include/micro_api/
            -I${TOP_DIR}/asc/asc-devkit/include/simt_api/
            -I${TOP_DIR}/asc/asc-devkit/impl/basic_api/
            -I${TOP_DIR}/asc/asc-devkit/impl/micro_api/
            -I${TOP_DIR}/asc/asc-devkit/impl/simt_api/
            -I${TOP_DIR}/asc/asc-tools/cpudebug/include/
            -I${TOP_DIR}/asc/asc-tools/cpudebug/model/cpulib/
            -I${TOP_DIR}/ace/comop/inc/external/hccl/
            -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
            -I${TILING_GEN_DIR}
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all -DONE_CORE_DUMP_SIZE=1024
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMAND_EXPAND_LISTS VERBATIM
            DEPENDS kernel_tiling_headers
    )

    add_custom_target(hccl_a2a_superkernel_mesh_1d_int32
            COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_a2a_mesh_1d_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_a2a_superkernel_mesh_1d_int32.o
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0
            -DASCENDC_DUMP --cce-aicore-arch=dav-c310-vec
            --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-dcci-insert-for-scalar=false
            -mllvm -cce-aicore-stack-size=0x8000 -D__CCE_AICORE__=310 -D__DAV_C310__
            -DHCCL_DTYPE_INT32 -Dsk_alltoall_mesh_1d=sk_alltoall_mesh_1d_int32
            -I${TOP_DIR}/asc/asc-devkit/
            -I${TOP_DIR}/asc/asc-devkit/include
            -I${TOP_DIR}/asc/asc-devkit/impl
            -I${TOP_DIR}/asc/asc-devkit/include/basic_api/
            -I${TOP_DIR}/asc/asc-devkit/include/micro_api/
            -I${TOP_DIR}/asc/asc-devkit/include/simt_api/
            -I${TOP_DIR}/asc/asc-devkit/impl/basic_api/
            -I${TOP_DIR}/asc/asc-devkit/impl/micro_api/
            -I${TOP_DIR}/asc/asc-devkit/impl/simt_api/
            -I${TOP_DIR}/asc/asc-tools/cpudebug/include/
            -I${TOP_DIR}/asc/asc-tools/cpudebug/model/cpulib/
            -I${TOP_DIR}/ace/comop/inc/external/hccl/
            -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
            -I${TILING_GEN_DIR}
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all -DONE_CORE_DUMP_SIZE=1024
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMAND_EXPAND_LISTS VERBATIM
            DEPENDS kernel_tiling_headers
    )

    add_custom_target(hccl_a2a_superkernel_mesh_1d_float16
            COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_a2a_mesh_1d_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_a2a_superkernel_mesh_1d_float16.o
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0
            -DASCENDC_DUMP --cce-aicore-arch=dav-c310-vec
            --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-dcci-insert-for-scalar=false
            -mllvm -cce-aicore-stack-size=0x8000 -D__CCE_AICORE__=310 -D__DAV_C310__
            -DHCCL_DTYPE_FP16 -Dsk_alltoall_mesh_1d=sk_alltoall_mesh_1d_float16
            -I${TOP_DIR}/asc/asc-devkit/
            -I${TOP_DIR}/asc/asc-devkit/include
            -I${TOP_DIR}/asc/asc-devkit/impl
            -I${TOP_DIR}/asc/asc-devkit/include/basic_api/
            -I${TOP_DIR}/asc/asc-devkit/include/micro_api/
            -I${TOP_DIR}/asc/asc-devkit/include/simt_api/
            -I${TOP_DIR}/asc/asc-devkit/impl/basic_api/
            -I${TOP_DIR}/asc/asc-devkit/impl/micro_api/
            -I${TOP_DIR}/asc/asc-devkit/impl/simt_api/
            -I${TOP_DIR}/asc/asc-tools/cpudebug/include/
            -I${TOP_DIR}/asc/asc-tools/cpudebug/model/cpulib/
            -I${TOP_DIR}/ace/comop/inc/external/hccl/
            -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
            -I${TILING_GEN_DIR}
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all -DONE_CORE_DUMP_SIZE=1024
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMAND_EXPAND_LISTS VERBATIM
            DEPENDS kernel_tiling_headers
    )

    add_custom_target(hccl_a2a_superkernel_mesh_1d_float32
            COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_a2a_mesh_1d_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_a2a_superkernel_mesh_1d_float32.o
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0
            -DASCENDC_DUMP --cce-aicore-arch=dav-c310-vec
            --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-dcci-insert-for-scalar=false
            -mllvm -cce-aicore-stack-size=0x8000 -D__CCE_AICORE__=310 -D__DAV_C310__
            -DHCCL_DTYPE_FP32 -Dsk_alltoall_mesh_1d=sk_alltoall_mesh_1d_float32
            -I${TOP_DIR}/asc/asc-devkit/
            -I${TOP_DIR}/asc/asc-devkit/include
            -I${TOP_DIR}/asc/asc-devkit/impl
            -I${TOP_DIR}/asc/asc-devkit/include/basic_api/
            -I${TOP_DIR}/asc/asc-devkit/include/micro_api/
            -I${TOP_DIR}/asc/asc-devkit/include/simt_api/
            -I${TOP_DIR}/asc/asc-devkit/impl/basic_api/
            -I${TOP_DIR}/asc/asc-devkit/impl/micro_api/
            -I${TOP_DIR}/asc/asc-devkit/impl/simt_api/
            -I${TOP_DIR}/asc/asc-tools/cpudebug/include/
            -I${TOP_DIR}/asc/asc-tools/cpudebug/model/cpulib/
            -I${TOP_DIR}/ace/comop/inc/external/hccl/
            -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
            -I${TILING_GEN_DIR}
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all -DONE_CORE_DUMP_SIZE=1024
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMAND_EXPAND_LISTS VERBATIM
            DEPENDS kernel_tiling_headers
    )

    add_custom_target(hccl_a2a_superkernel_mesh_1d_bfloat16
            COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_a2a_mesh_1d_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_a2a_superkernel_mesh_1d_bfloat16.o
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0
            -DASCENDC_DUMP --cce-aicore-arch=dav-c310-vec
            --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-dcci-insert-for-scalar=false
            -mllvm -cce-aicore-stack-size=0x8000 -D__CCE_AICORE__=310 -D__DAV_C310__
            -DHCCL_DTYPE_BFP16 -Dsk_alltoall_mesh_1d=sk_alltoall_mesh_1d_bfloat16
            -I${TOP_DIR}/asc/asc-devkit/
            -I${TOP_DIR}/asc/asc-devkit/include
            -I${TOP_DIR}/asc/asc-devkit/impl
            -I${TOP_DIR}/asc/asc-devkit/include/basic_api/
            -I${TOP_DIR}/asc/asc-devkit/include/micro_api/
            -I${TOP_DIR}/asc/asc-devkit/include/simt_api/
            -I${TOP_DIR}/asc/asc-devkit/impl/basic_api/
            -I${TOP_DIR}/asc/asc-devkit/impl/micro_api/
            -I${TOP_DIR}/asc/asc-devkit/impl/simt_api/
            -I${TOP_DIR}/asc/asc-tools/cpudebug/include/
            -I${TOP_DIR}/asc/asc-tools/cpudebug/model/cpulib/
            -I${TOP_DIR}/ace/comop/inc/external/hccl/
            -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
            -I${TILING_GEN_DIR}
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all -DONE_CORE_DUMP_SIZE=1024
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMAND_EXPAND_LISTS VERBATIM
            DEPENDS kernel_tiling_headers
    )

    add_custom_target(hccl_a2a_superkernel_mesh_1d_uint8
            COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_a2a_mesh_1d_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_a2a_superkernel_mesh_1d_uint8.o
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0
            -DASCENDC_DUMP --cce-aicore-arch=dav-c310-vec
            --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-dcci-insert-for-scalar=false
            -mllvm -cce-aicore-stack-size=0x8000 -D__CCE_AICORE__=310 -D__DAV_C310__
            -DHCCL_DTYPE_UINT8 -Dsk_alltoall_mesh_1d=sk_alltoall_mesh_1d_uint8
            -I${TOP_DIR}/asc/asc-devkit/
            -I${TOP_DIR}/asc/asc-devkit/include
            -I${TOP_DIR}/asc/asc-devkit/impl
            -I${TOP_DIR}/asc/asc-devkit/include/basic_api/
            -I${TOP_DIR}/asc/asc-devkit/include/micro_api/
            -I${TOP_DIR}/asc/asc-devkit/include/simt_api/
            -I${TOP_DIR}/asc/asc-devkit/impl/basic_api/
            -I${TOP_DIR}/asc/asc-devkit/impl/micro_api/
            -I${TOP_DIR}/asc/asc-devkit/impl/simt_api/
            -I${TOP_DIR}/asc/asc-tools/cpudebug/include/
            -I${TOP_DIR}/asc/asc-tools/cpudebug/model/cpulib/
            -I${TOP_DIR}/ace/comop/inc/external/hccl/
            -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
            -I${TILING_GEN_DIR}
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all -DONE_CORE_DUMP_SIZE=1024
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMAND_EXPAND_LISTS VERBATIM
            DEPENDS kernel_tiling_headers
    )

    add_custom_target(hccl_a2a_superkernel_mesh_1d_uint16
            COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_a2a_mesh_1d_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_a2a_superkernel_mesh_1d_uint16.o
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0
            -DASCENDC_DUMP --cce-aicore-arch=dav-c310-vec
            --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-dcci-insert-for-scalar=false
            -mllvm -cce-aicore-stack-size=0x8000 -D__CCE_AICORE__=310 -D__DAV_C310__
            -DHCCL_DTYPE_UINT16 -Dsk_alltoall_mesh_1d=sk_alltoall_mesh_1d_uint16
            -I${TOP_DIR}/asc/asc-devkit/
            -I${TOP_DIR}/asc/asc-devkit/include
            -I${TOP_DIR}/asc/asc-devkit/impl
            -I${TOP_DIR}/asc/asc-devkit/include/basic_api/
            -I${TOP_DIR}/asc/asc-devkit/include/micro_api/
            -I${TOP_DIR}/asc/asc-devkit/include/simt_api/
            -I${TOP_DIR}/asc/asc-devkit/impl/basic_api/
            -I${TOP_DIR}/asc/asc-devkit/impl/micro_api/
            -I${TOP_DIR}/asc/asc-devkit/impl/simt_api/
            -I${TOP_DIR}/asc/asc-tools/cpudebug/include/
            -I${TOP_DIR}/asc/asc-tools/cpudebug/model/cpulib/
            -I${TOP_DIR}/ace/comop/inc/external/hccl/
            -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
            -I${TILING_GEN_DIR}
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all -DONE_CORE_DUMP_SIZE=1024
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMAND_EXPAND_LISTS VERBATIM
            DEPENDS kernel_tiling_headers
    )

    add_custom_target(hccl_a2a_superkernel_mesh_1d_uint32
            COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_a2a_mesh_1d_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_a2a_superkernel_mesh_1d_uint32.o
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0
            -DASCENDC_DUMP --cce-aicore-arch=dav-c310-vec
            --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-dcci-insert-for-scalar=false
            -mllvm -cce-aicore-stack-size=0x8000 -D__CCE_AICORE__=310 -D__DAV_C310__
            -DHCCL_DTYPE_UINT32 -Dsk_alltoall_mesh_1d=sk_alltoall_mesh_1d_uint32
            -I${TOP_DIR}/asc/asc-devkit/
            -I${TOP_DIR}/asc/asc-devkit/include
            -I${TOP_DIR}/asc/asc-devkit/impl
            -I${TOP_DIR}/asc/asc-devkit/include/basic_api/
            -I${TOP_DIR}/asc/asc-devkit/include/micro_api/
            -I${TOP_DIR}/asc/asc-devkit/include/simt_api/
            -I${TOP_DIR}/asc/asc-devkit/impl/basic_api/
            -I${TOP_DIR}/asc/asc-devkit/impl/micro_api/
            -I${TOP_DIR}/asc/asc-devkit/impl/simt_api/
            -I${TOP_DIR}/asc/asc-tools/cpudebug/include/
            -I${TOP_DIR}/asc/asc-tools/cpudebug/model/cpulib/
            -I${TOP_DIR}/ace/comop/inc/external/hccl/
            -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
            -I${TILING_GEN_DIR}
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all -DONE_CORE_DUMP_SIZE=1024
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMAND_EXPAND_LISTS VERBATIM
            DEPENDS kernel_tiling_headers
    )

    add_custom_target(hccl_a2a_superkernel_mesh_1d_int64
            COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_a2a_mesh_1d_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_a2a_superkernel_mesh_1d_int64.o
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0
            -DASCENDC_DUMP --cce-aicore-arch=dav-c310-vec
            --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-dcci-insert-for-scalar=false
            -mllvm -cce-aicore-stack-size=0x8000 -D__CCE_AICORE__=310 -D__DAV_C310__
            -DHCCL_DTYPE_INT64 -Dsk_alltoall_mesh_1d=sk_alltoall_mesh_1d_int64
            -I${TOP_DIR}/asc/asc-devkit/
            -I${TOP_DIR}/asc/asc-devkit/include
            -I${TOP_DIR}/asc/asc-devkit/impl
            -I${TOP_DIR}/asc/asc-devkit/include/basic_api/
            -I${TOP_DIR}/asc/asc-devkit/include/micro_api/
            -I${TOP_DIR}/asc/asc-devkit/include/simt_api/
            -I${TOP_DIR}/asc/asc-devkit/impl/basic_api/
            -I${TOP_DIR}/asc/asc-devkit/impl/micro_api/
            -I${TOP_DIR}/asc/asc-devkit/impl/simt_api/
            -I${TOP_DIR}/asc/asc-tools/cpudebug/include/
            -I${TOP_DIR}/asc/asc-tools/cpudebug/model/cpulib/
            -I${TOP_DIR}/ace/comop/inc/external/hccl/
            -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
            -I${TILING_GEN_DIR}
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all -DONE_CORE_DUMP_SIZE=1024
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMAND_EXPAND_LISTS VERBATIM
            DEPENDS kernel_tiling_headers
    )

    add_custom_target(hccl_a2a_superkernel_mesh_1d_uint64
            COMMAND ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_a2a_mesh_1d_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_a2a_superkernel_mesh_1d_uint64.o
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0
            -DASCENDC_DUMP --cce-aicore-arch=dav-c310-vec
            --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-dcci-insert-for-scalar=false
            -mllvm -cce-aicore-stack-size=0x8000 -D__CCE_AICORE__=310 -D__DAV_C310__
            -DHCCL_DTYPE_UINT64 -Dsk_alltoall_mesh_1d=sk_alltoall_mesh_1d_uint64
            -I${TOP_DIR}/asc/asc-devkit/
            -I${TOP_DIR}/asc/asc-devkit/include
            -I${TOP_DIR}/asc/asc-devkit/impl
            -I${TOP_DIR}/asc/asc-devkit/include/basic_api/
            -I${TOP_DIR}/asc/asc-devkit/include/micro_api/
            -I${TOP_DIR}/asc/asc-devkit/include/simt_api/
            -I${TOP_DIR}/asc/asc-devkit/impl/basic_api/
            -I${TOP_DIR}/asc/asc-devkit/impl/micro_api/
            -I${TOP_DIR}/asc/asc-devkit/impl/simt_api/
            -I${TOP_DIR}/asc/asc-tools/cpudebug/include/
            -I${TOP_DIR}/asc/asc-tools/cpudebug/model/cpulib/
            -I${TOP_DIR}/ace/comop/inc/external/hccl/
            -I${CMAKE_CURRENT_SOURCE_DIR}/aiv_interface/
            -I${TILING_GEN_DIR}
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all -DONE_CORE_DUMP_SIZE=1024
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMAND_EXPAND_LISTS VERBATIM
            DEPENDS kernel_tiling_headers
    )
 
endif()
 
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/hccl_aiv_op_910_95.o ${INSTALL_OPTIONAL}
    COMPONENT hcomm
    DESTINATION ${INSTALL_LIBRARY_DIR}
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/hccl_ar_superkernel_mesh_1d_oneshot_int8.o ${INSTALL_OPTIONAL}
    COMPONENT hcomm
    DESTINATION ${INSTALL_LIBRARY_DIR}
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/hccl_ar_superkernel_mesh_1d_oneshot_int16.o ${INSTALL_OPTIONAL}
    COMPONENT hcomm
    DESTINATION ${INSTALL_LIBRARY_DIR}
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/hccl_ar_superkernel_mesh_1d_oneshot_int32.o ${INSTALL_OPTIONAL}
    COMPONENT hcomm
    DESTINATION ${INSTALL_LIBRARY_DIR}
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/hccl_ar_superkernel_mesh_1d_oneshot_float16.o ${INSTALL_OPTIONAL}
    COMPONENT hcomm
    DESTINATION ${INSTALL_LIBRARY_DIR}
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/hccl_ar_superkernel_mesh_1d_oneshot_float32.o ${INSTALL_OPTIONAL}
    COMPONENT hcomm
    DESTINATION ${INSTALL_LIBRARY_DIR}
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/hccl_ar_superkernel_mesh_1d_oneshot_bfloat16.o ${INSTALL_OPTIONAL}
    COMPONENT hcomm
    DESTINATION ${INSTALL_LIBRARY_DIR}
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/hccl_ar_superkernel_mesh_1d_twoshot_int8.o ${INSTALL_OPTIONAL}
    COMPONENT hcomm
    DESTINATION ${INSTALL_LIBRARY_DIR}
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/hccl_ar_superkernel_mesh_1d_twoshot_int16.o ${INSTALL_OPTIONAL}
    COMPONENT hcomm
    DESTINATION ${INSTALL_LIBRARY_DIR}
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/hccl_ar_superkernel_mesh_1d_twoshot_int32.o ${INSTALL_OPTIONAL}
    COMPONENT hcomm
    DESTINATION ${INSTALL_LIBRARY_DIR}
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/hccl_ar_superkernel_mesh_1d_twoshot_float16.o ${INSTALL_OPTIONAL}
    COMPONENT hcomm
    DESTINATION ${INSTALL_LIBRARY_DIR}
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/hccl_ar_superkernel_mesh_1d_twoshot_float32.o ${INSTALL_OPTIONAL}
    COMPONENT hcomm
    DESTINATION ${INSTALL_LIBRARY_DIR}
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/hccl_ar_superkernel_mesh_1d_twoshot_bfloat16.o ${INSTALL_OPTIONAL}
    COMPONENT hcomm
    DESTINATION ${INSTALL_LIBRARY_DIR}
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/hccl_rs_superkernel_mesh_1d_int8.o ${INSTALL_OPTIONAL}
    COMPONENT hcomm
    DESTINATION ${INSTALL_LIBRARY_DIR}
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/hccl_rs_superkernel_mesh_1d_int16.o ${INSTALL_OPTIONAL}
    COMPONENT hcomm
    DESTINATION ${INSTALL_LIBRARY_DIR}
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/hccl_rs_superkernel_mesh_1d_int32.o ${INSTALL_OPTIONAL}
    COMPONENT hcomm
    DESTINATION ${INSTALL_LIBRARY_DIR}
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/hccl_rs_superkernel_mesh_1d_float16.o ${INSTALL_OPTIONAL}
    COMPONENT hcomm
    DESTINATION ${INSTALL_LIBRARY_DIR}
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/hccl_rs_superkernel_mesh_1d_float32.o ${INSTALL_OPTIONAL}
    COMPONENT hcomm
    DESTINATION ${INSTALL_LIBRARY_DIR}
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/hccl_rs_superkernel_mesh_1d_bfloat16.o ${INSTALL_OPTIONAL}
    COMPONENT hcomm
    DESTINATION ${INSTALL_LIBRARY_DIR}
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_mesh_1d_int8.o ${INSTALL_OPTIONAL}
    COMPONENT hcomm
    DESTINATION ${INSTALL_LIBRARY_DIR}
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_mesh_1d_int16.o ${INSTALL_OPTIONAL}
    COMPONENT hcomm
    DESTINATION ${INSTALL_LIBRARY_DIR}
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_mesh_1d_int32.o ${INSTALL_OPTIONAL}
    COMPONENT hcomm
    DESTINATION ${INSTALL_LIBRARY_DIR}
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_mesh_1d_float16.o ${INSTALL_OPTIONAL}
    COMPONENT hcomm
    DESTINATION ${INSTALL_LIBRARY_DIR}
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_mesh_1d_float32.o ${INSTALL_OPTIONAL}
    COMPONENT hcomm
    DESTINATION ${INSTALL_LIBRARY_DIR}
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_mesh_1d_bfloat16.o ${INSTALL_OPTIONAL}
    COMPONENT hcomm
    DESTINATION ${INSTALL_LIBRARY_DIR}
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_mesh_1d_uint8.o ${INSTALL_OPTIONAL}
    COMPONENT hcomm
    DESTINATION ${INSTALL_LIBRARY_DIR}
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_mesh_1d_uint16.o ${INSTALL_OPTIONAL}
    COMPONENT hcomm
    DESTINATION ${INSTALL_LIBRARY_DIR}
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_mesh_1d_uint32.o ${INSTALL_OPTIONAL}
    COMPONENT hcomm
    DESTINATION ${INSTALL_LIBRARY_DIR}
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_mesh_1d_int64.o ${INSTALL_OPTIONAL}
    COMPONENT hcomm
    DESTINATION ${INSTALL_LIBRARY_DIR}
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/hccl_ag_superkernel_mesh_1d_uint64.o ${INSTALL_OPTIONAL}
    COMPONENT hcomm
    DESTINATION ${INSTALL_LIBRARY_DIR}
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/hccl_a2a_superkernel_mesh_1d_int8.o ${INSTALL_OPTIONAL}
    COMPONENT hcomm
    DESTINATION ${INSTALL_LIBRARY_DIR}
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/hccl_a2a_superkernel_mesh_1d_int16.o ${INSTALL_OPTIONAL}
    COMPONENT hcomm
    DESTINATION ${INSTALL_LIBRARY_DIR}
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/hccl_a2a_superkernel_mesh_1d_int32.o ${INSTALL_OPTIONAL}
    COMPONENT hcomm
    DESTINATION ${INSTALL_LIBRARY_DIR}
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/hccl_a2a_superkernel_mesh_1d_float16.o ${INSTALL_OPTIONAL}
    COMPONENT hcomm
    DESTINATION ${INSTALL_LIBRARY_DIR}
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/hccl_a2a_superkernel_mesh_1d_float32.o ${INSTALL_OPTIONAL}
    COMPONENT hcomm
    DESTINATION ${INSTALL_LIBRARY_DIR}
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/hccl_a2a_superkernel_mesh_1d_bfloat16.o ${INSTALL_OPTIONAL}
    COMPONENT hcomm
    DESTINATION ${INSTALL_LIBRARY_DIR}
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/hccl_a2a_superkernel_mesh_1d_uint8.o ${INSTALL_OPTIONAL}
    COMPONENT hcomm
    DESTINATION ${INSTALL_LIBRARY_DIR}
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/hccl_a2a_superkernel_mesh_1d_uint16.o ${INSTALL_OPTIONAL}
    COMPONENT hcomm
    DESTINATION ${INSTALL_LIBRARY_DIR}
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/hccl_a2a_superkernel_mesh_1d_uint32.o ${INSTALL_OPTIONAL}
    COMPONENT hcomm
    DESTINATION ${INSTALL_LIBRARY_DIR}
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/hccl_a2a_superkernel_mesh_1d_int64.o ${INSTALL_OPTIONAL}
    COMPONENT hcomm
    DESTINATION ${INSTALL_LIBRARY_DIR}
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/hccl_a2a_superkernel_mesh_1d_uint64.o ${INSTALL_OPTIONAL}
    COMPONENT hcomm
    DESTINATION ${INSTALL_LIBRARY_DIR}
)