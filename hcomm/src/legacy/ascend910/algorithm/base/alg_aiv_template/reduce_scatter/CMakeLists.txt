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
    aiv_reduce_scatter_op.h PROPERTIES LANGUAGE CXX
)

if(CCACHE_PROGRAM)
    set(BISHENG_CCACHE ${CCACHE_PROGRAM})
endif()
if(BUILD_OPEN_PROJECT)
    set(CCE_DIR ${ASCEND_CANN_PACKAGE_PATH}/tools/ccec_compiler/bin)
    add_custom_target(hccl_aiv_reduce_scatter_op ALL
        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/hccl_aiv_reduce_scatter_op_tmp.o ${CMAKE_CURRENT_BINARY_DIR}/hccl_aiv_reduce_scatter_op_sk1.o ${CMAKE_CURRENT_BINARY_DIR}/hccl_aiv_reduce_scatter_op_sk2.o ${CMAKE_CURRENT_BINARY_DIR}/hccl_aiv_reduce_scatter_op_sk3.o ${CMAKE_CURRENT_BINARY_DIR}/hccl_aiv_reduce_scatter_op_sk4.o
        COMMAND ${CCE_DIR}/ld.lld -m aicorelinux -Ttext=0 ${CMAKE_CURRENT_BINARY_DIR}/hccl_aiv_reduce_scatter_op_tmp.o ${CMAKE_CURRENT_BINARY_DIR}/hccl_aiv_reduce_scatter_op_sk1.o ${CMAKE_CURRENT_BINARY_DIR}/hccl_aiv_reduce_scatter_op_sk2.o ${CMAKE_CURRENT_BINARY_DIR}/hccl_aiv_reduce_scatter_op_sk3.o ${CMAKE_CURRENT_BINARY_DIR}/hccl_aiv_reduce_scatter_op_sk4.o -static -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_aiv_reduce_scatter_op.o
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/hccl_aiv_reduce_scatter_op_tmp.o
            COMMAND ${BISHENG_CCACHE} ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_reduce_scatter_op.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_aiv_reduce_scatter_op_tmp.o 
            -DASCENDC_DUMP=${ASCENDC_DUMP} -DTILING_KEY_VAR=0 --cce-aicore-arch=dav-c220-vec
            --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-dcci-insert-for-scalar=false
            -mllvm -cce-aicore-stack-size=0x8000
            "-I$<JOIN:$<TARGET_PROPERTY:asc_kernel_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
            "-I$<JOIN:$<TARGET_PROPERTY:kernel_tiling_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
            -I${CMAKE_CURRENT_SOURCE_DIR}/../
            -I${CMAKE_CURRENT_SOURCE_DIR}/../aiv_interface/
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all -DONE_CORE_DUMP_SIZE=1024
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMAND_EXPAND_LISTS VERBATIM
            DEPENDS kernel_tiling_headers
    )

    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/hccl_aiv_reduce_scatter_op_sk1.o
            COMMAND ${BISHENG_CCACHE} ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_reduce_scatter_op.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_aiv_reduce_scatter_op_sk1.o 
            -DASCENDC_DUMP=${ASCENDC_DUMP} -DTILING_KEY_VAR=0 -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DBUILD_SK_FUNC -DSK_FUNC_ID=1 --cce-aicore-arch=dav-c220-vec
            --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-dcci-insert-for-scalar=false
            -mllvm -cce-aicore-stack-size=0x8000
            "-I$<JOIN:$<TARGET_PROPERTY:asc_kernel_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
            "-I$<JOIN:$<TARGET_PROPERTY:kernel_tiling_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
            -I${CMAKE_CURRENT_SOURCE_DIR}/../
            -I${CMAKE_CURRENT_SOURCE_DIR}/../aiv_interface/
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all -DONE_CORE_DUMP_SIZE=1024
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMAND_EXPAND_LISTS VERBATIM
            DEPENDS kernel_tiling_headers
    )

    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/hccl_aiv_reduce_scatter_op_sk2.o
            COMMAND ${BISHENG_CCACHE} ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_reduce_scatter_op.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_aiv_reduce_scatter_op_sk2.o 
            -DASCENDC_DUMP=${ASCENDC_DUMP} -DTILING_KEY_VAR=0 -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DBUILD_SK_FUNC -DSK_FUNC_ID=2 --cce-aicore-arch=dav-c220-vec
            --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-dcci-insert-for-scalar=false
            -mllvm -cce-aicore-stack-size=0x8000
            "-I$<JOIN:$<TARGET_PROPERTY:asc_kernel_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
            "-I$<JOIN:$<TARGET_PROPERTY:kernel_tiling_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
            -I${CMAKE_CURRENT_SOURCE_DIR}/../
            -I${CMAKE_CURRENT_SOURCE_DIR}/../aiv_interface/
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all -DONE_CORE_DUMP_SIZE=1024
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMAND_EXPAND_LISTS VERBATIM
            DEPENDS kernel_tiling_headers
    )

    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/hccl_aiv_reduce_scatter_op_sk3.o
            COMMAND ${BISHENG_CCACHE} ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_reduce_scatter_op.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_aiv_reduce_scatter_op_sk3.o 
            -DASCENDC_DUMP=${ASCENDC_DUMP} -DTILING_KEY_VAR=0 -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DBUILD_SK_FUNC -DSK_FUNC_ID=3 --cce-aicore-arch=dav-c220-vec
            --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-dcci-insert-for-scalar=false
            -mllvm -cce-aicore-stack-size=0x8000
            "-I$<JOIN:$<TARGET_PROPERTY:asc_kernel_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
            "-I$<JOIN:$<TARGET_PROPERTY:kernel_tiling_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
            -I${CMAKE_CURRENT_SOURCE_DIR}/../
            -I${CMAKE_CURRENT_SOURCE_DIR}/../aiv_interface/
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all -DONE_CORE_DUMP_SIZE=1024
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMAND_EXPAND_LISTS VERBATIM
            DEPENDS kernel_tiling_headers
    )

    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/hccl_aiv_reduce_scatter_op_sk4.o
            COMMAND ${BISHENG_CCACHE} ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_reduce_scatter_op.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_aiv_reduce_scatter_op_sk4.o 
            -DASCENDC_DUMP=${ASCENDC_DUMP} -DTILING_KEY_VAR=0 -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DBUILD_SK_FUNC -DSK_FUNC_ID=4 --cce-aicore-arch=dav-c220-vec
            --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-dcci-insert-for-scalar=false
            -mllvm -cce-aicore-stack-size=0x8000
            "-I$<JOIN:$<TARGET_PROPERTY:asc_kernel_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
            "-I$<JOIN:$<TARGET_PROPERTY:kernel_tiling_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
            -I${CMAKE_CURRENT_SOURCE_DIR}/../
            -I${CMAKE_CURRENT_SOURCE_DIR}/../aiv_interface/
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all -DONE_CORE_DUMP_SIZE=1024
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMAND_EXPAND_LISTS VERBATIM
            DEPENDS kernel_tiling_headers
    )

    add_custom_target(hccl_rs_superkernel_int8 ALL
        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/hccl_rs_superkernel_int8.o)
    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/hccl_rs_superkernel_int8.o
            COMMAND ${BISHENG_CCACHE} ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_rs_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_rs_superkernel_int8.o --cce-aicore-arch=dav-c220-vec
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DASCENDC_DUMP=0 -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0 -DHCCL_DTYPE_INT8 -Dsk_reducescatter=sk_reducescatter_int8
            --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-stack-size=0x8000
            "-I$<JOIN:$<TARGET_PROPERTY:asc_kernel_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
            "-I$<JOIN:$<TARGET_PROPERTY:kernel_tiling_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
            -I${CMAKE_CURRENT_SOURCE_DIR}/../
            -I${CMAKE_CURRENT_SOURCE_DIR}/../aiv_interface/
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMAND_EXPAND_LISTS VERBATIM
            DEPENDS kernel_tiling_headers
    )

    add_custom_target(hccl_rs_superkernel_int16 ALL
        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/hccl_rs_superkernel_int16.o)
    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/hccl_rs_superkernel_int16.o
            COMMAND ${BISHENG_CCACHE} ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_rs_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_rs_superkernel_int16.o --cce-aicore-arch=dav-c220-vec
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DASCENDC_DUMP=0 -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0 -DHCCL_DTYPE_INT16 -Dsk_reducescatter=sk_reducescatter_int16
            --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-stack-size=0x8000
            "-I$<JOIN:$<TARGET_PROPERTY:asc_kernel_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
            "-I$<JOIN:$<TARGET_PROPERTY:kernel_tiling_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
            -I${CMAKE_CURRENT_SOURCE_DIR}/../
            -I${CMAKE_CURRENT_SOURCE_DIR}/../aiv_interface/
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMAND_EXPAND_LISTS VERBATIM
            DEPENDS kernel_tiling_headers
    )

    add_custom_target(hccl_rs_superkernel_int32 ALL
        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/hccl_rs_superkernel_int32.o)
    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/hccl_rs_superkernel_int32.o
            COMMAND ${BISHENG_CCACHE} ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_rs_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_rs_superkernel_int32.o --cce-aicore-arch=dav-c220-vec
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DASCENDC_DUMP=0 -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0 -DHCCL_DTYPE_INT32 -Dsk_reducescatter=sk_reducescatter_int32
            --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-stack-size=0x8000
            "-I$<JOIN:$<TARGET_PROPERTY:asc_kernel_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
            "-I$<JOIN:$<TARGET_PROPERTY:kernel_tiling_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
            -I${CMAKE_CURRENT_SOURCE_DIR}/../
            -I${CMAKE_CURRENT_SOURCE_DIR}/../aiv_interface/
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMAND_EXPAND_LISTS VERBATIM
            DEPENDS kernel_tiling_headers
    )

    add_custom_target(hccl_rs_superkernel_float16 ALL
        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/hccl_rs_superkernel_float16.o)
    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/hccl_rs_superkernel_float16.o
            COMMAND ${BISHENG_CCACHE} ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_rs_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_rs_superkernel_float16.o --cce-aicore-arch=dav-c220-vec
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DASCENDC_DUMP=0 -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0 -DHCCL_DTYPE_FP16 -Dsk_reducescatter=sk_reducescatter_float16
            --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-stack-size=0x8000
            "-I$<JOIN:$<TARGET_PROPERTY:asc_kernel_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
            "-I$<JOIN:$<TARGET_PROPERTY:kernel_tiling_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
            -I${CMAKE_CURRENT_SOURCE_DIR}/../
            -I${CMAKE_CURRENT_SOURCE_DIR}/../aiv_interface/
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMAND_EXPAND_LISTS VERBATIM
            DEPENDS kernel_tiling_headers
    )

    add_custom_target(hccl_rs_superkernel_float32 ALL
        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/hccl_rs_superkernel_float32.o)
    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/hccl_rs_superkernel_float32.o
            COMMAND ${BISHENG_CCACHE} ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_rs_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_rs_superkernel_float32.o --cce-aicore-arch=dav-c220-vec
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DASCENDC_DUMP=0 -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0 -DHCCL_DTYPE_FP32 -Dsk_reducescatter=sk_reducescatter_float32
            --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-stack-size=0x8000
            "-I$<JOIN:$<TARGET_PROPERTY:asc_kernel_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
            "-I$<JOIN:$<TARGET_PROPERTY:kernel_tiling_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
            -I${CMAKE_CURRENT_SOURCE_DIR}/../
            -I${CMAKE_CURRENT_SOURCE_DIR}/../aiv_interface/
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMAND_EXPAND_LISTS VERBATIM
            DEPENDS kernel_tiling_headers
    )

    add_custom_target(hccl_rs_superkernel_bfloat16 ALL
        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/hccl_rs_superkernel_bfloat16.o)
    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/hccl_rs_superkernel_bfloat16.o
            COMMAND ${BISHENG_CCACHE} ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_rs_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_rs_superkernel_bfloat16.o --cce-aicore-arch=dav-c220-vec
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DASCENDC_DUMP=0 -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0 -DHCCL_DTYPE_BFP16 -Dsk_reducescatter=sk_reducescatter_bfloat16
            --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-stack-size=0x8000
            "-I$<JOIN:$<TARGET_PROPERTY:asc_kernel_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
            "-I$<JOIN:$<TARGET_PROPERTY:kernel_tiling_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
            -I${CMAKE_CURRENT_SOURCE_DIR}/../
            -I${CMAKE_CURRENT_SOURCE_DIR}/../aiv_interface/
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMAND_EXPAND_LISTS VERBATIM
            DEPENDS kernel_tiling_headers
    )

    add_custom_target(hccl_sk_rs_deter ALL
        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/hccl_sk_rs_deter.o)
    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/hccl_sk_rs_deter.o
            COMMAND ${BISHENG_CCACHE} ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_sk_rs_deter.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_sk_rs_deter.o --cce-aicore-arch=dav-c220-vec
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DASCENDC_DUMP=0 -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0
            --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-stack-size=0x8000
            "-I$<JOIN:$<TARGET_PROPERTY:asc_kernel_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
            "-I$<JOIN:$<TARGET_PROPERTY:kernel_tiling_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
            -I${CMAKE_CURRENT_SOURCE_DIR}/../
            -I${CMAKE_CURRENT_SOURCE_DIR}/../aiv_interface/
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMAND_EXPAND_LISTS VERBATIM
            DEPENDS kernel_tiling_headers
    )

    add_custom_target(hccl_sk_rs_crossnode ALL
        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/hccl_sk_rs_crossnode.o)
    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/hccl_sk_rs_crossnode.o
            COMMAND ${BISHENG_CCACHE} ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_sk_rs_crossnode.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_sk_rs_crossnode.o --cce-aicore-arch=dav-c220-vec
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DASCENDC_DUMP=0 -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0
            --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-stack-size=0x8000
            "-I$<JOIN:$<TARGET_PROPERTY:asc_kernel_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
            "-I$<JOIN:$<TARGET_PROPERTY:kernel_tiling_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
            -I${CMAKE_CURRENT_SOURCE_DIR}/../
            -I${CMAKE_CURRENT_SOURCE_DIR}/../aiv_interface/
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMAND_EXPAND_LISTS VERBATIM
            DEPENDS kernel_tiling_headers
    )
else()
    get_filename_component(CCE_DIR "${CMAKE_LINKER}" PATH)

    set(TILING_GEN_DIR ${CMAKE_BINARY_DIR}/asc/asc-devkit)

    add_custom_target(hccl_aiv_reduce_scatter_op
        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/hccl_aiv_reduce_scatter_op_tmp.o ${CMAKE_CURRENT_BINARY_DIR}/hccl_aiv_reduce_scatter_op_sk1.o ${CMAKE_CURRENT_BINARY_DIR}/hccl_aiv_reduce_scatter_op_sk2.o ${CMAKE_CURRENT_BINARY_DIR}/hccl_aiv_reduce_scatter_op_sk3.o ${CMAKE_CURRENT_BINARY_DIR}/hccl_aiv_reduce_scatter_op_sk4.o kernel_tiling_headers
        COMMAND ${CCE_DIR}/ld.lld -m aicorelinux -Ttext=0 ${CMAKE_CURRENT_BINARY_DIR}/hccl_aiv_reduce_scatter_op_tmp.o ${CMAKE_CURRENT_BINARY_DIR}/hccl_aiv_reduce_scatter_op_sk1.o ${CMAKE_CURRENT_BINARY_DIR}/hccl_aiv_reduce_scatter_op_sk2.o ${CMAKE_CURRENT_BINARY_DIR}/hccl_aiv_reduce_scatter_op_sk3.o ${CMAKE_CURRENT_BINARY_DIR}/hccl_aiv_reduce_scatter_op_sk4.o -static -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_aiv_reduce_scatter_op.o
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )

    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/hccl_aiv_reduce_scatter_op_tmp.o
            COMMAND ${BISHENG_CCACHE} ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_reduce_scatter_op.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_aiv_reduce_scatter_op_tmp.o
            -DASCENDC_DUMP=${ASCENDC_DUMP} -DTILING_KEY_VAR=0 --cce-aicore-arch=dav-c220-vec
            --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-dcci-insert-for-scalar=false
            -mllvm -cce-aicore-stack-size=0x8000
            "-I$<JOIN:$<TARGET_PROPERTY:asc_kernel_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
            "-I$<JOIN:$<TARGET_PROPERTY:kernel_tiling_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
            -I${TOP_DIR}/hcomm/include/hccl/
            -I${CMAKE_CURRENT_SOURCE_DIR}/../
            -I${CMAKE_CURRENT_SOURCE_DIR}/../aiv_interface/
            -I${TILING_GEN_DIR}
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all -DONE_CORE_DUMP_SIZE=1024
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMAND_EXPAND_LISTS VERBATIM
            DEPENDS kernel_tiling_headers
    )

    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/hccl_aiv_reduce_scatter_op_sk1.o
            COMMAND ${BISHENG_CCACHE} ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_reduce_scatter_op.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_aiv_reduce_scatter_op_sk1.o
            -DASCENDC_DUMP=${ASCENDC_DUMP} -DTILING_KEY_VAR=0 -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DBUILD_SK_FUNC -DSK_FUNC_ID=1 --cce-aicore-arch=dav-c220-vec
            --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-dcci-insert-for-scalar=false
            -mllvm -cce-aicore-stack-size=0x8000
            "-I$<JOIN:$<TARGET_PROPERTY:asc_kernel_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
            "-I$<JOIN:$<TARGET_PROPERTY:kernel_tiling_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
            -I${TOP_DIR}/hcomm/include/hccl/
            -I${CMAKE_CURRENT_SOURCE_DIR}/../
            -I${CMAKE_CURRENT_SOURCE_DIR}/../aiv_interface/
            -I${TILING_GEN_DIR}
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all -DONE_CORE_DUMP_SIZE=1024
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMAND_EXPAND_LISTS VERBATIM
            DEPENDS kernel_tiling_headers
    )

    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/hccl_aiv_reduce_scatter_op_sk2.o
            COMMAND ${BISHENG_CCACHE} ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_reduce_scatter_op.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_aiv_reduce_scatter_op_sk2.o
            -DASCENDC_DUMP=${ASCENDC_DUMP} -DTILING_KEY_VAR=0 -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DBUILD_SK_FUNC -DSK_FUNC_ID=2 --cce-aicore-arch=dav-c220-vec
            --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-dcci-insert-for-scalar=false
            -mllvm -cce-aicore-stack-size=0x8000
            "-I$<JOIN:$<TARGET_PROPERTY:asc_kernel_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
            "-I$<JOIN:$<TARGET_PROPERTY:kernel_tiling_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
            -I${TOP_DIR}/hcomm/include/hccl/
            -I${CMAKE_CURRENT_SOURCE_DIR}/../
            -I${CMAKE_CURRENT_SOURCE_DIR}/../aiv_interface/
            -I${TILING_GEN_DIR}
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all -DONE_CORE_DUMP_SIZE=1024
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMAND_EXPAND_LISTS VERBATIM
            DEPENDS kernel_tiling_headers
    )

    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/hccl_aiv_reduce_scatter_op_sk3.o
            COMMAND ${BISHENG_CCACHE} ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_reduce_scatter_op.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_aiv_reduce_scatter_op_sk3.o
            -DASCENDC_DUMP=${ASCENDC_DUMP} -DTILING_KEY_VAR=0 -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DBUILD_SK_FUNC -DSK_FUNC_ID=3 --cce-aicore-arch=dav-c220-vec
            --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-dcci-insert-for-scalar=false
            -mllvm -cce-aicore-stack-size=0x8000
            "-I$<JOIN:$<TARGET_PROPERTY:asc_kernel_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
            "-I$<JOIN:$<TARGET_PROPERTY:kernel_tiling_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
            -I${TOP_DIR}/hcomm/include/hccl/
            -I${CMAKE_CURRENT_SOURCE_DIR}/../
            -I${CMAKE_CURRENT_SOURCE_DIR}/../aiv_interface/
            -I${TILING_GEN_DIR}
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all -DONE_CORE_DUMP_SIZE=1024
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMAND_EXPAND_LISTS VERBATIM
            DEPENDS kernel_tiling_headers
    )

    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/hccl_aiv_reduce_scatter_op_sk4.o
            COMMAND ${BISHENG_CCACHE} ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_reduce_scatter_op.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_aiv_reduce_scatter_op_sk4.o
            -DASCENDC_DUMP=${ASCENDC_DUMP} -DTILING_KEY_VAR=0 -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DBUILD_SK_FUNC -DSK_FUNC_ID=4 --cce-aicore-arch=dav-c220-vec
            --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-dcci-insert-for-scalar=false
            -mllvm -cce-aicore-stack-size=0x8000
            "-I$<JOIN:$<TARGET_PROPERTY:asc_kernel_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
            "-I$<JOIN:$<TARGET_PROPERTY:kernel_tiling_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
            -I${TOP_DIR}/hcomm/include/hccl/
            -I${CMAKE_CURRENT_SOURCE_DIR}/../
            -I${CMAKE_CURRENT_SOURCE_DIR}/../aiv_interface/
            -I${TILING_GEN_DIR}
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all -DONE_CORE_DUMP_SIZE=1024
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMAND_EXPAND_LISTS VERBATIM
            DEPENDS kernel_tiling_headers
    )

    add_custom_target(hccl_rs_superkernel_int8
            COMMAND ${BISHENG_CCACHE} ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_rs_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_rs_superkernel_int8.o --cce-aicore-arch=dav-c220-vec
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DASCENDC_DUMP=0 -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0 -DHCCL_DTYPE_INT8 -Dsk_reducescatter=sk_reducescatter_int8
            --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-stack-size=0x8000
            "-I$<JOIN:$<TARGET_PROPERTY:asc_kernel_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
            "-I$<JOIN:$<TARGET_PROPERTY:kernel_tiling_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
            -I${TOP_DIR}/hcomm/include/hccl/
            -I${CMAKE_CURRENT_SOURCE_DIR}/../
            -I${CMAKE_CURRENT_SOURCE_DIR}/../aiv_interface/
            -I${TILING_GEN_DIR}
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMAND_EXPAND_LISTS VERBATIM
            DEPENDS kernel_tiling_headers
    )

    add_custom_target(hccl_rs_superkernel_int16
            COMMAND ${BISHENG_CCACHE} ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_rs_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_rs_superkernel_int16.o --cce-aicore-arch=dav-c220-vec
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DASCENDC_DUMP=0 -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0 -DHCCL_DTYPE_INT16 -Dsk_reducescatter=sk_reducescatter_int16
            --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-stack-size=0x8000
            "-I$<JOIN:$<TARGET_PROPERTY:asc_kernel_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
            "-I$<JOIN:$<TARGET_PROPERTY:kernel_tiling_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
            -I${TOP_DIR}/hcomm/include/hccl/
            -I${CMAKE_CURRENT_SOURCE_DIR}/../
            -I${CMAKE_CURRENT_SOURCE_DIR}/../aiv_interface/
            -I${TILING_GEN_DIR}
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMAND_EXPAND_LISTS VERBATIM
            DEPENDS kernel_tiling_headers
    )

    add_custom_target(hccl_rs_superkernel_int32
            COMMAND ${BISHENG_CCACHE} ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_rs_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_rs_superkernel_int32.o --cce-aicore-arch=dav-c220-vec
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DASCENDC_DUMP=0 -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0 -DHCCL_DTYPE_INT32 -Dsk_reducescatter=sk_reducescatter_int32
            --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-stack-size=0x8000
            "-I$<JOIN:$<TARGET_PROPERTY:asc_kernel_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
            "-I$<JOIN:$<TARGET_PROPERTY:kernel_tiling_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
            -I${TOP_DIR}/hcomm/include/hccl/
            -I${CMAKE_CURRENT_SOURCE_DIR}/../
            -I${CMAKE_CURRENT_SOURCE_DIR}/../aiv_interface/
            -I${TILING_GEN_DIR}
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMAND_EXPAND_LISTS VERBATIM
            DEPENDS kernel_tiling_headers
    )

    add_custom_target(hccl_rs_superkernel_float16
            COMMAND ${BISHENG_CCACHE} ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_rs_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_rs_superkernel_float16.o --cce-aicore-arch=dav-c220-vec
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DASCENDC_DUMP=0 -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0 -DHCCL_DTYPE_FP16 -Dsk_reducescatter=sk_reducescatter_float16
            --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-stack-size=0x8000
            "-I$<JOIN:$<TARGET_PROPERTY:asc_kernel_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
            "-I$<JOIN:$<TARGET_PROPERTY:kernel_tiling_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
            -I${TOP_DIR}/hcomm/include/hccl/
            -I${CMAKE_CURRENT_SOURCE_DIR}/../
            -I${CMAKE_CURRENT_SOURCE_DIR}/../aiv_interface/
            -I${TILING_GEN_DIR}
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMAND_EXPAND_LISTS VERBATIM
            DEPENDS kernel_tiling_headers
    )

    add_custom_target(hccl_rs_superkernel_float32
            COMMAND ${BISHENG_CCACHE} ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_rs_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_rs_superkernel_float32.o --cce-aicore-arch=dav-c220-vec
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DASCENDC_DUMP=0 -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0 -DHCCL_DTYPE_FP32 -Dsk_reducescatter=sk_reducescatter_float32
            --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-stack-size=0x8000
            "-I$<JOIN:$<TARGET_PROPERTY:asc_kernel_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
            "-I$<JOIN:$<TARGET_PROPERTY:kernel_tiling_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
            -I${TOP_DIR}/hcomm/include/hccl/
            -I${CMAKE_CURRENT_SOURCE_DIR}/../
            -I${CMAKE_CURRENT_SOURCE_DIR}/../aiv_interface/
            -I${TILING_GEN_DIR}
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMAND_EXPAND_LISTS VERBATIM
            DEPENDS kernel_tiling_headers
    )

    add_custom_target(hccl_rs_superkernel_bfloat16
            COMMAND ${BISHENG_CCACHE} ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_rs_superkernel.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_rs_superkernel_bfloat16.o --cce-aicore-arch=dav-c220-vec
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DASCENDC_DUMP=0 -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0 -DHCCL_DTYPE_BFP16 -Dsk_reducescatter=sk_reducescatter_bfloat16
            --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-stack-size=0x8000
            "-I$<JOIN:$<TARGET_PROPERTY:asc_kernel_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
            "-I$<JOIN:$<TARGET_PROPERTY:kernel_tiling_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
            -I${TOP_DIR}/hcomm/include/hccl/
            -I${CMAKE_CURRENT_SOURCE_DIR}/../
            -I${CMAKE_CURRENT_SOURCE_DIR}/../aiv_interface/
            -I${TILING_GEN_DIR}
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMAND_EXPAND_LISTS VERBATIM
            DEPENDS kernel_tiling_headers
    )

    add_custom_target(hccl_sk_rs_deter
            COMMAND ${BISHENG_CCACHE} ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_sk_rs_deter.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_sk_rs_deter.o --cce-aicore-arch=dav-c220-vec
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DASCENDC_DUMP=0 -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0
            --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-stack-size=0x8000
            "-I$<JOIN:$<TARGET_PROPERTY:asc_kernel_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
            "-I$<JOIN:$<TARGET_PROPERTY:kernel_tiling_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
            -I${TOP_DIR}/hcomm/include/hccl/
            -I${CMAKE_CURRENT_SOURCE_DIR}/../
            -I${CMAKE_CURRENT_SOURCE_DIR}/../aiv_interface/
            -I${TILING_GEN_DIR}
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMAND_EXPAND_LISTS VERBATIM
            DEPENDS kernel_tiling_headers
   )

   add_custom_target(hccl_sk_rs_crossnode
            COMMAND ${BISHENG_CCACHE} ${CCE_DIR}/ccec -c -x cce -O2 ${CMAKE_CURRENT_SOURCE_DIR}/aiv_sk_rs_crossnode.h -o ${CMAKE_CURRENT_BINARY_DIR}/hccl_sk_rs_crossnode.o --cce-aicore-arch=dav-c220-vec
            -DTILING_KEY_VAR=0UL -D__ASCENDC_ENABLE_SUPER_KERNEL__ -DASCENDC_DUMP=0 -DL2_CACHE_HINT -DCONST_TILING -DDETERMINISTIC_MODE=0
            --cce-aicore-only -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-stack-size=0x8000
            "-I$<JOIN:$<TARGET_PROPERTY:asc_kernel_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
            "-I$<JOIN:$<TARGET_PROPERTY:kernel_tiling_headers,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
            -I${TOP_DIR}/hcomm/include/hccl/
            -I${CMAKE_CURRENT_SOURCE_DIR}/../
            -I${CMAKE_CURRENT_SOURCE_DIR}/../aiv_interface/
            -I${TILING_GEN_DIR}
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm
            --cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMAND_EXPAND_LISTS VERBATIM
            DEPENDS kernel_tiling_headers
   )
endif()

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/hccl_aiv_reduce_scatter_op.o ${INSTALL_OPTIONAL}
    COMPONENT hcomm
    DESTINATION ${INSTALL_LIBRARY_DIR}
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/hccl_rs_superkernel_int8.o ${INSTALL_OPTIONAL}
    COMPONENT hcomm
    DESTINATION ${INSTALL_LIBRARY_DIR}
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/hccl_rs_superkernel_int16.o ${INSTALL_OPTIONAL}
    COMPONENT hcomm
    DESTINATION ${INSTALL_LIBRARY_DIR}
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/hccl_rs_superkernel_int32.o ${INSTALL_OPTIONAL}
    COMPONENT hcomm
    DESTINATION ${INSTALL_LIBRARY_DIR}
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/hccl_rs_superkernel_float16.o ${INSTALL_OPTIONAL}
    COMPONENT hcomm
    DESTINATION ${INSTALL_LIBRARY_DIR}
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/hccl_rs_superkernel_float32.o ${INSTALL_OPTIONAL}
    COMPONENT hcomm
    DESTINATION ${INSTALL_LIBRARY_DIR}
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/hccl_rs_superkernel_bfloat16.o ${INSTALL_OPTIONAL}
    COMPONENT hcomm
    DESTINATION ${INSTALL_LIBRARY_DIR}
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/hccl_sk_rs_deter.o ${INSTALL_OPTIONAL}
    COMPONENT hcomm
    DESTINATION ${INSTALL_LIBRARY_DIR}
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/hccl_sk_rs_crossnode.o ${INSTALL_OPTIONAL}
    COMPONENT hcomm
    DESTINATION ${INSTALL_LIBRARY_DIR}
)