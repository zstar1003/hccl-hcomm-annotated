# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

if(TARGET hccl_v2)
    target_sources(hccl_v2 PRIVATE
        ccu_alg_template_base.cc
        ccu_temp_all_gather_mesh_1D.cc
        ccu_temp_all_gather_mesh_1D_detour.cc
        ccu_temp_all_gather_mesh_1D_mem2mem.cc
        ccu_temp_all_gather_mesh_2D.cc
        ccu_temp_all_gather_mesh_2D_mem2mem.cc
        ccu_temp_reduce_scatter_mesh_1D.cc
        ccu_temp_all_reduce_mesh_1D.cc
        ccu_temp_all_reduce_mesh_1D_one_shot.cc
        ccu_temp_all_to_all_mesh_1D.cc
        ccu_temp_reduce_mesh_1D.cc
        ccu_temp_reduce_mesh_1D_mem2mem.cc
        ccu_temp_reduce_nhr_1D_mem2mem.cc
        ccu_temp_all_to_all_v_mesh_1D.cc
        ccu_temp_all_to_all_v_mesh_2D.cc
        ccu_temp_scatter_mesh_1D.cc
        ccu_temp_broadcast_mesh_1D.cc
        ccu_temp_broadcast_mesh_1D_mem2mem.cc
        ccu_temp_reduce_scatter_mesh_2D.cc
        ccu_temp_reduce_scatter_mesh_2D_mem2mem.cc
        ccu_temp_broadcast_mesh_2D.cc
        ccu_temp_broadcast_mesh_2D_mem2mem.cc
        ccu_temp_broadcast_nhr_1D_mem2mem.cc
        ccu_temp_all_to_all_mesh2d.cc
        ccu_temp_all_reduce_mesh_2D_one_shot.cc
        ccu_temp_all_reduce_mesh_2D_two_shot.cc
        ccu_temp_all_reduce_mesh_2D_two_shot_mem2mem.cc
        ccu_temp_scatter_mesh_2D.cc
        ccu_temp_reduce_mesh_2D.cc
        ccu_temp_reduce_mesh_2D_mem2mem.cc
        ccu_temp_all_reduce_mesh_detour_1D.cc
        ccu_temp_all_reduce_mesh_1D_mem2mem.cc
        ccu_temp_reduce_scatter_mesh_detour_1D.cc
        ccu_temp_half_alltoallv_mesh_1D.cc
        ccu_temp_reduce_scatter_mesh_1D_mem2mem.cc
        ccu_temp_all_gather_mesh_1D_mem2mem_with_stride.cc
        ccu_temp_all_gather_nhr_1D_mem2mem.cc
        ccu_temp_all_gather_v_mesh_1D.cc
        ccu_temp_reduce_scatter_v_mesh_1D.cc
        ccu_temp_reduce_scatter_v_mesh_1D_mem2mem.cc
        ccu_temp_all_reduce_nhr_1D_mem2mem.cc
        ccu_temp_reduce_scatter_nhr_1D_mem2mem.cc
        ccu_temp_scatter_nhr_1D_mem2mem.cc
        ccu_temp_all_gather_mesh_1D_2die.cc
        ccu_temp_all_to_all_mesh_1D_2Die.cc
        ccu_temp_all_to_all_v_mesh_2Die.cc
        ccu_temp_reduce_scatter_mesh_1D_2die.cc
        ccu_temp_reduce_mesh_1D_two_shot_mem2mem.cc
    )
endif()
