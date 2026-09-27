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
            ins_alg_template_base.cc
            alg_data_trans_wrapper_utils.cc
            alg_data_trans_wrapper_mid.cc
            alg_data_trans_wrapper_high.cc
            ins_temp_all_gather_mesh.cc
            ins_temp_all_gather_mesh_2D.cc
            ins_temp_all_gather_nhr.cc
            ins_temp_reduce_scatter_nhr.cc
            ins_temp_scatter_nhr.cc
            ins_temp_all_reduce_mesh_2D_two_shot.cc
            ins_temp_all_to_all_mesh.cc
            ins_temp_all_to_all_mesh_2D.cc
            ins_temp_gather_nhr.cc
            ins_temp_broadcast_mesh1D_oneshot.cc
            ins_temp_broadcast_mesh_1D_two_shot.cc
            ins_temp_broadcast_mesh_2D_two_shot.cc
            ins_temp_reduce_scatter_mesh_1D.cc
            ins_temp_reduce_scatter_mesh_1D_meshchunk.cc
            ins_temp_reduce_scatter_mesh_2D.cc
            ins_temp_reduce_mesh_1D.cc
            ins_temp_reduce_mesh_1D_two_shot.cc
            ins_temp_reduce_mesh_2D.cc
            ins_temp_all_reduce_mesh_1D_one_shot.cc
            ins_temp_all_reduce_mesh_1D_two_shot.cc
            ins_temp_all_reduce_mesh_1D_two_shot_mesh_chunk.cc
            ins_temp_scatter_mesh_1d.cc
            ins_temp_scatter_mesh_2d.cc
            ins_temp_all_reduce_nhr.cc
            ins_temp_broadcast_nhr.cc
            ins_temp_all_reduce_aicpu_reduce.cc
            ins_temp_reduce_scatter_aicpu_reduce.cc
            ins_temp_reduce_aicpu_reduce.cc
            ins_temp_reduce_scatter_aicpu_reduce_mesh_2D.cc
            ins_temp_all_reduce_aicpu_reduce_mesh_2D.cc
            ins_temp_reduce_aicpu_reduce_mesh_2D.cc
            ins_temp_reduce_nhr.cc
            ins_temp_all_gather_mesh_opt.cc
            ins_temp_reduce_scatter_mesh_1D_opt.cc
    )
endif()

if(TARGET ccl_kernel)
    target_sources(ccl_kernel PRIVATE
            ins_alg_template_base.cc
            ins_temp_all_gather_mesh.cc
            ins_temp_all_gather_mesh_2D.cc
            ins_temp_all_gather_nhr.cc
            ins_temp_all_reduce_mesh_2D_two_shot.cc
            ins_temp_reduce_scatter_nhr.cc
            ins_temp_all_to_all_mesh.cc
            ins_temp_all_to_all_mesh_2D.cc
            ins_temp_scatter_nhr.cc
            ins_temp_gather_nhr.cc
            ins_temp_broadcast_mesh1D_oneshot.cc
            ins_temp_broadcast_mesh_1D_two_shot.cc
            ins_temp_broadcast_mesh_2D_two_shot.cc
            ins_temp_reduce_scatter_mesh_1D.cc
            ins_temp_reduce_scatter_mesh_1D_meshchunk.cc
            ins_temp_reduce_scatter_mesh_2D.cc
            ins_temp_reduce_mesh_1D.cc
            ins_temp_reduce_mesh_1D_two_shot.cc
            ins_temp_reduce_mesh_2D.cc
            ins_temp_all_reduce_mesh_1D_one_shot.cc
            ins_temp_all_reduce_mesh_1D_two_shot.cc
            ins_temp_all_reduce_mesh_1D_two_shot_mesh_chunk.cc
            ins_temp_scatter_mesh_1d.cc
            alg_data_trans_wrapper_mid.cc
            alg_data_trans_wrapper_high.cc
            alg_data_trans_wrapper_utils.cc
            ins_temp_scatter_mesh_2d.cc
            ins_temp_all_reduce_nhr.cc
            ins_temp_broadcast_nhr.cc
            ins_temp_all_reduce_aicpu_reduce.cc
            ins_temp_reduce_scatter_aicpu_reduce.cc
            ins_temp_reduce_aicpu_reduce.cc
            ins_temp_reduce_scatter_aicpu_reduce_mesh_2D.cc
            ins_temp_all_reduce_aicpu_reduce_mesh_2D.cc
            ins_temp_reduce_aicpu_reduce_mesh_2D.cc
            ins_temp_reduce_nhr.cc
            ins_temp_all_gather_mesh_opt.cc
            ins_temp_reduce_scatter_mesh_1D_opt.cc
    )
endif()
