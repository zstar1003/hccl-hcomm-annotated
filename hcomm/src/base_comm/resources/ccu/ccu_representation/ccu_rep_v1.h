/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_CCU_REPRESENTATION
#define HCOMM_CCU_REPRESENTATION

#include "ccu_rep_nop_v1.h"
#include "ccu_rep_loadarg_v1.h"
#include "ccu_rep_add_v1.h"
#include "ccu_rep_assign_v1.h"
#include "ccu_rep_mul_v1.h"
#include "ccu_rep_sub_v1.h"

#include "ccu_rep_setloop_v1.h"
#include "ccu_rep_loop_v1.h"
#include "ccu_rep_loopcall_v1.h"
#include "ccu_rep_loopgroup_bundle_v1.h"

#include "ccu_rep_loc_record_event.h"
#include "ccu_rep_loc_wait_event.h"
#include "ccu_rep_loc_wait_notify.h"

#include "ccu_rep_read_v1.h"
#include "ccu_rep_write_v1.h"
#include "ccu_rep_loccpy_v1.h"

#include "ccu_rep_rempostsem_v1.h"
#include "ccu_rep_rempostvar_v1.h"
#include "ccu_rep_remwaitsem_v1.h"

#include "ccu_rep_record_shared_notify.h"

#include "ccu_rep_buflocread_v1.h"
#include "ccu_rep_buflocwrite_v1.h"
#include "ccu_rep_bufreduce_v1.h"
#include "ccu_rep_bufread_v1.h"
#include "ccu_rep_bufwrite_v1.h"
#include "ccu_rep_remMem_v1.h"

#include "ccu_rep_funccall_v1.h"
#include "ccu_rep_funcblock_v1.h"

#include "ccu_condition_v1.h"
#include "ccu_repeat_v1.h"
#include "ccu_loopblock_v1.h"
#include "ccu_funcblock_v1.h"
#include "ccu_funccall_v1.h"

#include "ccu_interface_assist_v1.h"
#include "ccu_rep_load_v1.h"
#include "ccu_rep_store_v1.h"
#include "ccu_rep_load_var_v1.h"
#include "ccu_rep_store_var_v1.h"

#include "ccu_rep_and.h"
#include "ccu_rep_or.h"
#include "ccu_rep_not.h"
#include "ccu_rep_xor.h"

#include "ccu_rep_shl.h"
#include "ccu_rep_shr.h"

#include "ccu_datatype_v1.h"
#include "ccu_microcode_v1.h"

#include "hccl_res.h"

#include "ccu_rep_add_v1.h"
#include "ccu_rep_assign_v1.h"

#include "ccu_rep_write_var_atomic.h"
#include "ccu_rep_write_with_cnt_inc.h"
#include "ccu_rep_casc_cnt_wait.h"
#include "ccu_rep_casc_cnt_clear.h"

#include "ccu_rep_load_add_imm.h"
#include "ccu_rep_store_add_imm.h"

#endif // HCCL_CCU_REPRESENTATION
