/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_REPRESENTATION
#define HCCL_CCU_REPRESENTATION

#include "ccu_rep_nop.h"
#include "ccu_rep_loadarg.h"
#include "ccu_rep_add.h"
#include "ccu_rep_assign.h"

#include "ccu_rep_setloop.h"
#include "ccu_rep_loop.h"
#include "ccu_rep_loopgroup.h"

#include "ccu_rep_locpostsem.h"
#include "ccu_rep_locwaitsem.h"

#include "ccu_rep_read.h"
#include "ccu_rep_write.h"
#include "ccu_rep_loccpy.h"

#include "ccu_rep_rempostsem.h"
#include "ccu_rep_rempostvar.h"
#include "ccu_rep_remwaitgroup.h"
#include "ccu_rep_remwaitsem.h"

#include "ccu_rep_postsharedsem.h"
#include "ccu_rep_postsharedvar.h"

#include "ccu_rep_buflocread.h"
#include "ccu_rep_buflocwrite.h"
#include "ccu_rep_bufreduce.h"
#include "ccu_rep_bufread.h"
#include "ccu_rep_bufwrite.h"
#include "ccu_rep_remMem.h"

#include "ccu_rep_funccall.h"
#include "ccu_rep_funcblock.h"

#include "ccu_condition.h"
#include "ccu_repeat.h"
#include "ccu_loopblock.h"
#include "ccu_loopgroupcall.h"
#include "ccu_funcblock.h"
#include "ccu_funccall.h"

#include "ccu_interface_assist.h"
#include "ccu_rep_load.h"
#include "ccu_rep_store.h"
#include "ccu_rep_load_var.h"
#include "ccu_rep_store_var.h"

#endif // HCCL_CCU_REPRESENTATION
