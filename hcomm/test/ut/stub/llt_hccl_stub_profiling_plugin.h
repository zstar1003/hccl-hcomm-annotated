/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __LLT_HCCL_STUB_PROFILING_PLUGIN_H__
#define __LLT_HCCL_STUB_PROFILING_PLUGIN_H__

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <semaphore.h>
#include <assert.h> /* for assert  */
#include <errno.h>
#include <sys/time.h> /* 获取时间 */

#include <hccl/base.h>
#include <hccl/hccl_types.h>
#include <securec.h>
#include <unistd.h>
#include <signal.h>
#include <syscall.h>
#include <sys/prctl.h>
#include <syslog.h>
#include <sys/mman.h>
#include <sys/stat.h> /* For mode constants */
#include <fcntl.h>    /* For O_* constants */

#include <string>
#include <list>
#include <map>
#include <iostream>
#include <fstream>
#include <mutex>

#include "llt_hccl_stub.h"
#include "task_profiling_pub.h"

#endif /* __LLT_HCCL_STUB_PROFILING_PLUGIN_H__ */
