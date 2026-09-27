/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <stdio.h>

static void* dl_handle;

void* dlopen_stub(char* lib_path, int para) { return dl_handle; }

int dlclose_stub(void* handle) { return -1; }

int ibv_set_device_err_stub(int dev_id) { return -1; }

int ibv_set_device_ok_stub(int dev_id) { return 0; }

static int (*ibv_set_device_err_stub_p)(int dev_id) = ibv_set_device_err_stub;
static int (*ibv_set_device_ok_stub_p)(int dev_id) = ibv_set_device_ok_stub;

void* dlsym_stub_1(void* handle, char* func) { return ibv_set_device_err_stub_p; }

void* dlsym_stub_2(void* handle, char* func) { return ibv_set_device_ok_stub_p; }
