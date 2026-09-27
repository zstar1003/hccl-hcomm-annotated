# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

CXX ?= g++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra
THIRDLIB_ROOT ?= /usr/local/third_lib
INCLUDES := -I. -I.. -I../.. -I$(THIRDLIB_ROOT)/include
OUT_DIR := build

.PHONY: all clean

all: $(OUT_DIR)/proc_res

$(OUT_DIR):
	mkdir -p $(OUT_DIR)

$(OUT_DIR)/proc_res: proc_res.cpp | $(OUT_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) proc_res.cpp -o $@

clean:
	rm -f $(OUT_DIR)/proc_res
