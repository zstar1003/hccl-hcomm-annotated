# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

ifndef ASCEND_HOME_PATH
    $(error "ASCEND_HOME_PATH is not set, please ensure CANN is properly installed and \
             source environment variables by running `source /path/to/Ascend/cann/set_env.sh`")
endif

ifndef MPI_HOME
    $(error "MPI_HOME is not set, please ensure MPI is properly installed and \
             set environment variable by running `export MPI_HOME=/path/to/mpi`")
endif

CXXFLAGS := -std=c++17 \
        -Werror \
        -fstack-protector-strong \
        -fPIE -pie \
        -O2 \
        -s \
        -Wl,-z,relro \
        -Wl,-z,now \
        -Wl,-z,noexecstack \
        -Wl,--copy-dt-needed-entries

SOURCES = main.cc
ASCEND_INC_DIR = ${ASCEND_HOME_PATH}/include
ASCEND_LIB_DIR = ${ASCEND_HOME_PATH}/lib64
MPI_INC_DIR = ${MPI_HOME}/include
MPI_LIB_DIR = ${MPI_HOME}/lib

LIBS = -L$(ASCEND_LIB_DIR) -lhccl -lascendcl \
       -L$(MPI_LIB_DIR) -lmpi
INCS = -I$(ASCEND_INC_DIR) -I$(MPI_INC_DIR)
TARGET = one_device_per_process

# 设备数量，默认值为：8
N ?= 8

# Default target
all:
	g++ $(CXXFLAGS) $(SOURCES) $(INCS) $(LIBS) -o ${TARGET}
	@echo "${TARGET} compile completed"

# Test target
test: 
	export LD_LIBRARY_PATH=${MPI_LIB_DIR}:${LD_LIBRARY_PATH}; \
	${MPI_HOME}/bin/mpirun -n $(N) ./${TARGET}

# Clean build artifacts
clean:
	rm ${TARGET}

# Help
help:
	@echo ""
	@echo "HCCL Example: One device per process"
	@echo "===================================="
	@echo ""
	@echo "This example shows how to use HcclCommInitRootInfoConfig() to create"
	@echo "communicators for multiple NPUs using MPI."
	@echo ""
	@echo "Targets:"
	@echo "  all       - Build the example (default)"
	@echo "  test      - Build and run test"
	@echo "  clean     - Remove build artifacts"
	@echo "  help      - Show this help"
	@echo ""

.PHONY: all test clean help
