# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

LOCAL_PATH 		:= 	$(call my-dir)

include $(CLEAR_VARS)

LOCAL_MODULE 	:= 	librs

LOCAL_LDFLAGS	+= 	-lrt

LOCAL_LDFLAGS	+=	-ldl
LOCAL_LDFLAGS	+=	-Wl,-Bsymbolic -Wl,--exclude-libs,ALL
# LOCAL_PATH用于build核心加上LOCAL_SRC_FILES组合成源文件, 如果需要编译
# 源代码文件. 则需要PATH_BRIDGE引过去, PATH_BRIDGE和LOCAL_PATH组合成build
# 核心需要的源代码路径
PATH_BRIDGE		:=

LOCAL_SRC_FILES :=
#LOCAL_SRC_FILES += 	$(wildcard *.c *.cpp *.s *.cc *.C)
LOCAL_SRC_FILES += $(PATH_BRIDGE)rs.c $(PATH_BRIDGE)rs_epoll.c $(PATH_BRIDGE)rs_socket.c $(PATH_BRIDGE)rs_rdma.c $(PATH_BRIDGE)rs_drv_rdma.c $(PATH_BRIDGE)rs_drv_socket.c
LOCAL_SRC_FILES += $(PATH_BRIDGE)rs_ssl.c
LOCAL_SRC_FILES += $(PATH_BRIDGE)../../crypt/kmc/callback.c $(PATH_BRIDGE)../../crypt/kmc/kmc_crypt.c $(PATH_BRIDGE)../../crypt/priv/encrypt.c
LOCAL_SRC_FILES += $(PATH_BRIDGE)../../common/file_opt.c
LOCAL_SRC_FILES += $(PATH_BRIDGE)../../common/network_comm.c
LOCAL_C_INCLUDES:= $(TOPDIR)inc/network

IO_ROOT_DIR := $(TOPDIR)third_party

#第三方头文件搜索路径
LOCAL_C_INCLUDES+= 	$(IO_ROOT_DIR)/ofed/build/rdma-core/include
LOCAL_C_INCLUDES+=      $(TOPDIR)open_source/openssl/include
LOCAL_C_INCLUDES+= 	$(IO_ROOT_DIR)/../libc_sec/include
LOCAL_C_INCLUDES+=      ${TOPDIR}/hccl/src/platform/hccp/external_depends/rdma-core/providers/hns
LOCAL_C_INCLUDES+=	$(TOPDIR)inc/toolchain
LOCAL_C_INCLUDES+=	$(TOPDIR)inc/driver
LOCAL_C_INCLUDES+=	$(TOPDIR)drivers/network/include
LOCAL_C_INCLUDES+=	$(TOPDIR)drivers/network/crypt/inc
LOCAL_C_INCLUDES+=      $(TOPDIR)drivers/network/common
LOCAL_C_INCLUDES+=      $(TOPDIR)inc/mmpa
LOCAL_C_INCLUDES+=      $(TOPDIR)libkmc/include
LOCAL_C_INCLUDES+=      $(TOPDIR)libkmc/src/sdp
LOCAL_C_INCLUDES+=      $(TOPDIR)libkmc/src/common
LOCAL_C_INCLUDES+=      $(HOST_OUT_THIRD_PARTY_LIBS)/rmda_core/open_source/OFED-4.17-1/SRPMS/RH/rdma-core-17.5/build/include \

#第三方库搜索路径
LOCAL_LD_DIRS :=

LOCAL_CFLAGS += -Werror -lssl -lcrypto -std=c11 -Wfloat-equal -Wextra
## add more LOCAL_SRC_FILES and LOCAL_C_INCLUDES

LOCAL_SHARED_LIBRARIES := libibverbs libc_sec libslog libascend_hal libhns-rdmav17 libmmpa
LOCAL_STATIC_LIBRARIES := libKMC libSDP libssl libcrypto

LOCAL_CFLAGS += -DCONFIG_SSL

LOCAL_CFLAGS += -fvisibility=hidden
include $(BUILD_HOST_SHARED_LIBRARY)

include $(CLEAR_VARS)

LOCAL_MODULE 	:= 	librs

LOCAL_LDFLAGS	+= 	-lrt

LOCAL_LDFLAGS   +=      -ldl
LOCAL_LDFLAGS	+=	-Wl,-Bsymbolic -Wl,--exclude-libs,ALL
# LOCAL_PATH用于build核心加上LOCAL_SRC_FILES组合成源文件, 如果需要编译
# 源代码文件. 则需要PATH_BRIDGE引过去, PATH_BRIDGE和LOCAL_PATH组合成build
# 核心需要的源代码路径
PATH_BRIDGE		:=

LOCAL_SRC_FILES :=
LOCAL_SRC_FILES += 	$(wildcard *.c *.cpp *.s *.cc *.C)
LOCAL_SRC_FILES += $(PATH_BRIDGE)rs.c $(PATH_BRIDGE)rs_epoll.c $(PATH_BRIDGE)rs_socket.c $(PATH_BRIDGE)rs_rdma.c $(PATH_BRIDGE)rs_drv_rdma.c $(PATH_BRIDGE)rs_drv_socket.c
LOCAL_SRC_FILES += $(PATH_BRIDGE)../../crypt/kmc/callback.c $(PATH_BRIDGE)../../crypt/kmc/kmc_crypt.c $(PATH_BRIDGE)../../crypt/priv/encrypt.c
LOCAL_SRC_FILES += $(PATH_BRIDGE)../../common/file_opt.c
LOCAL_SRC_FILES += $(PATH_BRIDGE)rs_ssl.c

LOCAL_C_INCLUDES:=  $(TOPDIR)inc/network

IO_ROOT_DIR := $(TOPDIR)third_party

LOCAL_C_INCLUDES+= 	$(IO_ROOT_DIR)/ofed/build/rdma-core/include
LOCAL_C_INCLUDES+=      $(TOPDIR)open_source/openssl/include
LOCAL_C_INCLUDES+= 	$(IO_ROOT_DIR)/../libc_sec/include
LOCAL_C_INCLUDES+=      ${TOPDIR}/hccl/src/platform/hccp/external_depends/rdma-core/providers/hns
LOCAL_C_INCLUDES+=	$(TOPDIR)inc/toolchain
LOCAL_C_INCLUDES+=      $(TOPDIR)inc/driver
LOCAL_C_INCLUDES+=	$(TOPDIR)drivers/network/include
LOCAL_C_INCLUDES+=      $(TOPDIR)drivers/network/common
LOCAL_C_INCLUDES+=	$(TOPDIR)inc/driver
LOCAL_C_INCLUDES+=	$(TOPDIR)drivers/network/crypt/inc
LOCAL_C_INCLUDES+=      $(TOPDIR)inc/mmpa
LOCAL_C_INCLUDES+=      $(TOPDIR)libkmc/include
LOCAL_C_INCLUDES+=      $(TOPDIR)libkmc/src/sdp
LOCAL_C_INCLUDES+=      $(TOPDIR)libkmc/src/common
LOCAL_LD_DIRS :=

LOCAL_CFLAGS += -Werror -lssl -lcrypto -DRS_DEVICE_CONFIG -Dgoogle=ascend_private -std=c11 -Wfloat-equal -Wextra
## add more LOCAL_SRC_FILES and LOCAL_C_INCLUDES

LOCAL_SHARED_LIBRARIES := libibverbs libc_sec libslog libascend_hal libhns-rdmav17 libmmpa
LOCAL_STATIC_LIBRARIES := libKMC libSDP libssl libcrypto
LOCAL_WHOLE_STATIC_LIBRARIES := libascend_protobuf

LOCAL_CFLAGS += -DCONFIG_SSL

LOCAL_CFLAGS += -fvisibility=hidden
include $(BUILD_SHARED_LIBRARY)