#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set -e

shell_forder=$(cd "$(dirname "$0")"; pwd)
clang_path=${shell_forder}/../../legacy/.clang-format

search_dir=${shell_forder}/../../legacy
cd ${search_dir}

function format_file()
{
    file=$1
    if [ "${file##*.}"x = "cc"x ]||[ "${file##*.}"x = "h"x ];then
        echo "clang-format: $1"
        /usr/bin/clang-format -i -style=file $1
    fi
}

function list_dir()
{
    for i in `ls`; do
        if [ -d "$i" ]; then
            cd ./${i}
            list_dir
            cd ../
        else
            file=`readlink -f ${i}`
            format_file ${file}
        fi
    done
}

list_dir