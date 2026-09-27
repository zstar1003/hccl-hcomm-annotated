/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BKF_DISP_H
#define BKF_DISP_H

#include "bkf_comm.h"
#include "bkf_mem.h"

/*
1. lib实现将信息分包输出。
1.1 对外部的依赖:
   1) 以字串方式输入，引起内部函数调用
   2) 以字串方式输出。
      外部有长度有限的输出字串缓冲、能保存断点信息、能取回最新保存的断点信息(断点信息简称为上下文context, ctx)。
1.2 业务以如下两种方式写代码
1.2.1 简单的，一次输出所有内容:
      disp1()
      {
         printf(x); // 最终输出
         printf(y);
         printf(z);

         return;
      }
1.2.2 对于量大的数据，业务不需要计算输出缓冲是否满，lib会处理好，分包输出:
      disp2()
      {
          xxxInfo info;

          lastKey = getLastCtx(&info); // 获取ctx。ctx分key和info两部分，方便使用
          if (lastKey == NULL) { // 没有获取到ctx，代表首次调用该函数
              temp = appGetFirst(); // 获取第一个数据
          }
          else {
              temp = appFindNext(lastKey); // 获取下一个数据
          }

          if (temp != VOS_NULL) { // 有数据要显式
              printf(temp->a); // 显示数据
              printf(temp->b);

              info->cnt++; // 更新info信息
              info->xyz = temp->xyz;
              saveCtx(&temp->key, keyLen, &info, infoLen); // 保存ctx
          }
          else{ // 无数据了，输出info，一般info中包含比如显示的个数信息等
              printf(&info);
          }
      }

2. 关键函数
   注意，不能在补丁中注册对象和函数
   1) 注册对象: regObj(objName, objAddr)
   2) 注册函数: regFunc(funcName, func, funcHelpInfo, funcLvl, funcProcObjName, paramCnt, param1, param2, ...)
   3) 函数原型: func(funcProcObjAddr, param1, param2, ...)
2.1 regObj
   display函数一般需要句柄，并且同一个函数操作的句柄可能不止一个，但命令本身不可能直接输入句柄的地址。
   因此，通过{objName, objAddr}映射的方式。
   1) 如果display的时候用户显式输入objName，那取其对应的objAddr作为函数操作的句柄
   2) 否则，取该函数注册过的最小对象名对应的地址。
   举例来说，假定函数原型是 foo(hnd), 这个函数注册过两次：
   regFunc("foo", foo, "objNameAaa")
   regFunc("foo", foo, "objNameBbb")
   如果没有输入对象名，因为"objNameAaa"名字小，那取其对应的地址。
2.2 regFunc
   1) funcParamCount有最大数目限制
   2) 函数参数名字前缀代表类型。目前支持 uni(u32)/pc(字串)。
      如果数字大于u32最大值，取u32最大值
      实际形参名字可以不用和reg时一样，但类型需要匹配（不能注册时是一个数字，函数用时认为是一个字串）。
   3) 会设定函数的lvl
2.3 call func
   假定函数原型是 foo(hnd, arg1, arg2)，即一个句柄两个参数，那用户至少要输入3个字串，第1个是函数名。
   1) 这个函数可能被注册多次，每次除句柄名不一样，其余相同(这是个约束)。假如注册两次:
   regFunc("foo", foo, "objNameAaa", "uniParam1", "pcParam2")
   regFunc("foo", foo, "objNameBbb", "uniParam1", "pcParam2")

   2) 如果用户输入字串第一个不是foo，则找不到函数，提示错误。
   3) 否则，如果用户输入字串数量 < 3，则用户输入字串个数不匹配，提示错误。
   4) 否则，如果用户输入字串数量 = 3， 则输入的后两个字串解析成函数的参数，取foo注册函数中funcHndObjName最小的为
      其句柄参数。上面的例子中，因为"objNameAaa"字串小于"objNameBbb"，取前者。
   5) 否则，取第4个字串，找到其注册的地址
*/

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif
#pragma GCC visibility push(default)
#pragma pack(4)
/* common */

/**
 * @brief 诊断显示库句柄
 */
typedef struct tagBkfDisp BkfDisp;

/* init */
/**
 * @brief 诊断显示库初始化参数
 */
typedef struct tagBkfDispInitArg {
    char *name;             /**< 名称 */
    BkfMemMng *memMng;      /**< 内存库句柄,见bkf_mem.h,同一app内可复用 */
    uint8_t mayProcFuncLvl; /**< 注册函数lvl如果 >= 此值，才会真实生效 */
    uint8_t aucRsv[0x0f];
} BkfDispInitArg;

/* reg func */
/**
 * @brief 测试命令级别，正式版本可关闭测试命令
 */
#define BKF_DISP_FUNC_LVL_TEST ((uint8_t)0x40)
/**
 * @brief 默认命令级别
 */
#define BKF_DISP_FUNC_LVL_DEFAULT ((uint8_t)0x80)
/* display通用函数原型 */
typedef void (*F_BKF_DISP_FUNC)();

/* proc */
/**
 * @brief 命令最大参数个数
 */
#define BKF_DISP_FUNC_PARAM_CNT_MAX 5

/**
 * @brief 命令行输出缓冲最大长度,单位:字节
 */
#define BKF_DISP_PROC_OUT_BUF_LEN_MIN 1024

/**
 * @brief 打印缓冲最大长度,单位:字节
 */
#define BKF_DISP_PRINTF_BUF_LEN_MAX (BKF_1K * 3)

/**
 * @brief 上下文信息最大长度,单位:字节
 */
#define BKF_DISP_CTX_LEN_MAX (1024)

/**
 * @brief 命令行输出上下文信息
 */
typedef struct tagBkfDispProcCtx {
    int32_t ctx1Len;                   /**< 上下文1长度 */
    int32_t ctx2Len;                   /**< 上下文2长度 */
    uint8_t buf[BKF_DISP_CTX_LEN_MAX]; /**< 上下文内容 */
} BkfDispProcCtx;

/**
 * @brief 有剩余信息未输出完成
 */
#define BKF_DISP_PROC_HAS_MORE_DATA ((uint32_t)1)

/**
 * @brief 临时上下文信息最大长度，单位:字节
 */
#define BKF_DISP_TEMP_CTX_LEN_MAX (1024)

/**
 * @brief 临时上下文信息，方便app使用
 */
typedef struct tagBkfDispTempCtx {
    uint8_t buf[BKF_DISP_TEMP_CTX_LEN_MAX];
} BkfDispTempCtx;
BKF_STATIC_ASSERT(BKF_MBR_IS_FIRST(BkfDispTempCtx, buf[0]));

/**
 * @brief 临时上下文信息初始化宏
 */
#define BKF_DISP_TEMP_CTX_INIT(tempCtx)                                                                                \
    do {                                                                                                               \
        (void)BkfDispInitTempCtx(tempCtx);                                                                             \
    } while (0)

/**
 * @brief 临时上下文信息转换成对应类型
 */
#define BKF_DISP_TMEP_CTX_2X(tempCtx, type) ((type *)(tempCtx))

/* func */
/**
 * @brief 诊断显示库初始化
 *
 * @param[in] *arg 参数
 * @return 显示诊断库句柄
 *   @retval 非空 创建成功
 *   @retval 空 创建失败
 */
BkfDisp *BkfDispInit(BkfDispInitArg *arg);

/**
 * @brief 诊断显示库反初始化
 *
 * @param[in] *disp 诊断显示库句柄
 * @return none
 */
void BkfDispUninit(BkfDisp *disp);

/**
 * @brief 注册对象。用于获取字串对应的句柄地址。只能在初始化阶段调用。
 *
 * @param[in] *disp 诊断显示库句柄
 * @param[in] *objName 对象名
 * @param[in] *objAddr 对象地址
 * @return 注册对象结果
 *   @retval BKF_OK 成功
 *   @retval 其他 失败
 */
uint32_t BkfDispRegObj(BkfDisp *disp, char *objName, void *objAddr);

/**
 * @brief 去注册对象及相关注册的函数。
 *
 * @param[in] *disp 诊断显示库句柄
 * @param[in] *objName 对象名
 * @return none
 */
void BkfDispUnregObj(BkfDisp *disp, char *objName);

/**
 * @brief 注册显示函数。只能在初始化阶段调用。一般不直接使用，使用后面的宏封装。
 * @see BKF_DISP_REG_FUNC @see BKF_DISP_REG_TEST_FUNC
 *
 * @param[in] *disp 诊断显示库句柄
 * @param[in] *funcName 一般使用函数名。也可以使用仅包含字母数字的字串。用于获取字串对应的函数。
 * @param[in] funcAddr 函数地址
 * @param[in] *funcHelpInfo 函数帮助信息
 * @param[in] funcLvl 函数的级别。只有大于等于init时设定的函数级别，才会生效。
 * @see struct tagBkfDispInitArg::mayProcFuncLvl
 * @param[in] *funcMayProcObjName 函数操作的对象名。用于BkfDispProc时获取名字对应的地址。
 * @param[in] funcParamCnt 函数参数个数
 * @return 注册对象结果
 *   @retval BKF_OK 成功
 *   @retval 其他 失败
 * @note 此函数中，所有const的字串，一定要是一个“不能释放”的常量串。后续的...参数是函数的各形参，也需要是常量串。
 */
uint32_t BkfDispRegFunc(BkfDisp *disp, char *funcName, F_BKF_DISP_FUNC funcAddr, const char *funcHelpInfo,
    uint8_t funcLvl, char *funcMayProcObjName, uint8_t funcParamCnt, ...);

/**
 * @brief 诊断显示库处理函数。用户输入若干字串后，引起对应字串函数的调用，输出内部信息。
 *
 * @param[in] *disp 诊断显示库句柄
 * @param[in] **inStr 输入的若干字串
 * @param[in] inStrCnt 输入字串数量
 * @param[in] *outStrBuf 输出字串的buffer
 * @param[in] outStrBufLen 输出造船buffer的长度
 * @param[in] *ctx 需要保存的上下文信息。如果是第一次调用（非分包的后续调用），一定要初始化为0
 * @return 处理结果
 *   @retval BKF_DISP_PROC_FUNC_NOT_FIND 没有找到字串对应的函数
 *   @retval BKF_DISP_PROC_FINISH 处理结束
 *   @retval BKF_DISP_PROC_HAS_MORE_DATA 没有处理结束，还有更多的数据需要输出。
 * @see BkfDispRegFunc
 */
uint32_t BkfDispProc(BkfDisp *disp, char **inStr, uint8_t inStrCnt, char *outStrBuf, int32_t outStrBufLen,
    BkfDispProcCtx *ctx);

/**
 * @brief 输出字串。一般不直接使用，使用后面的宏封装。@see BKF_DISP_PRINTF
 *
 * @param[in] *disp 诊断显示库句柄
 * @param[in] *fmt 格式化字串。
 * @param[in] ... 格式化字串相关的可变参数
 * @return none
 */
void BkfDispPrintf(BkfDisp *disp, const char *fmt, ...);

/**
 * @brief 保存断点信息。一般不直接使用，使用后面的宏封装。@see BKF_DISP_SAVE_CTX
 *
 * @param[in] *disp 诊断显示库句柄
 * @param[in] *ctx1 断点信息1
 * @param[in] *ctx1Len 断点信息1长度
 * @param[in] *ctx2OrNull 断点信息2
 * @param[in] .ctx2Len 断点信息2长度
 * @return none
 */
void BkfDispSaveCtx(BkfDisp *disp, void *ctx1, int32_t ctx1Len, void *ctx2OrNull, int32_t ctx2Len);

/**
 * @brief 获取断点信息。一般不直接使用，使用后面的宏封装。@see BKF_DISP_GET_LAST_CTX
 *
 * @param[in] *disp 诊断显示库句柄
 * @param[in] *ctx2OrNull 断点信息2输出地址。如果非空，会将断点信息2拷贝到地址开始的内存中。
 * @return 断点信息1
 *   @retval 非空 有断点信息
 *   @retval 空 无断点信息
 */
void *BkfDispGetLastCtx(BkfDisp *disp, void *ctx2OrNull);

/**
 * @brief 保存最多3个数字的断点信息。方便使用的封装。一般不直接使用，使用后面的宏封装。@see BKF_DISP_SAVE_3NUM
 *
 * @param[in] *disp 诊断显示库句柄
 * @param[in] *val1 数字1
 * @param[in] *val2 数字2
 * @param[in] *val3 数字3
 * @return none
 */
void BkfDispSave3Num(BkfDisp *disp, int32_t val1, int32_t val2, int32_t val3);

/**
 * @brief 获取最多3个数字的断点信息。方便使用的封装。一般不直接使用，使用后面的宏封装。@see BKF_DISP_SAVE_3NUM
 *
 * @param[in] *disp 诊断显示库句柄
 * @param[in] *val1Out 数字1的输出地址
 * @param[in] *val2OutOrNull 数字2的输出地址。可以为空，代表不输出。
 * @param[in] *val3OutOrNull 数字3的输出地址。可以为空，代表不输出。
 * @return 断点信息是否获取陈宫
 *   @retval true 成功
 *   @retval false 失败
 */
BOOL BkfDispGetLast3Num(BkfDisp *disp, int32_t *val1Out, int32_t *val2OutOrNull, int32_t *val3OutOrNull);

/**
 * @brief 初始化 @see BkfDispTempCtx ，清零
 *
 * @param[in] *tempCtx 零时缓冲
 * @return 无
 */
uint32_t BkfDispInitTempCtx(BkfDispTempCtx *tempCtx);

/**
 * @brief 测试诊断函数的显示流程，格式化字串等是否正确。
 *
 * @param[in] *disp 诊断显示库句柄
 * @param[in] *funcName 函数名。"*"代表任意函数，否需要通过函数名匹配
 * @param[in] *objName 对象名。"*"代表任意对象名，否需要通过对象名匹配
 * @param[in] funcParamCnt 输入参数个数
 * @param[in] ... 参数 "字串"。
 * @return none
 */
void BkfDispTestFunc(BkfDisp *disp, char *funcName, char *objName, uint8_t funcParamCnt, ...);

/* 宏，方便使用 */
/**
 * @brief 注册默认级别的显示函数。易用性封装。@see BkfDispRegFunc
 */
#define BKF_DISP_REG_FUNC(disp, func, funcHelpInfo, funcProcObjName, funcParamCount, ...)                              \
    BkfDispRegFunc((disp), (char *)(#func), (func), (funcHelpInfo), (BKF_DISP_FUNC_LVL_DEFAULT), (funcProcObjName),    \
        (funcParamCount), ##__VA_ARGS__)

/**
 * @brief 注册默认级别的显示函数。易用性封装。@see BkfDispRegFunc
 */
#define BKF_DISP_REG_TEST_FUNC(disp, func, funcHelpInfo, funcProcObjName, funcParamCount, ...)                         \
    BkfDispRegFunc((disp), (char *)(#func), (func), (funcHelpInfo), (BKF_DISP_FUNC_LVL_TEST), (funcProcObjName),       \
        (funcParamCount), ##__VA_ARGS__)

/**
 * @brief 输出字串。易用性封装。@see BkfDispPrintf
 */
#define BKF_DISP_PRINTF(disp, fmt, ...)                                                                                \
    do {                                                                                                               \
        BkfDispPrintf((disp), fmt, ##__VA_ARGS__);                                                                     \
    } while (0)

/**
 * @brief 保存断点信息。简单封装。@see BkfDispSaveCtx
 */
#define BKF_DISP_SAVE_CTX(disp, ctx1, ctx1Len, ctx2OrNull, ctx2Len)                                                    \
    do {                                                                                                               \
        BkfDispSaveCtx((disp), (ctx1), (ctx1Len), (ctx2OrNull), (ctx2Len));                                            \
    } while (0)

/**
 * @brief 获取断点信息。简单封装。@see BkfDispGetLastCtx
 */
#define BKF_DISP_GET_LAST_CTX(disp, ctx2OrNull) BkfDispGetLastCtx((disp), (ctx2OrNull))

/**
 * @brief 保存3个数字的断点信息。简单封装。@see BkfDispSave3Num
 */
#define BKF_DISP_SAVE_3NUM(disp, val1, val2, val3)                                                                     \
    do {                                                                                                               \
        BkfDispSave3Num((disp), (val1), (val2), (val3));                                                               \
    } while (0)

/**
 * @brief 获取3个数字的断点信息。简单封装。@see BkfDispGetLast3Num
 */
#define BKF_DISP_GET_LAST_3NUM(disp, val1Out, val2OutOrNull, val3OutOrNull)                                            \
    BkfDispGetLast3Num((disp), (val1Out), (val2OutOrNull), (val3OutOrNull))

#pragma pack()
#pragma GCC visibility pop
#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif

#endif
