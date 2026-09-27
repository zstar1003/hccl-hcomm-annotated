编译命令如下：
1）同时编译hccl库和算法库
mkdir tmp && cd tmp && cmake ../cmake/superbuild -DHOST_PACKAGE=fwkacllib && TARGETS=hccl_v2 make -j64 host


编译后二进制所在目录：
1）算法模板和算法框架的so位置
./host-prefix/src/host-build/ace/comop/hccl/whole/hccl/v2/service/collective/alg/libhccl_v2.so
