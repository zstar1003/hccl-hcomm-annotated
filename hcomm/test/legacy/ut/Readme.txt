如何执行LLT

1）删除历史目录，从头开始执行（命令执行路径为代码根目录）
rm -rf tmp && mkdir tmp && cd tmp && cmake ../cmake/superbuild/ -DCUSTOM_PYTHON=python3 -DHOST_PACKAGE=ut -DBUILD_MOD=hccl_v2 -DFULL_COVERAGE=true -DCOVERAGE_RC_CONFIG=true && make -j16

2）不删除历史目录，再次执行（适用于调试代码的场景）(执行目录为tmp目录)
make -j16
