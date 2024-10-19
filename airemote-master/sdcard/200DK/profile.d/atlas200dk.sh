# /etc/profile.d/atlas200dk.sh with `755`

alias ll='ls -l --color'
alias python='python3.7.5'

export TZ='CST-8'
export CPU_ARCH=aarch64
export INSTALL_DIR=/usr/local/Ascend/
export THIRDPART_PATH=/home/haojiash/samples/cplusplus/common
export PATH=/usr/local/python3.7.5/bin:$PATH
export PYTHONPATH=/usr/local/lib/python3.7/site-packages/:$PYTHONPATH

export install_path=/usr/local/Ascend/ascend-toolkit/latest
export PATH=${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
export PYTHONPATH=/usr/local/Ascend/pyACL/python/site-packages/acl:$PYTHONPATH
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend_ddk/arm/lib:/usr/local/Ascend/acllib/lib64:$LD_LIBRARY_PATH

export ASCEND_OPP_PATH=${install_path}/opp
export ASCEND_AICPU_PATH=${install_path}

export DDK_HOME=/usr/local/Ascend/ascend-toolkit/latest/arm64-linux
export DDK_PATH=/usr/local/Ascend/ascend-toolkit/latest/
export NPU_HOST_LIB=/usr/local/Ascend/ascend-toolkit/latest/acllib/lib64/stub/
export ATLAS_UTILS_PATH='/usr/local/python3.7.5/lib/yacl'
