export USE_TVM_BASE=1
export OLD_LD_LIBRARY_PATH=${OLD_LD_LIBRARY_PATH:-${LD_LIBRARY_PATH}}
export LD_LIBRARY_PATH=${OLD_LD_LIBRARY_PATH}:/mnt/tvm_base/build

export PYTHONPATH=/mnt/tvm_base/python/build/lib.linux-x86_64-3.8/:/
