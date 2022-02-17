export USE_TVM_BASE=0
export OLD_LD_LIBRARY_PATH=${OLD_LD_LIBRARY_PATH:-${LD_LIBRARY_PATH}}
export LD_LIBRARY_PATH=${OLD_LD_LIBRARY_PATH}:/mnt/tvm/build

export PYTHONPATH=/mnt/tvm/python/build/lib.linux-x86_64-3.8/:/
