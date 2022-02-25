export OLD_PS1=${OLD_PS1:-${PS1}}
export PS1="(DietCode) ${OLD_PS1}"
export USE_DIETCODE=1
export OLD_LD_LIBRARY_PATH=${OLD_LD_LIBRARY_PATH:-${LD_LIBRARY_PATH}}
export LD_LIBRARY_PATH=${OLD_LD_LIBRARY_PATH}:/mnt/tvm/build

PYTHONVER=$(python3 -c 'import sys; print(str(sys.version_info[0]) + "." + str(sys.version_info[1]))')

export PYTHONPATH=/mnt/tvm/python/build/lib.linux-$(uname -m)-${PYTHONVER}/:/
