# <img src="./figures/DietCode_text.png" alt="DietCode" height="48"></img> [![RTX 3090 desktop](https://github.com/UofT-EcoSystem/DietCode/actions/workflows/rtx_3090_desktop.yml/badge.svg)](https://github.com/UofT-EcoSystem/DietCode/actions/workflows/rtx_3090_desktop.yml) [![RTX 2080 Ti server](https://github.com/UofT-EcoSystem/DietCode/actions/workflows/rtx_2080_ti_server.yml/badge.svg)](https://github.com/UofT-EcoSystem/DietCode/actions/workflows/rtx_2080_ti_server.yml)

Thank you for your interest in the <img src="./figures/DietCode_text.png"
alt="DietCode" height="16"></img> project. The key objective of *DietCode* is to
deliver high-performance programs for dynamic-shape tensor programs. Please
kindly go through the documentation below that guides you on how to build and
use the *DietCode* auto-scheduler framework.

:point_up: *Please refer to the top-left corner of the README for the table of
contents*.

## Prerequisites

- Minimum Requirement: A machine with a modern NVIDIA GPU and GPU driver
  version >= **465.19.01** (for CUDA 11.3 in
  [the Dockerfile](./dockerfiles/tvm.Dockerfile)) [1].

- Docker & Docker-Compose [2, 3]:

  ```Bash
  # Docker
  curl https://get.docker.com | sh && sudo systemctl --now enable docker
  ```

  ```Bash
  # Docker Post-Installation
  sudo groupadd docker
  sudo usermod -aG docker $USER
  newgrp docker
  ```

  ```Bash
  # NVIDIA Docker
  ./scripts/0-install_nvidia_docker.sh
  ```

  ```Bash
  # Docker-Compose
  sudo -H pip3 install docker-compose
  ```

## Code Organization

- [**`tests`**](./tests): Contains test cases for testing code generation
  changes (i.e., local padding & decision-tree dispatching) and benchmarking
  performance. Please click on the folder for more details.
  
- [**`tvm`**](./tvm): <img src="./figures/DietCode_text.png" alt="DietCode" height="16"></img>
  that is built on top of TVM. Please refer to
  [this page](https://github.com/UofT-EcoSystem/tvm/compare/bojian/DietCode_base...bojian/DietCode/stable)
  for the changes made at this branch.

- [**`tvm_base`**](./tvm_base): A clean TVM branch (denoted as *Base*) that has
  only a few changes (for performance benchmarking only). Please refer to
  [this page](https://github.com/UofT-EcoSystem/tvm/compare/bojian/DietCode_base...bojian/DietCode/base)
  for the changes made at this branch.

- [**`environ`**](./environ): Script files that can be activated to select
  different TVM branches (*DietCode* or *Base*). **Note that at least one of the
  scripts MUST be activated** before running the tests (some require the
  *DietCode* branch while others require *Base*):

  ```Bash
  # DietCode
  source environ/activate_dietcode.sh
  ```

  ```Bash
  # Base
  source environ/activate_base.sh
  ```

## Build Instructions

- Build and Run the Docker image:

  ```Bash
  docker-compose run --rm tvm-dev
  ```

- Build *DietCode* and *Base*:

  ```Bash
  # DietCode
  ./scripts/1-build.sh tvm
  ```

  ```Bash
  # Base
  ./scripts/1-build.sh tvm_base
  ```

## How To?

- **Get started with
  <img src="./figures/DietCode_text.png" alt="DietCode" height="16"></img>?**
  - Please refer to the [**`test`**](./test) folder that contains examples
    demonstrating the code generation optimizations and the auto-scheduler
    frontend interface of *DietCode*.

- **Know the implementation details of
  <img src="./figures/DietCode_text.png" alt="DietCode" height="16"></img>?**
  - Please refer to the [**`tvm`**](./tvm) submodule and
    [this page](https://github.com/UofT-EcoSystem/tvm/compare/bojian/DietCode_base...bojian/DietCode/stable)
    for the changes made at this branch. We did our best to document the code
    thoroughly, with the key changes highlighted using the `<DietCode>` tag.

## References

- [1] CUDA Compatibility. https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cuda-major-component-versions
- [2] Docker Post-Installation Steps. https://docs.docker.com/engine/install/linux-postinstall/#manage-docker-as-a-non-root-user
- [3] NVIDIA Docker Installation Steps. https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#installing-on-ubuntu-and-debian
