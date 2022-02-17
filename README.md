# DietCode

## Installation

- Clone the project by
  ```Bash
  git clone https://github.com/UofT-EcoSystem/DietCode -b MLSys2022_AE
  ```
- Install docker-compose, which is a wrapper on top of Docker.
  ```Bash
  sudo -H pip3 install docker-compose
  ```
- Build the Docker image that includes all the software dependencies required to
  run the experiments:
  ```Bash
  DietCode$ docker-compose build tvm-dev
  ```
- Create a running container out of the image:
  ```Bash
  DietCode$ docker-compose run --rm tvm-dev
  ```
- Build the DietCode and the TVM baseline.
  ```Bash
  /mnt$ ./scripts/1-compile.sh tvm
  /mnt$ ./scripts/1-compile.sh tvm_base
  ```

## Experiments

- Dense Layer with Dynamic Sequence Length (Section 5.3 of the main text)
  ```Bash
  /mnt$ ./scripts/2_1-experiment_dynamic_dense.sh
  ```
- BatchMatmul Layer with Dynamic Sequence Length (Section 5.4 of the main text)
  ```Bash
  /mnt$ ./scripts/2_2-experiment_dynamic_batch_matmul_nt.sh
  /mnt$ ./scripts/2_3-experiment_dynamic_batch_matmul_nn.sh
  ```
- BERT with Various Sequence Lengths (Section 5.2)
  ```Bash
  /mnt$ ./scripts/2_4-experiment_bert.sh
  ```

## Evaluation and Expected Results

After each experiment has been run, a CSV file named `temp_workspace.csv` will
be generated in each folder `ops/dense`, `ops/batch_matmul`, and `networks/bert`
respectively that reports the latency numbers (in seconds, the lower the
better). At the same time, `dietcode_autosched_timer.csv` (or
`ansor_autosched_timer.csv` if one is running the Ansor baseline) will be
generated in the same folder that reports the time to complete the
auto-scheduling process (also in seconds, the lower the better).

## Notes

With each experiment, the Ansor baseline is already provided, but can be
reobtained using the provided `./scripts/*_ansor_baseline.sh` script files. Note
that the entire auto-scheduling workflow takes time to complete. Therefore, we
one can use the 
```Bash
AUTO_SCHED_NTRIALS=200 ./scripts/...
```
prefix that uses fewer number auto-scheduling trials. The resulting tensor
programs will still be functionally correct but the performance can be
sub-optimal.

