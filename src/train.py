# Copyright 2024 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from llamafactory.train.tuner import run_exp
import wandb
import os

os.environ["NCCL_DEBUG"] = "WARN"  # INFO 或 "WARN" 或 "ERROR" 以减少输出量
os.environ["NCCL_SOCKET_IFNAME"] = "ens10f0np0"  # 替换为你的网络接口名
# 添加以下变量速度非常慢
# os.environ["NCCL_IB_DISABLE"] = "1"  # 禁用IB网络，如果不使用IB网络的话
# os.environ["NCCL_NET_GDR_LEVEL"] = "5"

os.environ["NCCL_P2P_LEVEL"] = "NVL" #NVLINK

# wandb 登录
# api_key = "60a763e8c62e44eff794afc94793f43732caa063"
api_key = "904d9b7af10d7a2ea3f3d1be89703a53c20deb47"
os.environ["WANDB_PROJECT"] = "table-sft"
wandb.login(key=api_key)


def main():
    run_exp()


def _mp_fn(index):
    # For xla_spawn (TPUs)
    run_exp()


if __name__ == "__main__":
    main()
