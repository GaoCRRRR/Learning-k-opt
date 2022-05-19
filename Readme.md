### 项目代码

本项目实现了基于强化学习的k-opt算法，用于求解旅行商问题。项目代码主要基于[learning-2opt-drl](https://github.com/paulorocosta/learning-2opt-drl)，本项目的主要贡献在于将2-opt算子扩展为任意的k-opt，具体代码实现见ActorCriticNetwork.py，PGTSP.py，PGTSP_2_stages.py，TSPEnvironment.py和utils.py这几个代码文件的改写，其余代码文件与原项目一致。

#### 训练

运行run.sh，该脚本中的命令如下：

```shell
python PGTSP.py --epochs 300 --n_actions 3 --gpu_n 1 --n_points 100
```

两阶段训练时，运行如下命令：

```shell
python PGTSP_2_stages.py best_policy/my_policy_3-opt-TSP100-epoch-289.pt --epochs 200 --n_actions 3 --gpu_n 1 --n_points 100
```

#### 推理

运行test.sh，该脚本中的命令如下：

```shell
python TestLearnedAgent.py --load_path best_policy/my_policy-3-opt-2-stages-TSP100-epoch-172 --n_points 100 --n_actions 3 --gpu_n 1 --n_steps 2000
```
