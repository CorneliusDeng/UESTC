import sys
import os
import socket
import setproctitle
import numpy as np
from pathlib import Path
import torch
from config import get_config
from envs.env_wrappers import DummyVecEnv

def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            # 选择连续环境
            # from envs.env_continuous import ContinuousActionEnv
            # env = ContinuousActionEnv()
            
            # 选择离散环境
            from envs.env_discrete import DiscreteActionEnv
            env = DiscreteActionEnv() # 创建环境，里面配置了很多参数，主要是obeservation、action space、share obeservation维度

            env.seed(all_args.seed + rank * 1000)
            return env
        return init_env
    
    return DummyVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            # 选择连续环境
            # from envs.env_continuous import ContinuousActionEnv
            # env = ContinuousActionEnv()

            # 选择离散环境
            from envs.env_discrete import DiscreteActionEnv
            env = DiscreteActionEnv()

            env.seed(all_args.seed + rank * 1000)
            return env
        return init_env
    
    return DummyVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])

# 解析参数
def parse_args(args, parser):
    parser.add_argument("--scenario_name", type=str, default="MyEnv", help="Which scenario to run on")
    parser.add_argument("--num_landmarks", type=int, default=3)
    parser.add_argument("--num_agents", type=int, default=2, help="number of players")

    all_args = parser.parse_known_args(args)[0]

    return all_args


def main(args):
    parser = get_config() # 获取配置文件
    all_args = parse_args(args, parser) # 解析参数

    if all_args.algorithm_name == "rmappo": # 循环神经网络的MAPPO
        assert all_args.use_recurrent_policy or all_args.use_naive_recurrent_policy, "check recurrent policy!"
    elif all_args.algorithm_name == "mappo": # MLP的MAPPO
        assert (all_args.use_recurrent_policy == False and all_args.use_naive_recurrent_policy == False), "check recurrent policy!"
    else:
        raise NotImplementedError

    assert (all_args.share_policy == True and all_args.scenario_name == "simple_speaker_listener") == False, "The simple_speaker_listener scenario can not use shared policy. Please check the config.py."

    # cuda
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads) # 设置线程数
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    # 保存路径
    run_dir = (
        Path(os.path.abspath(os.path.dirname(__file__)) + "/results")
        / all_args.env_name
        / all_args.scenario_name
        / all_args.algorithm_name
        / all_args.experiment_name
    )
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    if not run_dir.exists():
        curr_run = "run1"
    else:
        exst_run_nums = [
            int(str(folder.name).split("run")[1])
            for folder in run_dir.iterdir()
            if str(folder.name).startswith("run")
        ]
        if len(exst_run_nums) == 0:
            curr_run = "run1"
        else:
            curr_run = "run%i" % (max(exst_run_nums) + 1)
    run_dir = run_dir / curr_run
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    # 将当前进程的名称设置为一个字符串，方便在系统进程列表中查看当前进程的信息
    setproctitle.setproctitle(
        str(all_args.algorithm_name)
        + "-"
        + str(all_args.env_name)
        + "-"
        + str(all_args.experiment_name)
        + "@"
        + str(all_args.user_name)
    )

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # 初始化环境
    envs = make_train_env(all_args) # 创建多个env
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None
    num_agents = all_args.num_agents

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir,
    }

    # run experiments
    if all_args.share_policy: 
        from runner.shared.env_runner import EnvRunner as Runner
    else:
        from runner.separated.env_runner import EnvRunner as Runner

    runner = Runner(config)
    runner.run()

    # post process
    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()

    runner.writter.export_scalars_to_json(str(runner.log_dir + "/summary.json"))
    runner.writter.close()


if __name__ == "__main__":
    main(sys.argv[1:])