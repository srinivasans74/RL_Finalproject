from envs.half_cheetah_v4 import HalfCheetahEnv

def get_env_mujoco_config(args):
    # env = gym.make(args.env_name)
    # todo: create new environments
    if args.env_name == "HalfCheetah-v4":
        env = HalfCheetahEnv(goal_vel=0.3)
        print('\033[0;35m "Create HalfCheetah-v4 Environments!" \033[0m')
    else:
        print('\033[0;31m "error! Please input a correct task name!" \033[0m')
    return env
