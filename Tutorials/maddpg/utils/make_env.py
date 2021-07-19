import numpy as np

def make_env(args):
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    scenario = scenarios.load(args.scenario_name + ".py").Scenario()

    world = scenario.make_world()

    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)

    args.n_players = env.n
    args.n_agents = env.n - args.num_adversaries
    args.obs_shape = [env.observation_space[i].shape[0] for i in range(args.n_agents)]
    action_shape = []
    for content in env.action_space:
        action_shape.append(content.n)
    args.action_shape = action_shape[:args.n_agents]
    args.high_action = 1
    args.low_action = -1
    return env, args
