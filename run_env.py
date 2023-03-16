def run_env(env, policies, p):
    for i in range(p.time_steps):
        state = env.state()
        actions = []
        for i, policy in enumerate(policies):
            action = policy(state[i]).detach().numpy()
            actions.append(action)

        env.action(actions)

    return env.G()