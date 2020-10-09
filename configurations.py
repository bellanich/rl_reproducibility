
DISCOUNT_FACTORS = [0.99]
LEARNING_RATES = [0.001]
SEEDS = [42]
ENVIRONMENTS = ["CartPole-v1", "Acrobot-v1"] #  , , "MountainCar-v0"
POLICIES = ["gpomdp", "reinforce"]

NUM_EPISODES = 2000
HIDDEN_LAYERS = 128
sampling_freq = NUM_EPISODES // 10

def grid_search_configurations():
    for env in ENVIRONMENTS:
        for policy in POLICIES:
            for lr in LEARNING_RATES:
                for df in DISCOUNT_FACTORS:
                    for seed in SEEDS:
                        config = {
                            "environment" : env,
                            "policy" : policy,
                            "learning_rate" : lr,
                            "discount_factor" : df,
                            "seed" : seed,
                            "num_episodes" : NUM_EPISODES,
                            "hidden_layer" : HIDDEN_LAYERS,
                            "sampling_freq": sampling_freq,
                        }
                        yield config
