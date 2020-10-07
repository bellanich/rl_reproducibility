
DISCOUNT_FACTORS = [0.99]
LEARNING_RATES = [0.001]
SEEDS = [42]
ENVIRONMENTS = ["CartPole-v1", "Acrobot-v1", "MountainCar-v0"] #  #  #Ant-v2
POLICIES = ["gpomdp"] # Originally was 'gpomdb', but I'm pretty sure this was a typo.

NUM_EPISODES = 50
HIDDEN_LAYERS = 128
sampling_freq = 10

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
