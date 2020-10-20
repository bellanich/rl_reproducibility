
DISCOUNT_FACTORS = [0.90, 0.95, 0.99]
LEARNING_RATES = [0.001, 0.0001, 0.005]
SEEDS = [i for i in range(50, 55)]
ENVIRONMENTS = ["CartPole-v1", "Acrobot-v1"] 
POLICIES = ["reinforce", "gpomdp"]  # Options: "gpomdp", "reinforce"
BASELINES = [None, "normalized_baseline"]  # Options: None, random_baseline, normalized_baseline

NUM_EPISODES = 10000
HIDDEN_LAYERS = 128
SAMPLING_FREQ = 200

def grid_search_configurations():
    for env in ENVIRONMENTS:
        for policy in POLICIES:
            for lr in LEARNING_RATES:
                for df in DISCOUNT_FACTORS:
                    for seed in SEEDS:
                        for baseline in BASELINES:
                            if policy == "reinforce" and baseline == "normalized_baseline":
                                continue
                            config = {
                                "environment" : env,
                                "policy" : policy,
                                "learning_rate" : lr,
                                "discount_factor" : df,
                                "seed" : seed,
                                "num_episodes" : NUM_EPISODES,
                                "hidden_layer" : HIDDEN_LAYERS,
                                "sampling_freq": SAMPLING_FREQ,
                                "baseline": baseline
                            }
                            yield config
