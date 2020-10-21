
DISCOUNT_FACTORS = [0.99] # 0.90, 0.95,
LEARNING_RATES = [0.001] #  0.0001, 0.01
SEEDS = [42] #i for i in range(55, 60)
ENVIRONMENTS = ["CartPole-v1"]  # GridWorld , "Acrobot-v1"
POLICIES = ["reinforce"]  # Options: "gpomdp", "reinforce", normalized_gpomdp

NUM_EPISODES = 5 # 10000
HIDDEN_LAYERS = 128
SAMPLING_FREQ = 3 #100

def grid_search_configurations():
    for env in ENVIRONMENTS:
        # for policy in POLICIES:
            for lr in LEARNING_RATES:
                for df in DISCOUNT_FACTORS:
                    for seed in SEEDS:
                        # for baseline in BASELINES:
                            # if policy == "reinforce" and baseline == "normalized_baseline":
                            #     continue
                        config = {
                            "environment" : env,
                            "learning_rate" : lr,
                            "discount_factor" : df,
                            "seed" : seed,
                            "num_episodes" : NUM_EPISODES,
                            "hidden_layer" : HIDDEN_LAYERS,
                            "sampling_freq": SAMPLING_FREQ,
                            # Model now only trains using best policy, but validate son different policies. Otherwise
                            #  all our results will be random, since the model hasn't learned anything.
                            "policies": POLICIES
                        }
                        yield config
