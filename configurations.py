
DISCOUNT_FACTORS = [0.99] # 0.90, 0.95,
LEARNING_RATES = [0.0001, 0.001, 0.01] #  0.0001, 0.01
SEEDS = [i for i in range(50, 55)] #i for i in range(50, 60)
ENVIRONMENTS = ["GridWorld", "CartPole-v1"]  # Options: GridWorld, CartPole-v1
POLICIES = ["gpomdp", "reinforce", "normalized_gpomdp"]  # Options: "gpomdp",

NUM_EPISODES = 800 # 10000
HIDDEN_LAYERS = 128
SAMPLING_FREQ = 100 #100

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
                            # Model now only trains using best policy, but validates on different policies. Otherwise
                            #  all our results will be random, since the model hasn't learned anything.
                            "policies": POLICIES
                        }
                        yield config
