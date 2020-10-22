
DISCOUNT_FACTORS = [0.99] # 0.90, 0.95,
LEARNING_RATES = [0.0001] #  0.0001, 0.001, 0.01
SEEDS = [42] # i for i in range(50, 55)
ENVIRONMENTS = ["GridWorld"]  # Options: "GridWorld", "CartPole-v1"
POLICIES = ["gpomdp", "reinforce", "normalized_gpomdp"]  # Options: "gpomdp", "reinforce", "normalized_gpomdp"
# TRAIN_WITH_POLICIES = [True, False]

NUM_EPISODES = 10 #800
HIDDEN_LAYERS = 128
SAMPLING_FREQ = 2 #100

def grid_search_configurations():
    for env in ENVIRONMENTS:
        # for policy in POLICIES:
            for lr in LEARNING_RATES:
                for df in DISCOUNT_FACTORS:
                    for seed in SEEDS:
                        for train_with_policy in TRAIN_WITH_POLICIES:
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
                                "policies": POLICIES,
                                # Switch that determines whether or not we train with each policy function or only with
                                #   GPOMDP + whitening.
                                # "train_with_policies": train_with_policy
                            }
                            yield config
