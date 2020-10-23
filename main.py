"""This is a script such that imports can be shared between the two different
ways of training."""

TRAIN_WITH = "baseline_gpomdp"

if TRAIN_WITH == "baseline_gpomdp":
    import train_with_baseline_gpomdp.main
elif TRAIN_WITH == "policies":
    import train_with_policies.main
