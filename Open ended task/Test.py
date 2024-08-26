from scipy.stats import wilcoxon

alpha = 0.01

# We use a one-sided Wilcoxon signed-rank test, where the alternative is less
def test(x, y):
    return wilcoxon(x, y, correction=True, alternative='less')

# Applying Bonferroni correction to counteract the multiple hypotheses problem
def bonferroni_correction(alpha, num_hypotheses):
    return alpha / num_hypotheses