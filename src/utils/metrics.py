from sklearn.metrics import cohen_kappa_score


def quadratic_weighted_kappa(y_hat, y):
    return cohen_kappa_score(y_hat, y, weights="quadratic")
