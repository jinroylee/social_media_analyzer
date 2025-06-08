from sklearn.metrics import mean_absolute_error
from scipy.stats import spearmanr

def evaluate_predictions(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rho, _ = spearmanr(y_true, y_pred)
    return mae, rho