import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score

class XGBoostPredictor:
    def __init__(self, target_col, features):
        self.model = xgb.XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        self.target_col = target_col
        self.features = features

    def train(self, df_train):
        """
        Trains the XGBoost model on the provided training dataframe.
        """
        X = df_train[self.features]
        y = df_train[self.target_col]
        self.model.fit(X, y)
        print("XGBoost training completed.")

    def predict(self, df_test):
        """
        Predicts future prices for the testing dataframe.
        """
        X = df_test[self.features]
        return self.model.predict(X)

    def evaluate(self, df_test, predictions, current_price_col='close'):
        """
        Evaluates the predictions against the actual target values.
        Calculates RMSE, MAE, and Directional Accuracy.
        """
        y_true = df_test[self.target_col].values
        
        # Calculate regression metrics
        rmse = np.sqrt(mean_squared_error(y_true, predictions))
        mae = mean_absolute_error(y_true, predictions)
        
        # Calculate Directional Accuracy
        # 1 if price went up, 0 if it went down
        actual_movement = (y_true > df_test[current_price_col].values).astype(int)
        predicted_movement = (predictions > df_test[current_price_col].values).astype(int)
        dir_acc = accuracy_score(actual_movement, predicted_movement)
        
        return {
            "RMSE": rmse,
            "MAE": mae,
            "Directional Accuracy": dir_acc
        }
