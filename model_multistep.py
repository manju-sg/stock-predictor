import pandas as pd
import numpy as np
import xgboost as xgb
from model_xgboost import XGBoostPredictor

class MultiStepForecaster:
    def __init__(self, steps=7):
        self.steps = steps
        self.models = {}
        self.features = []

    def train_all(self, df, features_list, target_base_col='close'):
        self.features = features_list
        print(f"Training {self.steps} independent mathematical XGBoost models for direct multi-step forecasting...")
        
        for step in range(1, self.steps + 1):
            # Create the dynamic target offset by `step` days
            df_step = df.copy()
            target_col = f'target_{step}d'
            df_step[target_col] = df_step[target_base_col].shift(-step)
            
            # Drop trailing NaNs resulting from future shift
            df_step = df_step.dropna(subset=[target_col] + self.features).reset_index(drop=True)
            
            if df_step.empty:
                print(f"Warning: Not enough data for step {step}")
                continue
                
            X = df_step[self.features]
            y = df_step[target_col]
            
            # Optimize memory & runtime for quick inferencing
            model = xgb.XGBRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42 + step
            )
            model.fit(X, y)
            self.models[step] = model
            print(f" - Model {step} (predicting T+{step}) trained.")

    def forecast_from_latest(self, latest_row_df):
        """
        Given the single most recent row of features, predict the next N days.
        """
        predictions = []
        X_latest = latest_row_df[self.features]
        
        for step in range(1, self.steps + 1):
            if step in self.models:
                pred = self.models[step].predict(X_latest)[0]
                predictions.append(float(pred))
            else:
                predictions.append(None)
                
        return predictions
