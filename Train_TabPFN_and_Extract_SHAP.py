import os
import time
import random
import logging
import argparse
import warnings
import gc

# Limit multi-threading conflicts
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import shap
from tabpfn import TabPFNRegressor

warnings.filterwarnings("ignore")

# ================= 1. Global Configurations =================
TARGETS = ["EOS_PV", "LOS_PV", "LOS_BUF"]

FEATURES = [
    "TEMP", "PR", "PET", "RH", "WS", "GHI",
    "SM", "ST", "DEM", "SLP", "Pre_NDVI", "PV_Area", "AI"
]

GROUP_COL = "PlantID"


# ================= 2. Core Functions =================
def set_seed(seed: int = 42):
    """Fix random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def setup_logger(output_dir: str) -> logging.Logger:
    """Configure logging"""
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, f"experiment_ControlZone_{time.strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[logging.FileHandler(log_file, encoding='utf-8'), logging.StreamHandler()]
    )
    return logging.getLogger(__name__)


def export_shap_results(X_df, shap_matrix, zone_name, output_dir, target, logger):
    """Export ONLY the essential SHAP and feature values to CSV"""
    df_shap_values = pd.DataFrame(shap_matrix, columns=FEATURES)
    df_feature_values = X_df.reset_index(drop=True)

    df_shap_values.to_csv(os.path.join(output_dir, f"SHAP_Values_{target}_{zone_name}.csv"), index=False)
    df_feature_values.to_csv(os.path.join(output_dir, f"Feature_Values_{target}_{zone_name}.csv"), index=False)

    logger.info(f"    - {zone_name} Data export completed (Feature Values & SHAP Values)! (Sample size: {len(X_df)})")


# ================= 3. Main Pipeline =================
def main(args):
    logger = setup_logger(args.output_dir)
    set_seed(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"[{'GPU' if device == 'cuda' else 'CPU'}] Current computation device: {device.upper()}")

    try:
        if args.data_path.endswith('.csv'):
            df_raw = pd.read_csv(args.data_path)
        else:
            df_raw = pd.read_excel(args.data_path)
    except Exception as e:
        logger.error(f"Failed to read data: {e}")
        return

    for target in TARGETS:
        logger.info(f"Processing target variable: {target}")

        if target not in df_raw.columns:
            logger.warning(f"Target column '{target}' not found, skipping.")
            continue

        try:
            target_output_dir = os.path.join(args.output_dir, target)
            os.makedirs(target_output_dir, exist_ok=True)

            cols_to_check = FEATURES + [target, GROUP_COL]
            df = df_raw.dropna(subset=cols_to_check).reset_index(drop=True)

            X = df[FEATURES]
            y = df[target]
            groups = df[GROUP_COL]

            # --- GroupKFold Cross Validation ---
            cv = GroupKFold(n_splits=10)
            oof_actuals = []
            oof_predictions = []

            cv_model = TabPFNRegressor(device=device)

            for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups)):
                X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
                X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

                cv_model.fit(X_train, y_train)
                y_pred = cv_model.predict(X_test)

                oof_actuals.extend(y_test)
                oof_predictions.extend(y_pred)

                gc.collect()
                if device == 'cuda': torch.cuda.empty_cache()

            del cv_model
            gc.collect()
            if device == 'cuda': torch.cuda.empty_cache()

            rmse = np.sqrt(mean_squared_error(oof_actuals, oof_predictions))
            r2 = r2_score(oof_actuals, oof_predictions)
            logger.info(f"CV Global Assessment -> RMSE: {rmse:.4f}, R2: {r2:.4f}")

            # --- Full Data Training & SHAP Extraction ---
            final_model = TabPFNRegressor(device=device)
            final_model.fit(X, y)

            sample_size = min(100, len(X))
            background_data, _ = train_test_split(X, train_size=sample_size, random_state=args.seed)
            explainer = shap.Explainer(final_model.predict, background_data)

            shap_explanation = explainer(X)
            shap_values_matrix = shap_explanation.values

            # Only exporting the two core files
            export_shap_results(X, shap_values_matrix, "ControlZone", target_output_dir, target, logger)

            del final_model
            del explainer
            gc.collect()
            if device == 'cuda': torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"Error processing {target}: {e}", exc_info=True)
            continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TabPFN SHAP Pipeline")
    parser.add_argument("--data_path", type=str, default="./data/PV_Dataset.xlsx")
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--seed", type=int, default=42)

    try:
        args = parser.parse_args()
    except SystemExit:
        args = parser.parse_args(args=[])

    main(args)