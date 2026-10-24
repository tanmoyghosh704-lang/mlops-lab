import pandas as pd
import yaml

from pathlib import Path
from sklearn.model_selection import train_test_split


def prepare_data(input_path, train_path, test_path, test_size, seed):

    # 1. Load validated data
    df = pd.read_csv(input_path)

    # 2. Separate features and target
    X = df.drop(columns=["units"])
    y = df["units"]

    # 3. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed
    )

    # 4. Combine features and target
    train_df = X_train.copy()
    train_df["units"] = y_train

    test_df = X_test.copy()
    test_df["units"] = y_test

    # 5. Create output directories
    train_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.parent.mkdir(parents=True, exist_ok=True)

    # 6. Save datasets
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("DATA PREPARATION COMPLETED")
    print(f"Training samples: {len(train_df)}")
    print(f"Testing samples: {len(test_df)}")


if __name__ == "__main__":

    # Load configuration
    with open("params.yaml", "r") as file:
        config = yaml.safe_load(file)

    prepare_params = config["prepare"]

    prepare_data(
        input_path=Path("data/validated/sales.csv"),
        train_path=Path("data/processed/train.csv"),
        test_path=Path("data/processed/test.csv"),
        test_size=prepare_params["test_size"],
        seed=prepare_params["seed"]
    )