import pandas as pd
from pathlib import Path


def validate_sales_data(file_path):

    df = pd.read_csv(file_path)

    errors = []

    # 1. Check required columns
    required_columns = [
        "day_of_week",
        "price",
        "promo",
        "temperature",
        "units"
    ]

    missing_columns = set(required_columns) - set(df.columns)

    if missing_columns:
        errors.append(f"Missing columns: {missing_columns}")

    # Stop further checks if required columns are missing
    if missing_columns:
        raise ValueError("\n".join(errors))

    # 2. Check missing values
    if df.isnull().sum().sum() > 0:
        errors.append(
            f"Missing values:\n{df.isnull().sum()}"
        )

    # 3. Check duplicate rows
    if df.duplicated().any():
        errors.append(
            f"Found {df.duplicated().sum()} duplicate rows"
        )

    # 4. Check numeric columns
    numeric_columns = required_columns

    for col in numeric_columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            errors.append(f"{col} must be numeric")

    # 5. Check business rules / valid ranges

    if not df["day_of_week"].between(0, 6).all():
        errors.append("day_of_week must be between 0 and 6")

    if not df["price"].between(0, 100).all():
        errors.append("price must be between 0 and 100")

    if not df["promo"].isin([0, 1]).all():
        errors.append("promo must contain only 0 or 1")

    if not df["temperature"].between(-50, 60).all():
        errors.append("temperature must be between -50 and 60")

    if (df["units"] < 0).any():
        errors.append("units cannot be negative")

    # 6. Report validation result

    if errors:
        print("DATA VALIDATION FAILED")

        for error in errors:
            print("-", error)

        raise ValueError("Please fix the data quality issues.")

    print("DATA VALIDATION PASSED")
    print(f"Number of rows: {len(df)}")
    print(f"Number of columns: {len(df.columns)}")

    return df


if __name__ == "__main__":

    base_dir = Path(__file__).resolve().parent.parent

    input_path = base_dir / "data/raw/sales.csv"

    df = validate_sales_data(input_path)

    # Save validated data
    output_path = base_dir / "data/validated/sales.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)

    print(f"Validated data saved to: {output_path}")