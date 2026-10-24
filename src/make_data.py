import argparse
import os

import numpy as np
import pandas as pd


# 1. Define command-line arguments

parser = argparse.ArgumentParser()

parser.add_argument(
    "--n",
    type=int,
    default=5000
)

parser.add_argument(
    "--seed",
    type=int,
    default=42
)

parser.add_argument(
    "--drift",
    choices=["none", "data", "concept"],
    default="none"
)

parser.add_argument(
    "--out",
    default="data/raw/sales.csv"
)

args = parser.parse_args()


# 2. Initialize random generator

rng = np.random.default_rng(args.seed)


# 3. Introduce different drift conditions

# Data drift: price distribution changes
price_mean = 13.0 if args.drift == "data" else 10.0

# Concept drift: relationship between promotion and sales changes
promo_effect = 0.0 if args.drift == "concept" else 8.0


# 4. Generate input features

df = pd.DataFrame({
    "day_of_week": rng.integers(0, 7, args.n),

    "price": rng.normal(
        price_mean, 2.0, args.n
    ).round(2),

    "promo": rng.binomial(
        1, 0.3, args.n
    ),

    "temperature": rng.normal(
        28, 4, args.n
    ).round(1)
})


# 5. Generate target variable

df["units"] = (
    50
    + promo_effect * df["promo"]
    - 3 * (df["price"] - 10)
    + 5 * (df["day_of_week"] >= 5)
    + 0.5 * (df["temperature"] - 28)
    + rng.normal(0, 4, args.n)
).clip(lower=0).round()


# 6. Save the generated dataset

os.makedirs(
    os.path.dirname(args.out),
    exist_ok=True
)

df.to_csv(args.out, index=False)

print(f"Generated {len(df)} rows")
print(f"Drift condition: {args.drift}")
print(f"Saved to: {args.out}")