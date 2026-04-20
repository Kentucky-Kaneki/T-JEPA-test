"""
Data loading, preprocessing, and encoding for T-JEPA.
Supports the Adult (AD) dataset out-of-the-box and any sklearn-compatible tabular dataset.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


# ─────────────────────────────────────────────
#  Dataset loading
# ─────────────────────────────────────────────

def load_adult() -> tuple[pd.DataFrame, str, list[str], list[str]]:
    """
    Load the Adult Income dataset from OpenML.
    Returns (df, target_col, num_cols, cat_cols).
    """
    print("Downloading Adult dataset from OpenML …")
    data = fetch_openml("adult", version=2, as_frame=True, parser="auto")
    df = data.frame.copy()

    # Standardise target
    target_col = "class"
    df[target_col] = (df[target_col].astype(str).str.strip()
                      .map({"<=50K": 0, ">50K": 1,
                            "<=50K.": 0, ">50K.": 1})
                      .astype(int))
    df = df.dropna().reset_index(drop=True)

    num_cols = [c for c in data.feature_names
                if df[c].dtype in [np.float64, np.float32, np.int64, np.int32]]
    cat_cols = [c for c in data.feature_names if c not in num_cols]
    return df, target_col, num_cols, cat_cols


# ─────────────────────────────────────────────
#  Preprocessor
# ─────────────────────────────────────────────

class TabularPreprocessor:
    """
    Fits scalers/encoders on training data.
    Encodes samples into per-feature tensors as required by T-JEPA.
    """
    def __init__(self, num_cols: list[str], cat_cols: list[str]):
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.scalers: dict[str, StandardScaler] = {}
        self.encoders: dict[str, LabelEncoder]  = {}
        self.feature_order: list[str] = []
        self.feature_dims: list[int]  = []

    def fit(self, df: pd.DataFrame):
        self.feature_order = self.num_cols + self.cat_cols
        self.feature_dims  = []

        for col in self.num_cols:
            sc = StandardScaler()
            sc.fit(df[[col]])
            self.scalers[col] = sc
            self.feature_dims.append(1)

        for col in self.cat_cols:
            le = LabelEncoder()
            le.fit(df[col].astype(str))
            self.encoders[col] = le
            self.feature_dims.append(len(le.classes_))

        return self

    def transform(self, df: pd.DataFrame) -> list[np.ndarray]:
        """
        Returns list of length d, each array (N, e_j).
        """
        out = []
        for col in self.num_cols:
            arr = self.scalers[col].transform(df[[col]]).astype(np.float32)
            out.append(arr)                         # (N, 1)
        for col in self.cat_cols:
            cats = self.encoders[col].transform(df[col].astype(str))
            # One-hot
            n_cls = len(self.encoders[col].classes_)
            oh = np.zeros((len(cats), n_cls), dtype=np.float32)
            oh[np.arange(len(cats)), cats] = 1.0
            out.append(oh)                          # (N, n_cls)
        return out


# ─────────────────────────────────────────────
#  PyTorch Dataset
# ─────────────────────────────────────────────

class TabularDataset(Dataset):
    """
    Stores per-feature arrays; yields list of per-feature tensors + label.
    """
    def __init__(
        self,
        encoded_features: list[np.ndarray],  # length d, each (N, e_j)
        labels: np.ndarray | None = None,
    ):
        self.features = [torch.tensor(f) for f in encoded_features]
        self.labels   = (torch.tensor(labels, dtype=torch.long)
                         if labels is not None else None)
        self.n = self.features[0].shape[0]

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        x = [f[idx] for f in self.features]
        y = self.labels[idx] if self.labels is not None else -1
        return x, y


def collate_fn(batch):
    """Custom collate for list-of-tensors feature format."""
    xs, ys = zip(*batch)
    # xs: tuple of (list_of_tensors_per_sample)
    d = len(xs[0])
    x_batch = [torch.stack([xs[b][j] for b in range(len(xs))]) for j in range(d)]
    y_batch  = torch.stack(ys)
    return x_batch, y_batch


def build_loaders(
    df: pd.DataFrame,
    target_col: str,
    num_cols: list[str],
    cat_cols: list[str],
    batch_size: int = 256,
    val_size: float = 0.10,
    test_size: float = 0.10,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader, TabularPreprocessor]:

    # Split
    df_trainval, df_test = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=df[target_col]
    )
    df_train, df_val = train_test_split(
        df_trainval,
        test_size=val_size / (1 - test_size),
        random_state=seed,
        stratify=df_trainval[target_col],
    )

    # Fit preprocessor on train only
    prep = TabularPreprocessor(num_cols, cat_cols)
    prep.fit(df_train)

    def make_loader(split_df, shuffle):
        enc = prep.transform(split_df)
        y   = split_df[target_col].values
        ds  = TabularDataset(enc, y)
        return DataLoader(
            ds, batch_size=batch_size, shuffle=shuffle,
            collate_fn=collate_fn, num_workers=0, pin_memory=False,
        )

    train_loader = make_loader(df_train, shuffle=True)
    val_loader   = make_loader(df_val,   shuffle=False)
    test_loader  = make_loader(df_test,  shuffle=False)

    return train_loader, val_loader, test_loader, prep
