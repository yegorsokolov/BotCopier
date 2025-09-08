import numpy as np

from scripts.train_target_clone import _load_logs, _extract_features, _maybe_smote


def test_smote_balances_classes(tmp_path):
    data = tmp_path / "trades_raw.csv"
    lines = ["label,spread,hour"]
    for i in range(20):
        lines.append(f"0,{1.0 + i*0.01},{i%24}")
    for i in range(2):
        lines.append(f"1,{2.0 + i*0.01},{i%24}")
    data.write_text("\n".join(lines))

    df, features, _ = _load_logs(data)
    df, features, _, _ = _extract_features(df, features)
    X = df[features].to_numpy(dtype=float)
    y = df["label"].astype(int).to_numpy()
    w = np.ones_like(y, dtype=float)

    X_res, y_res, w_res = _maybe_smote(X, y, w, threshold=1.5)
    orig_counts = np.bincount(y)
    res_counts = np.bincount(y_res)
    assert res_counts[0] == res_counts[1]
    assert res_counts[1] > orig_counts[1]
    assert len(w_res) == len(y_res)
