import warnings

import joblib
import numpy as np

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    m = joblib.load("/home/flynn/ggTrader/models/ensemble_gate.joblib")  # read-only load
    import lightgbm
    import sklearn

    print("lightgbm", lightgbm.__version__, "sklearn", sklearn.__version__, type(m).__name__)
    x = np.random.default_rng(0).normal(size=(5, getattr(m, "n_features_in_", 10)))
    p = m.predict_proba(x)[:, 1] if hasattr(m, "predict_proba") else m.predict(x)
    print("pred", [round(float(v), 12) for v in p])
    print("warnings:", sorted({f"{x.category.__name__}: {str(x.message)[:90]}" for x in w}))
