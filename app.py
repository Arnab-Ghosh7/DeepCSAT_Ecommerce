import os, sys, re, string, pickle, io, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template


import numpy.random as _npr

class MT19937(_npr.MT19937):
    def __setstate__(self, state):
        try:
            super().__setstate__(state)
        except Exception:
            pass

def _safe_bit_generator_ctor(bit_generator_name="MT19937"):
    if isinstance(bit_generator_name, _npr.BitGenerator):
        return bit_generator_name
    if isinstance(bit_generator_name, type) and issubclass(bit_generator_name, _npr.BitGenerator):
        return bit_generator_name()
    if bit_generator_name in ('MT19937', getattr(_npr, 'MT19937', None)):
        return MT19937()
    try:
        from numpy.random._pickle import __bit_generator_ctor
        return __bit_generator_ctor(bit_generator_name)
    except Exception:
        return _npr.MT19937()

def _safe_randomstate_ctor(bit_generator_name="MT19937"):
    res = _safe_bit_generator_ctor(bit_generator_name)
    return _npr.RandomState(res)

class _CompatUnpickler(pickle.Unpickler):
    """Custom unpickler that silently fixes numpy + sklearn version mismatches."""

    _NP_REMAP = {
        # numpy < 1.17 used a different path for BitGenerators
        "numpy.random.mtrand":          _npr,
        "numpy.random._mt19937":        _npr,
        "numpy.random._pcg64":          _npr,
        "numpy.random._philox":         _npr,
        "numpy.random._sfc64":          _npr,
        "numpy.random._bounded_integers": _npr,
        "numpy.random._common":         _npr,
    }

    def find_class(self, module, name):
        if module == "numpy.random._pickle":
            if name == "__bit_generator_ctor":
                return _safe_bit_generator_ctor
            if name == "__randomstate_ctor":
                return _safe_randomstate_ctor
                
        # Redirect old numpy.random sub-modules to current numpy.random
        if module in self._NP_REMAP:
            if name == 'MT19937':
                return MT19937
            obj = getattr(self._NP_REMAP[module], name, None)
            if obj is not None:
                return obj
                
        # Redirect sklearn internals that moved between versions
        # e.g. sklearn.utils.fixes  ->  sklearn.utils
        if module.startswith("sklearn"):
            try:
                return super().find_class(module, name)
            except (AttributeError, ModuleNotFoundError):
                # Try parent module as fallback
                parent = ".".join(module.split(".")[:-1])
                try:
                    return super().find_class(parent, name)
                except Exception:
                    pass
        return super().find_class(module, name)


def _safe_load(path: str):
    """Load a pickle file using the compatibility unpickler."""
    with open(path, "rb") as f:
        data = f.read()
    return _CompatUnpickler(io.BytesIO(data)).load()


try:
    import sklearn.utils.validation as _skv
    _orig_check = getattr(_skv, "_check_feature_names", None)

    from sklearn.utils import estimator_checks as _skec  
    import sklearn.base as _skb

    _orig_validate = getattr(_skb.BaseEstimator, "__sklearn_tags__", None)
except Exception:
    pass

try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except ImportError:
    pass


MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deepcsat_model.pkl")

print("=" * 55)
print("  DeepCSAT - Starting Up")
print("=" * 55)
print(f"  Python       : {sys.version.split()[0]}")
try:
    import sklearn
    print(f"  scikit-learn : {sklearn.__version__}")
except Exception:
    pass
print(f"  numpy        : {np.__version__}")
print(f"  Model path   : {MODEL_PATH}")
print("=" * 55)

if not os.path.exists(MODEL_PATH):
    print(f"\n[ERROR] deepcsat_model.pkl not found at:\n  {MODEL_PATH}")
    print("\nMake sure deepcsat_model.pkl is in the same folder as app.py\n")
    sys.exit(1)

import traceback

# Try loading with compat unpickler first, fall back to standard pickle
artifacts = None
try:
    artifacts = _safe_load(MODEL_PATH)
    print("\nModel loaded OK (compatibility mode)\n")
except Exception as e1:
    print(f"  Compat load failed: {e1}")
    try:
        with open(MODEL_PATH, "rb") as f:
            artifacts = pickle.load(f)
        print("\nModel loaded OK (standard mode)\n")
    except Exception as e2:
        print(f"\n[ERROR] Both load attempts failed.")
        print(f"  Error: {e2}")
        print("\nThe pkl was created with a very different numpy/sklearn version.")
        print("Please install matching versions:")
        print("  pip install scikit-learn==1.6.1")
        print("  (or re-run the notebook on this machine to regenerate the pkl)\n")
        sys.exit(1)

MODEL    = artifacts["model"]
SCALER   = artifacts["scaler"]
TFIDF    = artifacts["tfidf"]
LE_DICT  = artifacts["label_encoders"]
SEL_COLS = artifacts["selected_cols"]
S_FEATS  = artifacts["structured_feats"]
CAT_COLS = artifacts["cat_cols"]


STOPWORDS = set(["i","me","my","we","our","you","your","he","him","his",
                  "she","her","it","its","they","them","their","this","that",
                  "am","is","are","was","were","have","has","do","does","the",
                  "and","but","or","for","a","an","in","on","to","at","with"])
CONTRACTIONS = {
    "don't":"do not","won't":"will not","can't":"cannot","isn't":"is not",
    "it's":"it is","i'm":"i am","i've":"i have","you're":"you are",
    "they're":"they are","we're":"we are",
}

def preprocess_text(text):
    if not isinstance(text, str): text = ""
    for c, e in CONTRACTIONS.items():
        text = re.sub(re.escape(c), e, text, flags=re.IGNORECASE)
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b\w*\d\w*\b", "", text)
    text = " ".join([w for w in text.split() if w not in STOPWORDS])
    return text.strip()

def predict_csat(data):
    row = {}
    for col in CAT_COLS:
        le  = LE_DICT[col]
        val = str(data.get(col, ""))
        row[col + "_enc"] = int(le.transform([val])[0]) if val in le.classes_ else 0
    row["response_time_mins"] = np.log1p(float(data.get("response_time_mins", 0)))
    row["Item_price"]         = np.log1p(float(data.get("Item_price",         0)))
    row["issue_hour"]         = int(data.get("issue_hour", 10))
    row["issue_dow"]          = int(data.get("issue_dow",  1))
    row["had_remark"]         = int(bool(str(data.get("Customer Remarks", "")).strip()))
    s_df       = pd.DataFrame([row])[S_FEATS]
    clean      = preprocess_text(data.get("Customer Remarks", ""))
    tfidf_arr  = TFIDF.transform([clean]).toarray()
    tfidf_df   = pd.DataFrame(tfidf_arr,
                               columns=[f"tfidf_{w}" for w in TFIDF.get_feature_names_out()])
    combined   = pd.concat([s_df.reset_index(drop=True),
                             tfidf_df.reset_index(drop=True)], axis=1)
    for c in SEL_COLS:
        if c not in combined.columns: combined[c] = 0
    combined = combined[SEL_COLS]
    X     = SCALER.transform(combined)
    pred  = int(MODEL.predict(X)[0])
    proba = MODEL.predict_proba(X)[0].tolist()
    return pred, proba

# Flask app
app = Flask(__name__, template_folder="templates")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    try:
        score, proba = predict_csat(data)
        return jsonify({"csat_score": score, "probabilities": proba})
    except Exception as exc:
        app.logger.error(f"Prediction error: {exc}")
        return jsonify({"error": str(exc)}), 500

if __name__ == "__main__":
    print("=" * 55)
    print("  DeepCSAT Web App  ->  http://127.0.0.1:5000")
    print("=" * 55 + "\n")
    app.run(debug=True, host="0.0.0.0", port=5000)