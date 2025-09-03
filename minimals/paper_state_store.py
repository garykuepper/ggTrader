import io
import json
import os
from datetime import datetime
from typing import Dict, Any, Optional

class PaperStateStore:
    def __init__(self, json_path: Optional[str] = None):
        # e.g., set via env STATE_PATH=C:/.../state/paper_state.json on Windows
        self.json_path = json_path or os.getenv("STATE_PATH", "./state/paper_state.json")

    @staticmethod
    def _default_state() -> Dict[str, Any]:
        return {
            "cash": 10_000.0,
            "transaction_fee": 0.004,   # 0.4%
            "positions": {},            # sym -> {qty, entry_price, current_price, entry_ts_iso, trailing}
            "next_entry_time": {},      # sym -> iso timestamp
            "equity_curve": [],         # [{ts, equity}]
            "trades": []                # [{symbol, side, qty, price, ts, fees}]
        }

    def _ensure_dir(self):
        d = os.path.dirname(os.path.abspath(self.json_path))
        if d and not os.path.exists(d):
            os.makedirs(d, exist_ok=True)

    def _merge_defaults(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """If the file is missing keys (e.g., after an update), add them."""
        default = self._default_state()
        for k, v in default.items():
            if k not in state:
                state[k] = v
        # sanity for types
        state["positions"] = state.get("positions", {}) or {}
        state["next_entry_time"] = state.get("next_entry_time", {}) or {}
        state["equity_curve"] = state.get("equity_curve", []) or []
        state["trades"] = state.get("trades", []) or []
        # numeric basics
        try:
            state["cash"] = float(state.get("cash", default["cash"]))
        except Exception:
            state["cash"] = default["cash"]
        try:
            state["transaction_fee"] = float(state.get("transaction_fee", default["transaction_fee"]))
        except Exception:
            state["transaction_fee"] = default["transaction_fee"]
        return state

    def load(self) -> Dict[str, Any]:
        self._ensure_dir()
        default = self._default_state()
        if not os.path.exists(self.json_path):
            return default

        # If file exists but is empty or invalid, recover to default
        try:
            if os.path.getsize(self.json_path) == 0:
                return default
        except OSError:
            return default

        try:
            with open(self.json_path, "r", encoding="utf-8") as f:
                state = json.load(f)
            if not isinstance(state, dict):
                return default
            return self._merge_defaults(state)
        except (json.JSONDecodeError, OSError, io.UnsupportedOperation):
            # Corrupted/partial write—fall back safely
            return default

    def save(self, state: Dict[str, Any]) -> None:
        """Atomic write to avoid truncation; safe on Windows & Linux."""
        self._ensure_dir()
        tmp_path = self.json_path + ".tmp"
        # merge defaults before saving, so schema stays consistent
        state = self._merge_defaults(state)
        # add a tiny safeguard timestamp (handy when inspecting files)
        state["_last_saved"] = datetime.utcnow().isoformat() + "Z"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        # On Windows, replace works from Python 3.8+
        os.replace(tmp_path, self.json_path)
