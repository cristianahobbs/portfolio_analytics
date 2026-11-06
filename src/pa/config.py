"""Load the YAML config and expose convenient views of it."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "config" / "portfolio.yaml"


@dataclass
class Config:
    raw: dict
    path: Path | None = None
    _assets: pd.DataFrame = field(init=False, repr=False)

    def __post_init__(self):
        a = pd.DataFrame(self.raw["assets"]).T
        a.index.name = "ticker"
        for col in ("weight", "duration", "spread_duration", "spread_bps", "adv_usd"):
            a[col] = a[col].astype(float)
        if "max_weight" not in a:
            a["max_weight"] = float("nan")
        a["max_weight"] = a["max_weight"].astype(float).fillna(self.constraints["max_weight"])
        self._assets = a
        total = a["weight"].sum()
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Portfolio weights sum to {total:.4f}, expected 1.0")

    def __getitem__(self, key):
        return self.raw[key]

    def get(self, key, default=None):
        return self.raw.get(key, default)

    # --- views -----------------------------------------------------------
    @property
    def assets(self) -> pd.DataFrame:
        return self._assets

    @property
    def tickers(self) -> list[str]:
        return list(self._assets.index)

    @property
    def weights(self) -> pd.Series:
        return self._assets["weight"].rename("weight")

    @property
    def asset_class(self) -> pd.Series:
        return self._assets["asset_class"]

    @property
    def group(self) -> pd.Series:
        return self._assets["group"]

    @property
    def constraints(self) -> dict:
        return self.raw["constraints"]

    @property
    def benchmark_weights(self) -> pd.Series:
        return pd.Series(self.raw["benchmark"]["weights"], dtype=float)

    @property
    def cash_ticker(self) -> str:
        return self.raw["cash_ticker"]

    @property
    def trading_days(self) -> int:
        return int(self.raw.get("trading_days", 252))

    @property
    def proxies(self) -> dict[str, str]:
        """Holding -> proxy ticker used for returns before the holding listed."""
        if "proxy" not in self._assets:
            return {}
        p = self._assets["proxy"].dropna()
        return {t: v for t, v in p.items() if isinstance(v, str) and v}

    def factor_tickers(self) -> list[str]:
        out = []
        for v in self.raw["factors"].values():
            out.extend(v if isinstance(v, list) else [v])
        return out

    def all_tickers(self) -> list[str]:
        """Every price series the analytics need (deduplicated, order-preserving)."""
        groups = self.raw["benchmark"]["groups"].values()
        extra = (list(self.benchmark_weights.index) + [g["proxy"] for g in groups]
                 + list(self.proxies.values())
                 + self.factor_tickers() + [self.raw["vix_ticker"], self.raw["fx"]["fx_ticker"]])
        return list(dict.fromkeys(self.tickers + extra))

    def with_weights(self, weights: pd.Series) -> "Config":
        """Copy of the config with new portfolio weights (used by the dashboard)."""
        raw = yaml.safe_load(yaml.safe_dump(self.raw))
        for t, w in weights.items():
            raw["assets"][t]["weight"] = float(w)
        return Config(raw, self.path)


def load_config(path: str | Path | None = None) -> Config:
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, encoding="utf-8") as f:
        return Config(yaml.safe_load(f), path)
