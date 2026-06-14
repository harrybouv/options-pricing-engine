import { useEffect, useMemo, useState } from "react";
import PlotlyComponent from "react-plotly.js";
import {
  Activity,
  BarChart3,
  RefreshCcw,
  Search,
  ShieldAlert,
} from "lucide-react";
import "./App.css";

const Plot = PlotlyComponent.default ?? PlotlyComponent;
const API_BASE = "http://127.0.0.1:8000";

const PLOT_CONFIG = {
  displaylogo: false,
  responsive: true,
  scrollZoom: true,
  modeBarButtonsToRemove: ["lasso2d", "select2d", "autoScale2d", "toggleSpikelines"],
};

function valid(x) {
  return x !== null && x !== undefined && x !== "" && Number.isFinite(Number(x));
}

function n(x, fallback = 0) {
  return valid(x) ? Number(x) : fallback;
}

function fmtNum(x, d = 2) {
  if (!valid(x)) return "—";
  return Number(x).toLocaleString(undefined, {
    minimumFractionDigits: d,
    maximumFractionDigits: d,
  });
}

function fmtMoney(x, d = 2) {
  if (!valid(x)) return "—";
  return `$${fmtNum(x, d)}`;
}

function fmtPct(x, d = 2) {
  if (!valid(x)) return "—";
  return `${(Number(x) * 100).toFixed(d)}%`;
}

function fmtSigned(x, d = 2) {
  if (!valid(x)) return "—";
  const v = Number(x);
  return `${v >= 0 ? "+" : ""}${v.toFixed(d)}`;
}

function fmtSignedPct(x, d = 2) {
  if (!valid(x)) return "—";
  const v = Number(x) * 100;
  return `${v >= 0 ? "+" : ""}${v.toFixed(d)}%`;
}

function getMid(row) {
  if (!row) return null;
  if (valid(row.mid)) return Number(row.mid);
  if (valid(row.market_price)) return Number(row.market_price);
  return null;
}

function cleanLabel(label) {
  if (!label) return "—";
  const s = String(label).toLowerCase();
  return s.charAt(0).toUpperCase() + s.slice(1);
}

function tone(x, positiveGood = true) {
  if (!valid(x)) return "";
  const v = Number(x);
  if (Math.abs(v) < 1e-12) return "";
  if (positiveGood) return v > 0 ? "good" : "bad";
  return v > 0 ? "bad" : "good";
}

function valueFor(row, key) {
  if (!row) return null;
  if (key === "sigma") return row.sigma;
  if (key === "delta") return row.delta;
  if (key === "gamma") return row.gamma;
  if (key === "vega_1pct") return valid(row.vega_1pct) ? row.vega_1pct : valid(row.vega) ? row.vega / 100 : null;
  if (key === "theta_day") return valid(row.theta_day) ? row.theta_day : valid(row.theta) ? row.theta / 365 : null;
  return row[key];
}

function movingAverage(values, window) {
  return values.map((_, i) => {
    if (i + 1 < window) return null;
    const slice = values.slice(i + 1 - window, i + 1).filter(valid);
    if (slice.length < window) return null;
    return slice.reduce((a, b) => a + Number(b), 0) / window;
  });
}

function rollingRealisedVol(history, window) {
  const out = [];
  const closes = history.map((d) => Number(d.Close));

  for (let i = 0; i < closes.length; i++) {
    if (i < window) {
      out.push(null);
      continue;
    }

    const returns = [];
    for (let j = i - window + 1; j <= i; j++) {
      if (closes[j] > 0 && closes[j - 1] > 0) {
        returns.push(Math.log(closes[j] / closes[j - 1]));
      }
    }

    if (returns.length < window - 1) {
      out.push(null);
      continue;
    }

    const mean = returns.reduce((a, b) => a + b, 0) / returns.length;
    const variance =
      returns.reduce((a, b) => a + Math.pow(b - mean, 2), 0) /
      Math.max(returns.length - 1, 1);

    out.push(Math.sqrt(variance) * Math.sqrt(252));
  }

  return out;
}

function filterHistory(history, range) {
  if (!history?.length) return [];
  if (range === "ALL") return history;

  const days = {
    "1M": 31,
    "3M": 92,
    "6M": 183,
    "1Y": 366,
    "2Y": 732,
    "5Y": 1830,
  }[range];

  return days ? history.slice(-days) : history;
}

function erf(x) {
  const sign = x >= 0 ? 1 : -1;
  x = Math.abs(x);

  const a1 = 0.254829592;
  const a2 = -0.284496736;
  const a3 = 1.421413741;
  const a4 = -1.453152027;
  const a5 = 1.061405429;
  const p = 0.3275911;

  const t = 1 / (1 + p * x);
  const y =
    1 -
    (((((a5 * t + a4) * t + a3) * t + a2) * t + a1) *
      t *
      Math.exp(-x * x));

  return sign * y;
}

function normCdf(x) {
  return 0.5 * (1 + erf(x / Math.sqrt(2)));
}

function bsPrice(S, K, r, q, T, sigma, type) {
  S = n(S);
  K = n(K);
  r = n(r);
  q = n(q);
  T = n(T);
  sigma = n(sigma);

  if (S <= 0 || K <= 0 || T <= 0 || sigma <= 0) return null;

  const d1 =
    (Math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) /
    (sigma * Math.sqrt(T));
  const d2 = d1 - sigma * Math.sqrt(T);

  if (type === "call") {
    return S * Math.exp(-q * T) * normCdf(d1) - K * Math.exp(-r * T) * normCdf(d2);
  }

  return K * Math.exp(-r * T) * normCdf(-d2) - S * Math.exp(-q * T) * normCdf(-d1);
}

function payoffData(S0, K, premium, type) {
  const minS = Math.max(0.01, S0 * 0.65);
  const maxS = S0 * 1.45;
  const xs = Array.from({ length: 180 }, (_, i) => minS + i * ((maxS - minS) / 179));

  const ys = xs.map((S) => {
    const gross = type === "call" ? Math.max(S - K, 0) : Math.max(K - S, 0);
    return gross - premium;
  });

  return { xs, ys };
}

function valueData(S0, K, r, q, T, sigma, type) {
  const minS = Math.max(0.01, S0 * 0.7);
  const maxS = S0 * 1.3;
  const xs = Array.from({ length: 180 }, (_, i) => minS + i * ((maxS - minS) / 179));
  const ys = xs.map((S) => bsPrice(S, K, r, q, T, sigma, type));
  return { xs, ys };
}

function riskMatrix(S0, K, r, q, T, sigma, type, basePrice) {
  const spotShocks = [-0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15];
  const volShocks = [-0.1, -0.05, 0, 0.05, 0.1];

  const z = volShocks.map((dv) =>
    spotShocks.map((ds) => {
      const shockedS = Math.max(0.01, S0 * (1 + ds));
      const shockedSigma = Math.max(0.001, sigma + dv);
      const shockedPrice = bsPrice(shockedS, K, r, q, T, shockedSigma, type);
      return shockedPrice === null || basePrice === null ? null : shockedPrice - basePrice;
    })
  );

  return { spotShocks, volShocks, z };
}

function buildTermStructure(chain, selected) {
  if (!chain?.length || !selected) return [];

  const expiries = [...new Set(chain.map((x) => x.expiry))];

  return expiries
    .map((expiry) => {
      const group = chain.filter((x) => x.expiry === expiry && valid(x.sigma));
      if (!group.length) return null;

      const atm = [...group].sort(
        (a, b) => Math.abs(n(a.moneyness) - 1) - Math.abs(n(b.moneyness) - 1)
      )[0];

      const nearK = [...group].sort(
        (a, b) => Math.abs(n(a.strike) - n(selected.strike)) - Math.abs(n(b.strike) - n(selected.strike))
      )[0];

      return {
        expiry,
        T: n(atm.T),
        atmSigma: n(atm.sigma),
        selectedSigma: valid(nearK?.sigma) ? n(nearK.sigma) : null,
      };
    })
    .filter(Boolean)
    .sort((a, b) => a.T - b.T);
}

function buildSurfaceFromGrid(surfaceGrid) {
  const clean = (surfaceGrid ?? []).filter((d) => valid(d.T) && valid(d.moneyness) && valid(d.sigma));

  const x = [...new Set(clean.map((d) => Number(d.T)))].sort((a, b) => a - b);
  const y = [...new Set(clean.map((d) => Number(d.moneyness)))].sort((a, b) => a - b);

  const map = new Map(
    clean.map((d) => [
      `${Number(d.T).toFixed(8)}|${Number(d.moneyness).toFixed(8)}`,
      Number(d.sigma),
    ])
  );

  const z = y.map((m) =>
    x.map((t) => map.get(`${Number(t).toFixed(8)}|${Number(m).toFixed(8)}`) ?? null)
  );

  return { x, y, z };
}

function buildSurfaceFromChain(chain, key) {
  const clean = (chain ?? []).filter((d) => valid(d.T) && valid(d.moneyness) && valid(valueFor(d, key)));

  const x = [...new Set(clean.map((d) => Number(d.T)))].sort((a, b) => a - b);
  const y = [...new Set(clean.map((d) => Number(d.moneyness)))].sort((a, b) => a - b);

  const map = new Map(
    clean.map((d) => [
      `${Number(d.T).toFixed(8)}|${Number(d.moneyness).toFixed(8)}`,
      Number(valueFor(d, key)),
    ])
  );

  const z = y.map((m) =>
    x.map((t) => map.get(`${Number(t).toFixed(8)}|${Number(m).toFixed(8)}`) ?? null)
  );

  return { x, y, z };
}

function interpolate1d(xs, ys, x) {
  if (!xs.length || x < xs[0] || x > xs[xs.length - 1]) return null;

  for (let i = 0; i < xs.length - 1; i++) {
    const x0 = xs[i];
    const x1 = xs[i + 1];
    if (x >= x0 && x <= x1) {
      const y0 = ys[i];
      const y1 = ys[i + 1];
      if (!valid(y0) || !valid(y1) || x1 === x0) return null;
      const w = (x - x0) / (x1 - x0);
      return y0 + w * (y1 - y0);
    }
  }

  return null;
}

function buildSmoothedSurfaceFromChain(chain, key) {
  const clean = (chain ?? []).filter((d) => valid(d.T) && valid(d.moneyness) && valid(valueFor(d, key)));
  const expiries = [...new Set(clean.map((d) => d.expiry))]
    .map((expiry) => {
      const rows = clean.filter((d) => d.expiry === expiry);
      const T = rows.find((r) => valid(r.T))?.T;
      return valid(T) ? { expiry, T: Number(T), rows } : null;
    })
    .filter(Boolean)
    .sort((a, b) => a.T - b.T);

  const allM = clean.map((d) => Number(d.moneyness)).filter((m) => m >= 0.75 && m <= 1.35);
  if (!expiries.length || allM.length < 3) return buildSurfaceFromChain(chain, key);

  const lo = Math.max(0.8, Math.min(...allM));
  const hi = Math.min(1.25, Math.max(...allM));
  const points = 55;
  const y = Array.from({ length: points }, (_, i) => lo + (i * (hi - lo)) / (points - 1));
  const x = expiries.map((e) => e.T);

  const columns = expiries.map(({ rows }) => {
    const grouped = new Map();

    rows.forEach((row) => {
      const m = Number(row.moneyness);
      const v = Number(valueFor(row, key));
      if (!valid(m) || !valid(v)) return;
      const bucket = Number(m.toFixed(4));
      if (!grouped.has(bucket)) grouped.set(bucket, []);
      grouped.get(bucket).push(v);
    });

    const pairs = [...grouped.entries()]
      .map(([m, values]) => [Number(m), values.reduce((a, b) => a + b, 0) / values.length])
      .sort((a, b) => a[0] - b[0]);

    if (pairs.length < 3) return y.map(() => null);

    const xs = pairs.map((p) => p[0]);
    const ys = pairs.map((p) => p[1]);
    return y.map((m) => interpolate1d(xs, ys, m));
  });

  const z = y.map((_, rowIndex) => columns.map((col) => col[rowIndex]));
  return { x, y, z };
}

function metricLabel(key) {
  const labels = {
    sigma: "Implied volatility",
    delta: "Delta",
    gamma: "Gamma",
    vega_1pct: "Vega / +1 vol point",
    theta_day: "Theta / day",
  };
  return labels[key] ?? key;
}

function metricShort(key) {
  const labels = {
    sigma: "IV",
    delta: "Delta",
    gamma: "Gamma",
    vega_1pct: "Vega",
    theta_day: "Theta",
  };
  return labels[key] ?? key;
}

function metricTickFormat(key) {
  return key === "sigma" ? ".0%" : undefined;
}

function surfaceColorscale(key) {
  if (["delta", "theta_day"].includes(key)) {
    return [
      [0, "#8a2634"],
      [0.48, "#e8edf2"],
      [0.52, "#e8edf2"],
      [1, "#1c6d8f"],
    ];
  }

  return [
    [0, "#07111f"],
    [0.25, "#12365a"],
    [0.5, "#0f7490"],
    [0.74, "#f2cc60"],
    [1, "#fb7185"],
  ];
}

function flattenZ(z) {
  return (z ?? []).flat().filter((v) => valid(v)).map(Number);
}

function tagClass(label) {
  return String(label ?? "")
    .toLowerCase()
    .replaceAll("/", "-")
    .replaceAll("%", "pct")
    .replaceAll("+", "plus")
    .replaceAll(" ", "-");
}

function binomialAmericanPrice(S0, K, r, q, T, sigma, N, type) {
  S0 = n(S0); K = n(K); r = n(r); q = n(q); T = n(T); sigma = n(sigma); N = Math.max(5, Math.floor(n(N, 100)));
  if (S0 <= 0 || K <= 0 || T <= 0 || sigma <= 0) return null;

  const dt = T / N;
  const u = Math.exp(sigma * Math.sqrt(dt));
  const d = Math.exp(-sigma * Math.sqrt(dt));
  const p = (Math.exp((r - q) * dt) - d) / (u - d);
  if (p < 0 || p > 1 || !Number.isFinite(p)) return null;

  const values = [];
  for (let j = 0; j <= N; j++) {
    const S = S0 * Math.pow(u, j) * Math.pow(d, N - j);
    values[j] = type === "call" ? Math.max(S - K, 0) : Math.max(K - S, 0);
  }

  for (let i = N - 1; i >= 0; i--) {
    for (let j = 0; j <= i; j++) {
      const S = S0 * Math.pow(u, j) * Math.pow(d, i - j);
      const continuation = Math.exp(-r * dt) * (p * values[j + 1] + (1 - p) * values[j]);
      const exercise = type === "call" ? Math.max(S - K, 0) : Math.max(K - S, 0);
      values[j] = Math.max(continuation, exercise);
    }
  }

  return values[0];
}

function greekBadge(title, value, selected) {
  if (!valid(value)) return "Unavailable";

  const v = Number(value);
  const abs = Math.abs(v);

  if (title === "Delta") {
    if (abs >= 0.75) return "High directional";
    if (abs >= 0.45) return "Medium directional";
    return "Low directional";
  }

  if (title === "Gamma") {
    if (selected && Math.abs(n(selected.moneyness) - 1) < 0.04 && n(selected.T) < 0.12) {
      return "High convexity";
    }
    if (abs > 0.015) return "Medium convexity";
    return "Low convexity";
  }

  if (title === "Vega") {
    if (selected && n(selected.T) > 0.35) return "Long vol exposure";
    if (selected && n(selected.T) > 0.12) return "Medium vol exposure";
    return "Short-dated";
  }

  if (title === "Theta") {
    if (v < -0.2) return "High decay";
    if (v < -0.05) return "Medium decay";
    return "Low decay";
  }

  if (title === "IV/RV") {
    if (v > 0.04) return "IV rich";
    if (v < -0.02) return "IV cheap";
    return "In line";
  }

  return "Normal";
}

function greekRead(title, value, selected) {
  if (!valid(value)) return "No stable exposure returned.";

  if (title === "Delta") {
    const abs = Math.abs(Number(value));
    if (abs >= 0.75) return "Behaves close to underlying exposure.";
    if (abs >= 0.45) return "ATM-style directional exposure.";
    return "Low directional sensitivity.";
  }

  if (title === "Gamma") {
    if (selected && Math.abs(n(selected.moneyness) - 1) < 0.04 && n(selected.T) < 0.12) {
      return "Small spot moves materially change delta.";
    }
    return "Convexity is present but controlled.";
  }

  if (title === "Vega") {
    if (selected && n(selected.T) > 0.35) return "Volatility moves can dominate.";
    if (selected && n(selected.T) < 0.08) return "Short expiry; gamma/theta dominate.";
    return "Moderate volatility exposure.";
  }

  if (title === "Theta") {
    if (Number(value) < -0.2) return "Heavy premium decay.";
    if (Number(value) < -0.05) return "Meaningful daily decay.";
    return "Low daily decay.";
  }

  return "Exposure diagnostic.";
}

function modelExplanation(model) {
  if (model === "European") {
    return "Closed-form Black-Scholes benchmark. Fast, clean, and useful as the reference model.";
  }
  if (model === "American") {
    return "Binomial tree model with backward induction and early-exercise logic.";
  }
  if (model === "Asian") {
    return "Monte Carlo path-dependent pricing based on the arithmetic average price.";
  }
  if (model === "Bermudan") {
    return "Longstaff-Schwartz Monte Carlo: regression estimates continuation value at exercise dates.";
  }
  return "Pricing model.";
}

function simulatePaths({ S0, r, q, T, sigma, steps = 126, paths = 70, seed = 11 }) {
  let s = seed;

  function rand() {
    s = (s * 1664525 + 1013904223) % 4294967296;
    return s / 4294967296;
  }

  function normal() {
    const u1 = Math.max(rand(), 1e-12);
    const u2 = Math.max(rand(), 1e-12);
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  }

  S0 = Math.max(0.01, n(S0, 100));
  r = n(r, 0.05);
  q = n(q, 0);
  T = Math.max(0.001, n(T, 1));
  sigma = Math.max(0.001, n(sigma, 0.2));
  steps = Math.max(12, Math.min(252, Number(steps)));
  paths = Math.max(20, Math.min(100, Number(paths)));

  const dt = T / steps;
  const x = Array.from({ length: steps + 1 }, (_, i) => (i * T) / steps);
  const matrix = [];

  for (let p = 0; p < paths; p++) {
    const row = [S0];
    for (let i = 1; i <= steps; i++) {
      const z = normal();
      const prev = row[row.length - 1];
      row.push(prev * Math.exp((r - q - 0.5 * sigma * sigma) * dt + sigma * Math.sqrt(dt) * z));
    }
    matrix.push(row);
  }

  const terminal = matrix.map((row) => row[row.length - 1]);
  const mean = x.map((_, i) => matrix.reduce((a, row) => a + row[i], 0) / matrix.length);

  return { x, matrix, terminal, mean };
}

function seCurve(S0, K, r, q, T, sigma, type) {
  const sizes = [1000, 2000, 5000, 10000, 20000, 50000, 100000];
  const base = Math.max(1, bsPrice(S0, K, r, q, T, sigma, type) ?? 1);

  return {
    sizes,
    plain: sizes.map((m) => (base * 2.2) / Math.sqrt(m)),
    antithetic: sizes.map((m) => (base * 1.8) / Math.sqrt(m)),
    control: sizes.map((m) => (base * 1.05) / Math.sqrt(m)),
    combined: sizes.map((m) => (base * 0.82) / Math.sqrt(m)),
  };
}


function bsDecomposition(S, K, r, q, T, sigma, type) {
  S = n(S);
  K = n(K);
  r = n(r);
  q = n(q);
  T = n(T);
  sigma = n(sigma);

  if (S <= 0 || K <= 0 || T <= 0 || sigma <= 0) return null;

  const d1 =
    (Math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) /
    (sigma * Math.sqrt(T));
  const d2 = d1 - sigma * Math.sqrt(T);
  const discountedSpot = S * Math.exp(-q * T);
  const discountedStrike = K * Math.exp(-r * T);

  if (type === "call") {
    return {
      d1,
      d2,
      spotLeg: discountedSpot * normCdf(d1),
      strikeLeg: discountedStrike * normCdf(d2),
      price: discountedSpot * normCdf(d1) - discountedStrike * normCdf(d2),
      direction: "Call = discounted spot leg − discounted strike leg",
    };
  }

  return {
    d1,
    d2,
    spotLeg: discountedSpot * normCdf(-d1),
    strikeLeg: discountedStrike * normCdf(-d2),
    price: discountedStrike * normCdf(-d2) - discountedSpot * normCdf(-d1),
    direction: "Put = discounted strike leg − discounted spot leg",
  };
}

function binomialTreeVisual(S0, K, T, sigma, steps = 6, type = "call") {
  S0 = Math.max(0.01, n(S0, 100));
  K = Math.max(0.01, n(K, 100));
  T = Math.max(0.001, n(T, 1));
  sigma = Math.max(0.001, n(sigma, 0.2));
  steps = Math.max(3, Math.min(7, Math.floor(n(steps, 6))));

  const dt = T / steps;
  const u = Math.exp(sigma * Math.sqrt(dt));
  const d = Math.exp(-sigma * Math.sqrt(dt));

  const edgeX = [];
  const edgeY = [];
  const nodeX = [];
  const nodeY = [];
  const nodeText = [];
  const intrinsic = [];

  for (let i = 0; i <= steps; i++) {
    for (let j = 0; j <= i; j++) {
      const y = j - i / 2;
      const S = S0 * Math.pow(u, j) * Math.pow(d, i - j);
      nodeX.push(i);
      nodeY.push(y);
      nodeText.push(`Step ${i}<br>Node ${j}<br>S ${fmtMoney(S)}`);
      intrinsic.push(type === "call" ? Math.max(S - K, 0) : Math.max(K - S, 0));

      if (i < steps) {
        edgeX.push(i, i + 1, null, i, i + 1, null);
        edgeY.push(y, j + 1 - (i + 1) / 2, null, y, j - (i + 1) / 2, null);
      }
    }
  }

  return { edgeX, edgeY, nodeX, nodeY, nodeText, intrinsic, steps };
}

function binomialConvergenceData(S0, K, r, q, T, sigma, type) {
  const steps = [25, 50, 100, 200, 300, 500, 750];
  return {
    steps,
    values: steps.map((N) => binomialAmericanPrice(S0, K, r, q, T, sigma, N, type)),
  };
}

function modelAssumptionRows(model, custom, customPrice) {
  const rows = [
    ["Model", model],
    ["S0", fmtMoney(custom.S0)],
    ["K", fmtMoney(custom.K)],
    ["T", `${fmtNum(custom.T, 4)}y`],
    ["σ", fmtPct(custom.sigma)],
  ];

  if (model === "American") rows.push(["Tree steps", fmtNum(custom.binomial_steps, 0)]);
  if (model === "Asian" || model === "Bermudan") {
    rows.push(["MC paths", fmtNum(custom.mc_paths, 0)]);
    rows.push(["MC steps", fmtNum(custom.mc_steps, 0)]);
    rows.push(["Std. error", fmtNum(customPrice?.standard_error, 4)]);
  }
  if (model === "Bermudan") rows.push(["Exercise dates", fmtNum(custom.exercise_dates, 0)]);

  return rows;
}

function App() {
  const [ticker, setTicker] = useState("AAPL");
  const [side, setSide] = useState("calls");
  const [model, setModel] = useState("European");
  const [view, setView] = useState("Volatility");
  const [historyRange, setHistoryRange] = useState("6M");
  const [surfaceMetric, setSurfaceMetric] = useState("sigma");

  const [data, setData] = useState(null);
  const [selectedId, setSelectedId] = useState(null);
  const [price, setPrice] = useState(null);
  const [customPrice, setCustomPrice] = useState(null);

  const [loading, setLoading] = useState(false);
  const [pricingCustom, setPricingCustom] = useState(false);
  const [error, setError] = useState("");

  const [custom, setCustom] = useState({
    option_type: "call",
    model: "European",
    S0: 291.13,
    K: 290,
    T: 0.25,
    r: 0.05,
    q: 0.0037,
    sigma: 0.23,
    binomial_steps: 300,
    mc_paths: 25000,
    mc_steps: 252,
    exercise_dates: 12,
  });

  const r = 0.05;

  async function loadTerminal() {
    setLoading(true);
    setError("");

    try {
      const res = await fetch(`${API_BASE}/terminal/${ticker}?side=${side}&period=5y`);
      if (!res.ok) throw new Error(`Backend returned ${res.status}`);

      const json = await res.json();

      setData(json);
      setSelectedId(json.selected_contract?.contract_id ?? json.chain?.[0]?.contract_id ?? null);
    } catch (err) {
      setError(err.message || "Failed to fetch terminal data. Check uvicorn is running on 127.0.0.1:8000.");
    } finally {
      setLoading(false);
    }
  }

  async function priceSelected() {
    if (!selected || !data) return;

    const body = {
      model,
      S0: data.spot,
      K: selected.strike,
      r,
      q: data.q ?? 0,
      T: selected.T,
      sigma: selected.sigma,
      option_type: data.option_type,
      binomial_steps: 300,
      mc_paths: 25000,
      mc_steps: 252,
      exercise_dates: 12,
    };

    const res = await fetch(`${API_BASE}/price`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });

    if (!res.ok) throw new Error(await res.text());

    const json = await res.json();
    setPrice(json);
  }

  async function priceCustom() {
    setPricingCustom(true);
    setError("");

    try {
      const body = {
        model: custom.model,
        S0: Number(custom.S0),
        K: Number(custom.K),
        r: Number(custom.r),
        q: Number(custom.q),
        T: Number(custom.T),
        sigma: Number(custom.sigma),
        option_type: custom.option_type,
        binomial_steps: Number(custom.binomial_steps),
        mc_paths: Number(custom.mc_paths),
        mc_steps: Number(custom.mc_steps),
        exercise_dates: Number(custom.exercise_dates),
      };

      const res = await fetch(`${API_BASE}/price`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });

      if (!res.ok) throw new Error(await res.text());

      const json = await res.json();
      setCustomPrice(json);
    } catch (err) {
      setError(err.message || "Custom pricing failed.");
    } finally {
      setPricingCustom(false);
    }
  }

  useEffect(() => {
    loadTerminal();
  }, [side]);

  const chain = data?.chain ?? [];

  const selected = useMemo(() => {
    if (!chain.length) return null;
    return chain.find((x) => x.contract_id === selectedId) ?? chain[0];
  }, [chain, selectedId]);

  useEffect(() => {
    if (selected && data) {
      priceSelected().catch((e) => setError(e.message));
    }
  }, [selectedId, model, data]);

  useEffect(() => {
    if (selected && data) {
      setCustom((c) => ({
        ...c,
        option_type: data.option_type,
        model,
        S0: Number(data.spot).toFixed(2),
        K: Number(selected.strike).toFixed(2),
        T: Number(selected.T).toFixed(4),
        q: Number(data.q ?? 0).toFixed(4),
        sigma: Number(selected.sigma).toFixed(4),
      }));
    }
  }, [selectedId, model, data]);

  const expiries = [...new Set(chain.map((x) => x.expiry))];

  function selectExpiry(expiry) {
    if (!data || !expiry) return;

    const rows = chain.filter((x) => x.expiry === expiry);
    const nearest = [...rows].sort(
      (a, b) => Math.abs(n(a.strike) - n(data.spot)) - Math.abs(n(b.strike) - n(data.spot))
    )[0];

    if (nearest) setSelectedId(nearest.contract_id);
  }

  const selectedExpiryChain = chain
    .filter((x) => x.expiry === selected?.expiry)
    .sort((a, b) => n(a.strike) - n(b.strike));

  const marketMid = selected ? getMid(selected) : null;
  const modelPrice = valid(price?.price) ? Number(price.price) : null;
  const edge = modelPrice !== null && marketMid !== null ? modelPrice - marketMid : null;
  const edgePct = edge !== null && marketMid ? edge / marketMid : null;

  const intrinsic =
    selected && data
      ? data.option_type === "call"
        ? Math.max(n(data.spot) - n(selected.strike), 0)
        : Math.max(n(selected.strike) - n(data.spot), 0)
      : null;

  const extrinsic = marketMid !== null && intrinsic !== null ? marketMid - intrinsic : null;

  const breakeven =
    selected && marketMid !== null
      ? data.option_type === "call"
        ? n(selected.strike) + marketMid
        : n(selected.strike) - marketMid
      : null;

  const ivRv = selected ? n(selected.sigma) - n(data?.price_stats?.rv_20, NaN) : null;

  const rawHistory = data?.history ?? [];
  const history = filterHistory(rawHistory, historyRange);
  const dates = history.map((d) => d.Date);
  const closes = history.map((d) => Number(d.Close));
  const ma20 = movingAverage(closes, 20);
  const ma50 = movingAverage(closes, 50);
  const rv20 = rollingRealisedVol(history, 20);
  const rv50 = rollingRealisedVol(history, 50);

  const surface =
    surfaceMetric === "sigma"
      ? buildSurfaceFromGrid(data?.surface_grid ?? [])
      : buildSmoothedSurfaceFromChain(chain, surfaceMetric);

  const surfaceValues = flattenZ(surface.z);
  const surfaceAbsMax = Math.max(...surfaceValues.map((v) => Math.abs(v)), 0.0001);
  const useDivergingSurface = ["delta", "theta_day"].includes(surfaceMetric);

  const smile = selectedExpiryChain.filter((x) => valid(x.sigma));
  const term = buildTermStructure(chain, selected);

  const payoff =
    selected && data
      ? payoffData(n(data.spot), n(selected.strike), marketMid ?? modelPrice ?? 0, data.option_type)
      : null;

  const sensitivity =
    selected && data
      ? valueData(n(data.spot), n(selected.strike), r, n(data.q), n(selected.T), n(selected.sigma), data.option_type)
      : null;

  const risk =
    selected && data
      ? riskMatrix(n(data.spot), n(selected.strike), r, n(data.q), n(selected.T), n(selected.sigma), data.option_type, modelPrice)
      : null;

  const riskValues = flattenZ(risk?.z);
  const riskAbsMax = Math.max(...riskValues.map((v) => Math.abs(v)), 0.01);

  const customPayoff = payoffData(
    n(custom.S0),
    n(custom.K),
    valid(customPrice?.price) ? Number(customPrice.price) : bsPrice(n(custom.S0), n(custom.K), n(custom.r), n(custom.q), n(custom.T), n(custom.sigma), custom.option_type) ?? 0,
    custom.option_type
  );

  const customValue = valueData(
    n(custom.S0),
    n(custom.K),
    n(custom.r),
    n(custom.q),
    n(custom.T),
    n(custom.sigma),
    custom.option_type
  );

  const paths = simulatePaths({
    S0: n(custom.S0),
    r: n(custom.r),
    q: n(custom.q),
    T: n(custom.T),
    sigma: n(custom.sigma),
    steps: Math.min(252, n(custom.mc_steps, 126)),
    paths: 75,
  });

  const se = seCurve(n(custom.S0), n(custom.K), n(custom.r), n(custom.q), n(custom.T), n(custom.sigma), custom.option_type);

  const displayCustomPrice = valid(customPrice?.price)
    ? Number(customPrice.price)
    : bsPrice(n(custom.S0), n(custom.K), n(custom.r), n(custom.q), n(custom.T), n(custom.sigma), custom.option_type);

  const baseLayout = (height = 360) => ({
    height,
    autosize: true,
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(2,6,12,0.88)",
    font: { color: "#dbe7f5", family: "Inter, Arial" },
    margin: { l: 56, r: 22, t: 18, b: 48 },
    xaxis: {
      gridcolor: "rgba(148,163,184,0.10)",
      zerolinecolor: "rgba(148,163,184,0.20)",
      color: "#9fb0c8",
      linecolor: "rgba(148,163,184,0.18)",
      tickfont: { size: 11 },
    },
    yaxis: {
      gridcolor: "rgba(148,163,184,0.10)",
      zerolinecolor: "rgba(148,163,184,0.20)",
      color: "#9fb0c8",
      linecolor: "rgba(148,163,184,0.18)",
      tickfont: { size: 11 },
    },
    legend: {
      orientation: "h",
      y: 1.1,
      x: 0,
      font: { size: 11, color: "#aebbd0" },
      bgcolor: "rgba(0,0,0,0)",
    },
    hoverlabel: {
      bgcolor: "#0b111d",
      bordercolor: "rgba(148,163,184,0.22)",
      font: { color: "#e8eef8" },
    },
  });

  const scene = {
    xaxis: {
      title: { text: "Time to expiry (years)", font: { color: "#c7d4e8", size: 12 } },
      backgroundcolor: "rgba(3,7,13,0.95)",
      gridcolor: "rgba(148,163,184,0.16)",
      zerolinecolor: "rgba(148,163,184,0.18)",
      color: "#9fb0c8",
    },
    yaxis: {
      title: { text: "Moneyness (K / S)", font: { color: "#c7d4e8", size: 12 } },
      backgroundcolor: "rgba(3,7,13,0.95)",
      gridcolor: "rgba(148,163,184,0.16)",
      zerolinecolor: "rgba(148,163,184,0.18)",
      color: "#9fb0c8",
    },
    zaxis: {
      title: { text: metricLabel(surfaceMetric), font: { color: "#c7d4e8", size: 12 } },
      tickformat: metricTickFormat(surfaceMetric),
      backgroundcolor: "rgba(3,7,13,0.95)",
      gridcolor: "rgba(148,163,184,0.16)",
      zerolinecolor: "rgba(148,163,184,0.18)",
      color: "#9fb0c8",
    },
    camera: { eye: { x: 1.55, y: 1.65, z: 0.95 } },
  };

  return (
    <div className="app">
      <header className="topbar">
        <div>
          <div className="eyebrow">Options analytics</div>
          <h1>Options Pricing & Risk Workstation</h1>
          <p>Option chains, volatility surfaces, Greeks, pricing models and scenario shocks.</p>
        </div>

        <div className="status-strip">
          <span>Backend · 127.0.0.1:8000</span>
          <span>{loading ? "Loading market data" : "Session live"}</span>
        </div>
      </header>

      <section className="command">
        <div className="field ticker-field">
          <label>Ticker</label>
          <div className="input-row">
            <Search size={16} />
            <input
              value={ticker}
              onChange={(e) => setTicker(e.target.value.toUpperCase())}
              onKeyDown={(e) => {
                if (e.key === "Enter") loadTerminal();
              }}
            />
          </div>
        </div>

        <div className="field">
          <label>Surface</label>
          <select value={side} onChange={(e) => setSide(e.target.value)}>
            <option value="calls">Calls</option>
            <option value="puts">Puts</option>
          </select>
        </div>

        <div className="field">
          <label>Selected model</label>
          <select value={model} onChange={(e) => setModel(e.target.value)}>
            <option>European</option>
            <option>American</option>
            <option>Asian</option>
            <option>Bermudan</option>
          </select>
        </div>

        <button onClick={loadTerminal} className="refresh" disabled={loading}>
          <RefreshCcw size={16} />
          {loading ? "Loading" : "Refresh"}
        </button>
      </section>

      {error && (
        <div className="error">
          <ShieldAlert size={16} />
          <span>{error}</span>
        </div>
      )}

      {data && selected && (
        <>
          <section className="contract">
            <div>
              <div className="contract-label">Selected contract</div>
              <div className="contract-title">
                {data.ticker} {fmtNum(selected.strike, 0)}
                {data.option_type === "call" ? "C" : "P"} · {selected.expiry}
              </div>
              <div className="contract-sub">
                {selected.contractSymbol || selected.contract_id} · K/S {fmtNum(selected.moneyness, 3)} · T {fmtNum(selected.T, 4)}y
              </div>
            </div>

            <div className="contract-right">
              <div className="mini-field">
                <label>Expiry</label>
                <select value={selected.expiry} onChange={(e) => selectExpiry(e.target.value)}>
                  {expiries.map((expiry) => (
                    <option key={expiry} value={expiry}>
                      {expiry}
                    </option>
                  ))}
                </select>
              </div>

              <div className="contract-pills">
                <span>Spot {fmtMoney(data.spot)}</span>
                <span>IV {fmtPct(selected.sigma)}</span>
                <span>Mid {fmtMoney(marketMid)}</span>
                <span>Spread {fmtPct(selected.relative_spread)}</span>
                <span className={tone(edge, true)}>
                  Model-mid diff {fmtSigned(edge)} / {fmtPct(edgePct)}
                </span>
              </div>
            </div>
          </section>

          <section className="kpis">
            <Kpi title="Spot" value={fmtMoney(data.spot)} note={`${fmtSigned(data.price_stats?.change)} today`} />
            <Kpi title="20D realised vol" value={fmtPct(data.price_stats?.rv_20)} note="Annualised close-to-close" />
            <Kpi title="Selected IV" value={fmtPct(selected.sigma)} note={`IV/RV ${fmtSignedPct(ivRv)}`} />
            <Kpi title="Theoretical price" value={fmtMoney(modelPrice)} note={price?.method ?? "Pricing model"} />
            <Kpi title="Market mid" value={fmtMoney(marketMid)} note={`Bid ${fmtMoney(selected.bid)} / Ask ${fmtMoney(selected.ask)}`} />
            <Kpi title="Quote quality" value={cleanLabel(data.surface_quality?.label)} note={`${data.surface_quality?.contracts ?? 0} contracts`} />
          </section>

          <main className="workspace">
            <aside className="left-panel">
              <Panel title="Market">
                <Metric label="Daily move" value={`${fmtSigned(data.price_stats?.change)} / ${fmtPct(data.price_stats?.change_pct)}`} />
                <Metric label="20D range" value={`${fmtMoney(data.price_stats?.low_20)} – ${fmtMoney(data.price_stats?.high_20)}`} />
                <Metric label="50D RV" value={fmtPct(data.price_stats?.rv_50)} />
              </Panel>

              <Panel title="Contract economics">
                <Metric label="Strike" value={fmtMoney(selected.strike)} />
                <Metric label="Intrinsic" value={fmtMoney(intrinsic)} />
                <Metric label="Extrinsic" value={fmtMoney(extrinsic)} />
                <Metric label="Breakeven" value={fmtMoney(breakeven)} />
              </Panel>

              <Panel title="Liquidity">
                <Metric label="Spread" value={fmtPct(selected.relative_spread)} accent={n(selected.relative_spread) <= 0.05 ? "good" : "bad"} />
                <Metric label="Volume" value={fmtNum(selected.volume, 0)} />
                <Metric label="Open interest" value={fmtNum(selected.openInterest, 0)} />
                <Metric label="Expiries" value={fmtNum(data.surface_quality?.expiries, 0)} />
              </Panel>
            </aside>

            <section className="main-panel">
              <div className="view-tabs">
                {["Market", "Volatility", "Pricer", "Risk"].map((x) => (
                  <button key={x} className={view === x ? "active" : ""} onClick={() => setView(x)}>
                    {x}
                  </button>
                ))}
              </div>

              {view === "Market" && (
                <div className="market-grid">
                  <ChartCard title={`${data.ticker} price history`} subtitle="Candles, 20D MA and 50D MA.">
                    <div className="range-buttons">
                      {["1M", "3M", "6M", "1Y", "2Y", "5Y", "ALL"].map((x) => (
                        <button key={x} className={historyRange === x ? "active" : ""} onClick={() => setHistoryRange(x)}>
                          {x}
                        </button>
                      ))}
                    </div>

                    <Plot
                      data={[
                        {
                          type: "candlestick",
                          x: dates,
                          open: history.map((d) => d.Open),
                          high: history.map((d) => d.High),
                          low: history.map((d) => d.Low),
                          close: history.map((d) => d.Close),
                          name: "OHLC",
                          increasing: { line: { color: "#32d583" }, fillcolor: "#32d583" },
                          decreasing: { line: { color: "#fb7185" }, fillcolor: "#fb7185" },
                        },
                        {
                          type: "scatter",
                          mode: "lines",
                          x: dates,
                          y: ma20,
                          name: "20D MA",
                          line: { color: "#7dd3fc", width: 1.8 },
                        },
                        {
                          type: "scatter",
                          mode: "lines",
                          x: dates,
                          y: ma50,
                          name: "50D MA",
                          line: { color: "#c4b5fd", width: 1.8 },
                        },
                      ]}
                      layout={{
                        ...baseLayout(520),
                        xaxis: { ...baseLayout().xaxis, rangeslider: { visible: false } },
                        yaxis: { ...baseLayout().yaxis, title: "Share price ($)", tickprefix: "$" },
                      }}
                      config={PLOT_CONFIG}
                      style={{ width: "100%" }}
                    />
                  </ChartCard>

                  <div className="stacked-charts">
                    <ChartCard title="Realised volatility" subtitle="20D and 50D annualised close-to-close volatility.">
                      <Plot
                        data={[
                          {
                            type: "scatter",
                            mode: "lines",
                            x: dates,
                            y: rv20,
                            name: "20D RV",
                            line: { color: "#7dd3fc", width: 2 },
                          },
                          {
                            type: "scatter",
                            mode: "lines",
                            x: dates,
                            y: rv50,
                            name: "50D RV",
                            line: { color: "#c4b5fd", width: 2 },
                          },
                        ]}
                        layout={{
                          ...baseLayout(250),
                          yaxis: { ...baseLayout().yaxis, title: "Annualised volatility", tickformat: ".0%" },
                        }}
                        config={PLOT_CONFIG}
                        style={{ width: "100%" }}
                      />
                    </ChartCard>

                    <ChartCard title="Volume" subtitle="Daily shares traded.">
                      <Plot
                        data={[
                          {
                            type: "bar",
                            x: dates,
                            y: history.map((d) => d.Volume),
                            name: "Volume",
                            marker: { color: "rgba(125, 211, 252, 0.42)" },
                          },
                        ]}
                        layout={{
                          ...baseLayout(250),
                          yaxis: { ...baseLayout().yaxis, title: "Shares" },
                        }}
                        config={PLOT_CONFIG}
                        style={{ width: "100%" }}
                      />
                    </ChartCard>
                  </div>
                </div>
              )}

              {view === "Volatility" && (
                <div className="vol-grid">
                  <ChartCard className="surface-card" title="3D surface" subtitle="IV or Greek exposure by expiry and moneyness.">
                    <div className="surface-switch">
                      {[
                        ["sigma", "IV"],
                        ["delta", "Delta"],
                        ["gamma", "Gamma"],
                        ["vega_1pct", "Vega"],
                        ["theta_day", "Theta"],
                      ].map(([key, label]) => (
                        <button key={key} className={surfaceMetric === key ? "active" : ""} onClick={() => setSurfaceMetric(key)}>
                          {label}
                        </button>
                      ))}
                    </div>

                    <Plot
                      data={[
                        {
                          type: "surface",
                          x: surface.x,
                          y: surface.y,
                          z: surface.z,
                          name: surfaceMetric,
                          opacity: 0.97,
                          contours: {
                            z: { show: true, usecolormap: true, project: { z: true } },
                          },
                          colorscale: surfaceColorscale(surfaceMetric),
                          cmin: useDivergingSurface ? -surfaceAbsMax : undefined,
                          cmax: useDivergingSurface ? surfaceAbsMax : undefined,
                          cmid: useDivergingSurface ? 0 : undefined,
                          colorbar: {
                            title: metricShort(surfaceMetric),
                            thickness: 12,
                            len: 0.72,
                          },
                          hovertemplate: "T %{x:.3f}y<br>Moneyness %{y:.3f}<br>" + metricShort(surfaceMetric) + " %{z:.4f}<extra></extra>",
                        },
                        {
                          type: "scatter3d",
                          mode: "markers",
                          x: [n(selected.T)],
                          y: [n(selected.moneyness)],
                          z: [surfaceMetric === "sigma" ? n(selected.sigma) : n(valueFor(selected, surfaceMetric))],
                          name: "Selected",
                          marker: {
                            size: 6,
                            color: "#ffffff",
                            symbol: "diamond",
                            line: { color: "#7dd3fc", width: 2 },
                          },
                        },
                      ]}
                      layout={{
                        ...baseLayout(580),
                        margin: { l: 0, r: 0, t: 4, b: 0 },
                        scene,
                      }}
                      config={PLOT_CONFIG}
                      style={{ width: "100%" }}
                    />
                  </ChartCard>

                  <div className="vol-side">
                    <Panel title="Surface">
                      <Metric label="Metric" value={metricLabel(surfaceMetric)} />
                      <Metric label="Selected value" value={surfaceMetric === "sigma" ? fmtPct(selected.sigma) : fmtNum(valueFor(selected, surfaceMetric), 4)} />
                      <Metric label="Quality" value={cleanLabel(data.surface_quality?.label)} />
                      <Metric label="Contracts" value={fmtNum(data.surface_quality?.contracts, 0)} />
                      <p className="panel-note">
                        IV uses the interpolated grid. Greeks are smoothed across expiry and moneyness for readability.
                      </p>
                    </Panel>

                    <Panel title="Greek tags">
                      <Signal title="Delta" value={fmtNum(price?.greeks?.delta, 4)} badge={greekBadge("Delta", price?.greeks?.delta, selected)} text={greekRead("Delta", price?.greeks?.delta, selected)} />
                      <Signal title="Gamma" value={fmtNum(price?.greeks?.gamma, 5)} badge={greekBadge("Gamma", price?.greeks?.gamma, selected)} text={greekRead("Gamma", price?.greeks?.gamma, selected)} />
                      <Signal title="Vega" value={fmtNum(price?.greeks?.vega_1pct, 4)} badge={greekBadge("Vega", price?.greeks?.vega_1pct, selected)} text={greekRead("Vega", price?.greeks?.vega_1pct, selected)} />
                      <Signal title="Theta" value={fmtNum(price?.greeks?.theta_day, 4)} badge={greekBadge("Theta", price?.greeks?.theta_day, selected)} text={greekRead("Theta", price?.greeks?.theta_day, selected)} />
                    </Panel>
                  </div>

                  <ChartCard className="heatmap-card" title={`${metricShort(surfaceMetric)} heatmap`} subtitle="Same surface as a 2D map for cleaner Greek reading.">
                    <Plot
                      data={[
                        {
                          type: "heatmap",
                          x: surface.x,
                          y: surface.y,
                          z: surface.z,
                          colorscale: surfaceColorscale(surfaceMetric),
                          zmin: useDivergingSurface ? -surfaceAbsMax : undefined,
                          zmax: useDivergingSurface ? surfaceAbsMax : undefined,
                          zmid: useDivergingSurface ? 0 : undefined,
                          colorbar: { title: metricShort(surfaceMetric), thickness: 12 },
                          hovertemplate: "T %{x:.3f}y<br>Moneyness %{y:.3f}<br>" + metricShort(surfaceMetric) + " %{z:.4f}<extra></extra>",
                        },
                        {
                          type: "scatter",
                          mode: "markers",
                          x: [n(selected.T)],
                          y: [n(selected.moneyness)],
                          name: "Selected",
                          marker: { color: "#f8fafc", size: 11, symbol: "x", line: { color: "#7dd3fc", width: 2 } },
                        },
                      ]}
                      layout={{
                        ...baseLayout(330),
                        xaxis: { ...baseLayout().xaxis, title: "Time to expiry (years)" },
                        yaxis: { ...baseLayout().yaxis, title: "Moneyness (K / S)" },
                      }}
                      config={PLOT_CONFIG}
                      style={{ width: "100%" }}
                    />
                  </ChartCard>

                  <ChartCard title={`Smile · ${selected.expiry}`} subtitle="Same expiry. Selected contract marked.">
                    <Plot
                      data={[
                        {
                          type: "scatter",
                          mode: "lines+markers",
                          x: smile.map((x) => x.strike),
                          y: smile.map((x) => x.sigma),
                          name: selected.expiry,
                          line: { color: "#7dd3fc", width: 2.2 },
                          marker: { size: 5 },
                        },
                        {
                          type: "scatter",
                          mode: "markers",
                          x: [selected.strike],
                          y: [selected.sigma],
                          name: "Selected",
                          marker: { color: "#f8fafc", size: 12, symbol: "x" },
                        },
                      ]}
                      layout={{
                        ...baseLayout(315),
                        xaxis: { ...baseLayout().xaxis, title: "Strike ($)", tickprefix: "$" },
                        yaxis: { ...baseLayout().yaxis, title: "Implied volatility", tickformat: ".0%" },
                      }}
                      config={PLOT_CONFIG}
                      style={{ width: "100%" }}
                    />
                  </ChartCard>

                  <ChartCard title="Term structure" subtitle="ATM IV and nearest selected strike by expiry.">
                    <Plot
                      data={[
                        {
                          type: "scatter",
                          mode: "lines+markers",
                          x: term.map((x) => x.T),
                          y: term.map((x) => x.atmSigma),
                          name: "ATM IV",
                          line: { color: "#7dd3fc", width: 2.2 },
                        },
                        {
                          type: "scatter",
                          mode: "lines+markers",
                          x: term.map((x) => x.T),
                          y: term.map((x) => x.selectedSigma),
                          name: `Nearest K=${fmtNum(selected.strike, 0)}`,
                          line: { color: "#f2cc60", width: 2.2 },
                        },
                      ]}
                      layout={{
                        ...baseLayout(315),
                        xaxis: { ...baseLayout().xaxis, title: "Time to expiry (years)" },
                        yaxis: { ...baseLayout().yaxis, title: "Implied volatility", tickformat: ".0%" },
                      }}
                      config={PLOT_CONFIG}
                      style={{ width: "100%" }}
                    />
                  </ChartCard>
                </div>
              )}

              {view === "Pricer" && (
                <div className="pricer-grid">
                  <Panel title="Custom pricer" className="pricer-control-panel">
                    <div className="input-grid">
                      <Input label="S0" value={custom.S0} onChange={(v) => setCustom({ ...custom, S0: v })} />
                      <Input label="K" value={custom.K} onChange={(v) => setCustom({ ...custom, K: v })} />
                      <Input label="T years" value={custom.T} onChange={(v) => setCustom({ ...custom, T: v })} />
                      <Input label="r" value={custom.r} onChange={(v) => setCustom({ ...custom, r: v })} />
                      <Input label="q" value={custom.q} onChange={(v) => setCustom({ ...custom, q: v })} />
                      <Input label="sigma" value={custom.sigma} onChange={(v) => setCustom({ ...custom, sigma: v })} />

                      <div className="field compact">
                        <label>Type</label>
                        <select value={custom.option_type} onChange={(e) => setCustom({ ...custom, option_type: e.target.value })}>
                          <option value="call">Call</option>
                          <option value="put">Put</option>
                        </select>
                      </div>

                      <div className="field compact">
                        <label>Model</label>
                        <select value={custom.model} onChange={(e) => setCustom({ ...custom, model: e.target.value })}>
                          <option>European</option>
                          <option>American</option>
                          <option>Asian</option>
                          <option>Bermudan</option>
                        </select>
                      </div>

                      {custom.model === "American" && (
                        <Input label="Tree steps" value={custom.binomial_steps} onChange={(v) => setCustom({ ...custom, binomial_steps: v })} />
                      )}

                      {(custom.model === "Asian" || custom.model === "Bermudan") && (
                        <>
                          <Input label="MC paths" value={custom.mc_paths} onChange={(v) => setCustom({ ...custom, mc_paths: v })} />
                          <Input label="MC steps" value={custom.mc_steps} onChange={(v) => setCustom({ ...custom, mc_steps: v })} />
                        </>
                      )}

                      {custom.model === "Bermudan" && (
                        <Input label="Exercise dates" value={custom.exercise_dates} onChange={(v) => setCustom({ ...custom, exercise_dates: v })} />
                      )}
                    </div>

                    <button className="price-button" onClick={priceCustom} disabled={pricingCustom}>
                      {pricingCustom ? "Pricing..." : "Price custom option"}
                    </button>

                    <p className="panel-note">
                      Inputs are pulled from the selected listed contract by default, then can be stressed manually without changing the market selection.
                    </p>
                  </Panel>

                  <Panel title="Price output" className="price-output-panel">
                    <div className="giant-price">{fmtMoney(customPrice?.price)}</div>
                    <Metric label="Method" value={customPrice?.method ?? custom.model} />
                    <Metric label="European base" value={fmtMoney(customPrice?.european_price)} />
                    <Metric label="Model premium" value={fmtSigned(customPrice?.model_premium)} />
                    <Metric label="Standard error" value={fmtNum(customPrice?.standard_error, 4)} />
                    <Metric label="95% CI low" value={fmtMoney(customPrice?.ci_low)} />
                    <Metric label="95% CI high" value={fmtMoney(customPrice?.ci_high)} />
                    <p className="panel-note">{modelExplanation(custom.model)}</p>
                  </Panel>

                  <ModelMechanism
                    custom={custom}
                    customPrice={customPrice}
                    paths={paths}
                    se={se}
                    baseLayout={baseLayout}
                  />

                  <ChartCard className="wide-card" title="Mark-to-market value vs spot" subtitle="Reprice today across spot, holding IV and time fixed.">
                    <Plot
                      data={[
                        {
                          type: "scatter",
                          mode: "lines",
                          x: customValue.xs,
                          y: customValue.ys,
                          name: "Model value",
                          line: { color: "#c4b5fd", width: 2.7 },
                        },
                      ]}
                      layout={{
                        ...baseLayout(370),
                        shapes: [
                          { type: "line", x0: n(custom.S0), x1: n(custom.S0), y0: 0, y1: 1, yref: "paper", line: { color: "#dbe7f5", width: 1, dash: "dash" } },
                          { type: "line", x0: n(custom.K), x1: n(custom.K), y0: 0, y1: 1, yref: "paper", line: { color: "#f2cc60", width: 1, dash: "dot" } },
                        ],
                        xaxis: { ...baseLayout().xaxis, title: "Spot price today ($)", tickprefix: "$" },
                        yaxis: { ...baseLayout().yaxis, title: "Model option value ($)", tickprefix: "$" },
                      }}
                      config={PLOT_CONFIG}
                      style={{ width: "100%" }}
                    />
                  </ChartCard>

                  <ChartCard title="Expiry payoff / long PnL" subtitle="Net PnL at expiry after paying the model premium.">
                    <Plot
                      data={[
                        {
                          type: "scatter",
                          mode: "lines",
                          x: customPayoff.xs,
                          y: customPayoff.ys,
                          name: "Expiry PnL",
                          line: { color: "#7dd3fc", width: 2.7 },
                        },
                      ]}
                      layout={{
                        ...baseLayout(370),
                        shapes: [
                          { type: "line", x0: n(custom.S0), x1: n(custom.S0), y0: 0, y1: 1, yref: "paper", line: { color: "#dbe7f5", width: 1, dash: "dash" } },
                          { type: "line", x0: n(custom.K), x1: n(custom.K), y0: 0, y1: 1, yref: "paper", line: { color: "#f2cc60", width: 1, dash: "dot" } },
                          { type: "line", x0: 0, x1: 1, xref: "paper", y0: 0, y1: 0, line: { color: "rgba(219,231,245,0.35)", width: 1 } },
                        ],
                        xaxis: { ...baseLayout().xaxis, title: "Underlying price at expiry ($)", tickprefix: "$" },
                        yaxis: { ...baseLayout().yaxis, title: "Net PnL after premium ($)", tickprefix: "$" },
                      }}
                      config={PLOT_CONFIG}
                      style={{ width: "100%" }}
                    />
                  </ChartCard>
                </div>
              )}

              {view === "Models" && (
                <div className="models-grid">
                  <ChartCard title={`${custom.model} path view`} subtitle="GBM paths from the custom inputs. Bermudan marks exercise dates.">
                    <Plot
                      data={[
                        ...paths.matrix.slice(0, 50).map((row) => ({
                          type: "scatter",
                          mode: "lines",
                          x: paths.x,
                          y: row,
                          showlegend: false,
                          line: { color: "rgba(125,211,252,0.18)", width: 1 },
                        })),
                        {
                          type: "scatter",
                          mode: "lines",
                          x: paths.x,
                          y: paths.mean,
                          name: "Mean path",
                          line: { color: "#f2cc60", width: 3 },
                        },
                        ...(custom.model === "Bermudan"
                          ? Array.from({ length: Math.max(1, n(custom.exercise_dates, 12)) }, (_, i) => {
                              const t = ((i + 1) * n(custom.T)) / Math.max(1, n(custom.exercise_dates, 12));
                              return {
                                type: "scatter",
                                mode: "lines",
                                x: [t, t],
                                y: [Math.min(...paths.terminal) * 0.88, Math.max(...paths.terminal) * 1.08],
                                name: i === 0 ? "Exercise dates" : "",
                                showlegend: i === 0,
                                line: { color: "rgba(242,204,96,0.28)", width: 1, dash: "dot" },
                              };
                            })
                          : []),
                      ]}
                      layout={{
                        ...baseLayout(500),
                        xaxis: { ...baseLayout().xaxis, title: "Time (years)" },
                        yaxis: { ...baseLayout().yaxis, title: "Simulated underlying price ($)", tickprefix: "$" },
                      }}
                      config={PLOT_CONFIG}
                      style={{ width: "100%" }}
                    />
                  </ChartCard>

                  <Panel title="Model notes">
                    <ModelLine title="European" text="Closed-form Black-Scholes benchmark." active={custom.model === "European"} />
                    <ModelLine title="American" text="Binomial backward induction with early exercise." active={custom.model === "American"} />
                    <ModelLine title="Asian" text="Monte Carlo average-price payoff." active={custom.model === "Asian"} />
                    <ModelLine title="Bermudan" text="Longstaff-Schwartz regression over exercise dates." active={custom.model === "Bermudan"} />
                  </Panel>

                  <ChartCard title="Terminal distribution" subtitle="Simulated terminal underlying prices.">
                    <Plot
                      data={[
                        {
                          type: "histogram",
                          x: paths.terminal,
                          nbinsx: 34,
                          marker: { color: "rgba(125,211,252,0.56)" },
                          name: "Terminal price",
                        },
                      ]}
                      layout={{
                        ...baseLayout(330),
                        xaxis: { ...baseLayout().xaxis, title: "Terminal underlying price ($)", tickprefix: "$" },
                        yaxis: { ...baseLayout().yaxis, title: "Frequency" },
                      }}
                      config={PLOT_CONFIG}
                      style={{ width: "100%" }}
                    />
                  </ChartCard>

                  <ChartCard title="Monte Carlo error vs simulations" subtitle="Plain, antithetic, control variate and combined estimators.">
                    <Plot
                      data={[
                        { type: "scatter", mode: "lines+markers", x: se.sizes, y: se.plain, name: "Plain MC", line: { color: "#7dd3fc", width: 2 } },
                        { type: "scatter", mode: "lines+markers", x: se.sizes, y: se.antithetic, name: "Antithetic", line: { color: "#f2cc60", width: 2 } },
                        { type: "scatter", mode: "lines+markers", x: se.sizes, y: se.control, name: "Control variate", line: { color: "#32d583", width: 2 } },
                        { type: "scatter", mode: "lines+markers", x: se.sizes, y: se.combined, name: "Antithetic + control", line: { color: "#fb7185", width: 2 } },
                      ]}
                      layout={{
                        ...baseLayout(330),
                        xaxis: { ...baseLayout().xaxis, title: "Number of simulations", type: "log" },
                        yaxis: { ...baseLayout().yaxis, title: "Standard error" },
                      }}
                      config={PLOT_CONFIG}
                      style={{ width: "100%" }}
                    />
                  </ChartCard>
                </div>
              )}

              {view === "Risk" && risk && (
                <div className="risk-grid">
                  <ChartCard title="Scenario matrix" subtitle="Selected contract value change from spot and IV shocks.">
                    <Plot
                      data={[
                        {
                          type: "heatmap",
                          x: risk.spotShocks,
                          y: risk.volShocks,
                          z: risk.z,
                          zmin: -riskAbsMax,
                          zmax: riskAbsMax,
                          zmid: 0,
                          colorscale: [
                            [0, "#8a2634"],
                            [0.45, "#f0d8c8"],
                            [0.5, "#f3f7fb"],
                            [0.55, "#c7e9f7"],
                            [1, "#1c5d7d"],
                          ],
                          colorbar: { title: "PnL" },
                          hovertemplate: "Spot %{x:+.0%}<br>Vol %{y:+.0%}<br>PnL %{z:.2f}<extra></extra>",
                        },
                      ]}
                      layout={{
                        ...baseLayout(540),
                        xaxis: { ...baseLayout().xaxis, title: "Spot shock", tickformat: "+.0%" },
                        yaxis: { ...baseLayout().yaxis, title: "Volatility shock", tickformat: "+.0%" },
                      }}
                      config={PLOT_CONFIG}
                      style={{ width: "100%" }}
                    />
                  </ChartCard>

                  <div className="greek-grid">
                    <Greek title="Delta" value={fmtNum(price?.greeks?.delta, 4)} badge={greekBadge("Delta", price?.greeks?.delta, selected)} note={greekRead("Delta", price?.greeks?.delta, selected)} />
                    <Greek title="Gamma" value={fmtNum(price?.greeks?.gamma, 5)} badge={greekBadge("Gamma", price?.greeks?.gamma, selected)} note={greekRead("Gamma", price?.greeks?.gamma, selected)} />
                    <Greek title="Vega" value={fmtNum(price?.greeks?.vega_1pct, 4)} badge={greekBadge("Vega", price?.greeks?.vega_1pct, selected)} note="Value change per +1 vol point." />
                    <Greek title="Theta" value={fmtNum(price?.greeks?.theta_day, 4)} badge={greekBadge("Theta", price?.greeks?.theta_day, selected)} note={greekRead("Theta", price?.greeks?.theta_day, selected)} />
                    <Greek title="IV/RV" value={fmtSignedPct(ivRv)} badge={greekBadge("IV/RV", ivRv, selected)} note="Selected IV versus recent realised volatility." />
                  </div>
                </div>
              )}
            </section>

            <aside className="right-panel">
              <Panel title="Price read">
                <div className="big-price">{fmtMoney(modelPrice)}</div>
                <Metric label="Market mid" value={fmtMoney(marketMid)} />
                <Metric label="Model-mid diff" value={`${fmtSigned(edge)} / ${fmtPct(edgePct)}`} accent={tone(edge, true)} />
                <Metric label="European base" value={fmtMoney(price?.european_price)} />
                <Metric label="Model premium" value={fmtSigned(price?.model_premium)} />
                <p className="panel-note">Theoretical-mid difference is not a tradable edge before spreads, stale quotes, dividends, borrow and model risk.</p>
              </Panel>

              <Panel title="Position Greeks">
                <GreekMini title="Delta" value={fmtNum(price?.greeks?.delta, 4)} badge={greekBadge("Delta", price?.greeks?.delta, selected)} />
                <GreekMini title="Gamma" value={fmtNum(price?.greeks?.gamma, 5)} badge={greekBadge("Gamma", price?.greeks?.gamma, selected)} />
                <GreekMini title="Vega" value={fmtNum(price?.greeks?.vega_1pct, 4)} badge={greekBadge("Vega", price?.greeks?.vega_1pct, selected)} />
                <GreekMini title="Theta" value={fmtNum(price?.greeks?.theta_day, 4)} badge={greekBadge("Theta", price?.greeks?.theta_day, selected)} />
              </Panel>

              <Panel title="Inputs">
                <Metric label="S0" value={fmtMoney(data.spot)} />
                <Metric label="K" value={fmtMoney(selected.strike)} />
                <Metric label="T" value={`${fmtNum(selected.T, 4)}y`} />
                <Metric label="r" value={fmtPct(r)} />
                <Metric label="q" value={fmtPct(data.q ?? 0)} />
                <Metric label="σ" value={fmtPct(selected.sigma)} />
              </Panel>
            </aside>
          </main>

          <section className="chain-section">
            <div className="section-header">
              <div>
                <h2>Option chain</h2>
                <p>Selected expiry: {selected.expiry}. Click a row to reprice the workstation.</p>
              </div>
              <BarChart3 size={20} />
            </div>

            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Contract</th>
                    <th>K</th>
                    <th>Bid</th>
                    <th>Ask</th>
                    <th>Mid</th>
                    <th>Spread</th>
                    <th>IV</th>
                    <th>Delta</th>
                    <th>Gamma</th>
                    <th>Vega/1%</th>
                    <th>Theta/day</th>
                    <th>Volume</th>
                    <th>OI</th>
                  </tr>
                </thead>
                <tbody>
                  {selectedExpiryChain.map((row) => (
                    <tr
                      key={row.contract_id}
                      onClick={() => setSelectedId(row.contract_id)}
                      className={row.contract_id === selected.contract_id ? "selected" : ""}
                    >
                      <td>{row.contractSymbol || row.contract_id}</td>
                      <td>{fmtNum(row.strike, 2)}</td>
                      <td>{fmtMoney(row.bid)}</td>
                      <td>{fmtMoney(row.ask)}</td>
                      <td>{fmtMoney(getMid(row))}</td>
                      <td>{fmtPct(row.relative_spread)}</td>
                      <td>{fmtPct(row.sigma)}</td>
                      <td>{fmtNum(row.delta, 4)}</td>
                      <td>{fmtNum(row.gamma, 5)}</td>
                      <td>{fmtNum(valueFor(row, "vega_1pct"), 4)}</td>
                      <td>{fmtNum(valueFor(row, "theta_day"), 4)}</td>
                      <td>{fmtNum(row.volume, 0)}</td>
                      <td>{fmtNum(row.openInterest, 0)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        </>
      )}

      {!data && !loading && (
        <div className="empty">
          <Activity size={26} />
          <p>Press Refresh to load market data.</p>
        </div>
      )}
    </div>
  );
}

function ModelMechanism({ custom, customPrice, paths, se, baseLayout }) {
  const model = custom.model;
  const S0 = n(custom.S0);
  const K = n(custom.K);
  const r = n(custom.r);
  const q = n(custom.q);
  const T = n(custom.T);
  const sigma = n(custom.sigma);
  const type = custom.option_type;
  const assumptions = modelAssumptionRows(model, custom, customPrice);

  const plotConfig = {
    ...PLOT_CONFIG,
    responsive: true,
  };

  if (model === "European") {
    const bs = bsDecomposition(S0, K, r, q, T, sigma, type);

    return (
      <ChartCard
        className="model-lab-card wide-card"
        title="Model mechanism · European Black-Scholes"
        subtitle="Closed-form benchmark with the pricing legs made explicit."
      >
        <div className="model-lab-split formula-layout">
          <div className="model-equation-panel">
            <div className="model-frame-bar">
              <span>Closed-form pricing identity</span>
              <em>{type === "call" ? "Call option" : "Put option"}</em>
            </div>

            <div className="model-equation clean-equation">
              {type === "call" ? (
                <>
                  <span>C</span>
                  <em>=</em>
                  <span>S·e<sup>−qT</sup>·N(d₁)</span>
                  <em>−</em>
                  <span>K·e<sup>−rT</sup>·N(d₂)</span>
                </>
              ) : (
                <>
                  <span>P</span>
                  <em>=</em>
                  <span>K·e<sup>−rT</sup>·N(−d₂)</span>
                  <em>−</em>
                  <span>S·e<sup>−qT</sup>·N(−d₁)</span>
                </>
              )}
            </div>

            <div className="formula-grid">
              <div className="formula-term">
                <span>d₁</span>
                <strong>{fmtNum(bs?.d1, 4)}</strong>
              </div>
              <div className="formula-term">
                <span>d₂</span>
                <strong>{fmtNum(bs?.d2, 4)}</strong>
              </div>
              <div className="formula-term">
                <span>Spot leg</span>
                <strong>{fmtMoney(bs?.spotLeg)}</strong>
              </div>
              <div className="formula-term">
                <span>Strike leg</span>
                <strong>{fmtMoney(bs?.strikeLeg)}</strong>
              </div>
              <div className="formula-term">
                <span>Local check</span>
                <strong>{fmtMoney(bs?.price)}</strong>
              </div>
              <div className="formula-term">
                <span>Backend price</span>
                <strong>{fmtMoney(customPrice?.price)}</strong>
              </div>
            </div>

            <p className="model-disclosure">
              {bs?.direction}. This is a theoretical value under lognormal diffusion,
              constant volatility, constant rates and continuous dividend yield.
            </p>
          </div>

          <div className="model-copy-panel">
            <ModelLine
              title="Use case"
              text="Reference valuation and sanity check. Best used as the base case before testing early exercise, path dependence or scenario risk."
              active
            />
            <ModelLine
              title="Failure mode"
              text="Does not explain smile, skew, discrete dividends, jump risk, stale quotes, liquidity or execution cost."
            />
            {assumptions.map(([label, value]) => (
              <Metric key={label} label={label} value={value} />
            ))}
          </div>
        </div>
      </ChartCard>
    );
  }

  if (model === "American") {
    const tree = binomialTreeVisual(S0, K, T, sigma, 6, type);
    const convergence = binomialConvergenceData(S0, K, r, q, T, sigma, type);

    return (
      <ChartCard
        className="model-lab-card wide-card"
        title="Model mechanism · American binomial tree"
        subtitle="Backward induction with early-exercise comparison at every node."
      >
        <div className="model-lab-split">
          <div className="model-visual-frame">
            <div className="model-frame-bar">
              <span>Recombining price tree</span>
              <em>node colour = intrinsic value</em>
            </div>

            <Plot
              useResizeHandler
              className="model-plot"
              data={[
                {
                  type: "scatter",
                  mode: "lines",
                  x: tree.edgeX,
                  y: tree.edgeY,
                  hoverinfo: "skip",
                  showlegend: false,
                  line: {
                    color: "rgba(125,211,252,0.24)",
                    width: 1.25,
                  },
                },
                {
                  type: "scatter",
                  mode: "markers",
                  x: tree.nodeX,
                  y: tree.nodeY,
                  text: tree.nodeText,
                  hovertemplate: "%{text}<extra></extra>",
                  showlegend: false,
                  marker: {
                    size: 12,
                    color: tree.intrinsic,
                    colorscale: [
                      [0, "#12365a"],
                      [0.55, "#0f7490"],
                      [1, "#f2cc60"],
                    ],
                    line: {
                      color: "rgba(244,248,255,0.58)",
                      width: 1,
                    },
                  },
                },
              ]}
              layout={{
                ...baseLayout(390),
                autosize: true,
                showlegend: false,
                margin: { l: 54, r: 22, t: 12, b: 56 },
                xaxis: {
                  ...baseLayout().xaxis,
                  title: { text: "Tree step", font: { color: "#c7d4e8", size: 12 } },
                  range: [-0.35, tree.steps + 0.35],
                  dtick: 1,
                  showgrid: false,
                  zeroline: false,
                },
                yaxis: {
                  ...baseLayout().yaxis,
                  title: { text: "Up / down branch level", font: { color: "#c7d4e8", size: 12 } },
                  range: [-(tree.steps / 2) - 0.65, tree.steps / 2 + 0.65],
                  showticklabels: false,
                  showgrid: false,
                  zeroline: false,
                },
              }}
              config={plotConfig}
              style={{ width: "100%", height: "390px" }}
            />
          </div>

          <div className="model-copy-panel">
            <div className="model-visual-frame compact-model-frame">
              <div className="model-frame-bar">
                <span>Price convergence</span>
                <em>steps → stable value</em>
              </div>

              <Plot
                useResizeHandler
                className="model-plot"
                data={[
                  {
                    type: "scatter",
                    mode: "lines+markers",
                    x: convergence.steps,
                    y: convergence.values,
                    name: "American price",
                    hovertemplate: "Steps %{x}<br>Price $%{y:.4f}<extra></extra>",
                    line: { color: "#7dd3fc", width: 2.4 },
                    marker: { size: 6, color: "#7dd3fc" },
                  },
                ]}
                layout={{
                  ...baseLayout(205),
                  autosize: true,
                  showlegend: false,
                  margin: { l: 58, r: 14, t: 8, b: 48 },
                  xaxis: {
                    ...baseLayout().xaxis,
                    title: { text: "Tree steps", font: { color: "#c7d4e8", size: 11 } },
                  },
                  yaxis: {
                    ...baseLayout().yaxis,
                    title: { text: "Option value", font: { color: "#c7d4e8", size: 11 } },
                    tickprefix: "$",
                  },
                }}
                config={plotConfig}
                style={{ width: "100%", height: "205px" }}
              />
            </div>

            <ModelLine
              title="Backward induction"
              text="At each node the model takes the greater of discounted continuation value and immediate exercise value."
              active
            />
            <ModelLine
              title="Desk interpretation"
              text="The premium above the European benchmark is early-exercise value, not a standalone tradable edge."
            />
            {assumptions.map(([label, value]) => (
              <Metric key={label} label={label} value={value} />
            ))}
          </div>
        </div>
      </ChartCard>
    );
  }

  const isBermudan = model === "Bermudan";
  const exerciseCount = Math.max(1, n(custom.exercise_dates, 12));
  const terminalMin = paths.terminal?.length ? Math.min(...paths.terminal) : S0 * 0.8;
  const terminalMax = paths.terminal?.length ? Math.max(...paths.terminal) : S0 * 1.2;
  const yMin = terminalMin * 0.88;
  const yMax = terminalMax * 1.08;
  const averagePrices = paths.matrix.map((row) => row.reduce((a, b) => a + b, 0) / row.length);

  return (
    <ChartCard
      className="model-lab-card wide-card"
      title={`Model mechanism · ${model} Monte Carlo`}
      subtitle={
        isBermudan
          ? "Longstaff-Schwartz exercise windows over simulated paths."
          : "Arithmetic-average payoff estimated across simulated paths."
      }
    >
      <div className="model-lab-split">
        <div className="model-visual-frame">
          <div className="model-frame-bar">
            <span>{isBermudan ? "Simulated paths with exercise dates" : "Simulated paths"}</span>
            <em>{isBermudan ? "vertical markers = exercise windows" : "yellow line = mean path"}</em>
          </div>

          <Plot
            useResizeHandler
            className="model-plot"
            data={[
              ...paths.matrix.slice(0, 54).map((row) => ({
                type: "scatter",
                mode: "lines",
                x: paths.x,
                y: row,
                showlegend: false,
                hoverinfo: "skip",
                line: { color: "rgba(125,211,252,0.15)", width: 1 },
              })),
              {
                type: "scatter",
                mode: "lines",
                x: paths.x,
                y: paths.mean,
                name: "Mean path",
                hovertemplate: "t %{x:.3f}y<br>Mean $%{y:.2f}<extra></extra>",
                line: { color: "#f2cc60", width: 3 },
              },
              ...(isBermudan
                ? Array.from({ length: exerciseCount }, (_, i) => {
                    const t = ((i + 1) * T) / exerciseCount;
                    return {
                      type: "scatter",
                      mode: "lines",
                      x: [t, t],
                      y: [yMin, yMax],
                      name: i === 0 ? "Exercise dates" : "",
                      showlegend: i === 0,
                      hoverinfo: "skip",
                      line: {
                        color: "rgba(242,204,96,0.38)",
                        width: 1,
                        dash: "dot",
                      },
                    };
                  })
                : []),
            ]}
            layout={{
              ...baseLayout(390),
              autosize: true,
              margin: { l: 62, r: 22, t: 14, b: 56 },
              xaxis: {
                ...baseLayout().xaxis,
                title: { text: "Time to expiry (years)", font: { color: "#c7d4e8", size: 12 } },
              },
              yaxis: {
                ...baseLayout().yaxis,
                title: { text: "Simulated underlying price", font: { color: "#c7d4e8", size: 12 } },
                tickprefix: "$",
              },
              legend: {
                ...baseLayout().legend,
                y: 1.08,
              },
            }}
            config={plotConfig}
            style={{ width: "100%", height: "390px" }}
          />
        </div>

        <div className="model-copy-panel">
          <div className="model-visual-frame compact-model-frame">
            <div className="model-frame-bar">
              <span>{isBermudan ? "Terminal distribution" : "Average-price distribution"}</span>
              <em>{isBermudan ? "expiry values" : "path averages"}</em>
            </div>

            <Plot
              useResizeHandler
              className="model-plot"
              data={[
                {
                  type: "histogram",
                  x: isBermudan ? paths.terminal : averagePrices,
                  nbinsx: 30,
                  marker: { color: "rgba(125,211,252,0.52)" },
                  name: isBermudan ? "Terminal" : "Average price",
                  hovertemplate: "$%{x:.2f}<br>Count %{y}<extra></extra>",
                },
              ]}
              layout={{
                ...baseLayout(205),
                autosize: true,
                showlegend: false,
                margin: { l: 56, r: 14, t: 8, b: 48 },
                xaxis: {
                  ...baseLayout().xaxis,
                  title: {
                    text: isBermudan ? "Terminal price" : "Path average",
                    font: { color: "#c7d4e8", size: 11 },
                  },
                  tickprefix: "$",
                },
                yaxis: {
                  ...baseLayout().yaxis,
                  title: { text: "Frequency", font: { color: "#c7d4e8", size: 11 } },
                },
              }}
              config={plotConfig}
              style={{ width: "100%", height: "205px" }}
            />
          </div>

          <ModelLine
            title={isBermudan ? "Longstaff-Schwartz" : "Average-price payoff"}
            text={
              isBermudan
                ? "Regression estimates continuation value at allowed exercise dates, then exercises only when immediate value dominates."
                : "Each simulated path is reduced to its arithmetic average before applying the option payoff."
            }
            active
          />
          <ModelLine
            title="Error control"
            text="Monte Carlo output must be read with standard error and confidence interval, not as an exact quote."
          />
          {assumptions.map(([label, value]) => (
            <Metric key={label} label={label} value={value} />
          ))}
        </div>
      </div>
    </ChartCard>
  );
}


function Kpi({ title, value, note }) {
  return (
    <div className="kpi">
      <div className="kpi-title">{title}</div>
      <div className="kpi-value">{value}</div>
      <div className="kpi-note">{note}</div>
    </div>
  );
}

function Panel({ title, children, className = "" }) {
  return (
    <div className={`panel ${className}`}>
      <div className="panel-title">{title}</div>
      {children}
    </div>
  );
}

function Metric({ label, value, accent }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong className={accent || ""}>{value}</strong>
    </div>
  );
}

function ChartCard({ title, subtitle, children, className = "" }) {
  return (
    <div className={`chart-card ${className}`}>
      <div className="chart-head">
        <div>
          <h3>{title}</h3>
          {subtitle && <p>{subtitle}</p>}
        </div>
      </div>
      {children}
    </div>
  );
}

function Greek({ title, value, badge, note }) {
  return (
    <div className="greek-card">
      <div className="greek-top">
        <div className="greek-title">{title}</div>
        <div className={`greek-badge ${tagClass(badge)}`}>
          {badge}
        </div>
      </div>
      <div className="greek-value">{value}</div>
      <div className="greek-note">{note}</div>
    </div>
  );
}

function Signal({ title, value, badge, text }) {
  return (
    <div className="signal">
      <div className="signal-top">
        <span>{title}</span>
        <em className={`greek-badge ${tagClass(badge)}`}>{badge}</em>
      </div>
      <strong>{value}</strong>
      <p>{text}</p>
    </div>
  );
}

function GreekMini({ title, value, badge }) {
  return (
    <div className="greek-mini">
      <div>
        <span>{title}</span>
        <strong>{value}</strong>
      </div>
      <em className={`greek-badge ${tagClass(badge)}`}>{badge}</em>
    </div>
  );
}

function Input({ label, value, onChange }) {
  return (
    <div className="field compact">
      <label>{label}</label>
      <input className="small-input" value={value} onChange={(e) => onChange(e.target.value)} />
    </div>
  );
}

function ModelLine({ title, text, active }) {
  return (
    <div className={`model-line ${active ? "active" : ""}`}>
      <strong>{title}</strong>
      <p>{text}</p>
    </div>
  );
}

export default App;