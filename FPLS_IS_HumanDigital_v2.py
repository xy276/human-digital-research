# -*- coding: utf-8 -*-
"""
================================================================================
 FPLS 2024 -> 2025 : When Does Digital Information Substitute for Human Expertise?
 Human -> Digital transitions in consumer financial advice   (IS direction)
================================================================================

 DESIGN
   Prospective two-wave panel.  Every mechanism variable is measured in 2024;
   the outcome is the 2024 -> 2025 transition.  Predictors precede outcomes,
   which removes the most obvious reverse-causality story.

 HUMAN STATE (two definitions, both reported)
   REL : Q23    "Do you work with a financial professional?"   -> the relationship
   USE : Q20_11 advisor used as a decision source, past 12 months -> consultation
   REL is primary.  USE-based "exits" can simply mean "did not consult this year".

 DIGITAL STATE : Q20_9  online / digital resources used, past 12 months
 AI            : Q20_12 exists in 2025 only -> extension, never a predictor of
                 2024->2025 transitions

 OUTCOMES among consumers with human advice in 2024 (H24 = 1)
   outcome4 = 0 Retain       H25=1, D25=0
              1 Complement   H25=1, D25=1
              2 Substitute   H25=0, D25=1
              3 Drop out     H25=0, D25=0
   EXIT  = 1[H25 = 0]
   SUBST = 1[H25 = 0]  within H24=1 & D25=1  -> substitute vs complement   (HEADLINE)
   ENTRY = 1[H25 = 1]  among H24 = 0          -> digital as gateway?

 MECHANISMS (all 2024)
   capability     FIN_LIT (Q56-60; 77 = DK counted incorrect), CONF (S1)
   complexity     COMPLEX_PLAN (Q21), S4_ORD investable assets
   trust          TRUST (Q50), RELATIONAL (Q48,Q50,Q51,Q52)   [advised only]
   cost           FEE_IMPORT (Q26) importance of fee / pricing structure
   format         FORMAT_IMPORT (Q29) importance of in-person vs virtual format
   digital orient DIGITAL (Q20_9), SOCIAL (Q20_10)
   relationship   ENGAGE_SELF (Q34), TENURE (Q33)                [advised only]

 ESTIMATION
   Linear probability models with HC1 robust SEs (main) and binary logits with
   sandwich SEs (robustness), implemented in numpy.  No statsmodels / patsy.

 OUTPUT  ->  OUTDIR  (CSV per table + one Excel workbook)
================================================================================
"""

import os
import re
import math
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    from scipy import stats as _st
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False


# ==============================================================================
# CONFIG
# ==============================================================================

PATH24 = r"E:\下载\FPLS\FPLS24.csv"
PATH25 = r"E:\下载\FPLS\FPLS25.csv"
OUTDIR = r"E:\下载\FPLS\fpls_is"

PRIMARY_HUMAN = "REL"          # "REL" = Q23 relationship ; "USE" = Q20_11 consultation
RESTRICT_WAVECOMP = True       # keep 2025 respondents flagged as Wave-1 completers
Q21_DK_AS_ZERO = False         # Q21 "77 Don't know" (5.5%): False = missing (paper), True = no complex plan (robustness)
MIN_LEVEL_N = 15               # categorical levels rarer than this are pooled
WARN_CELL = 100                # warn when a key outcome cell is below this

# 2024 predictors available for everyone
CORE = ["FIN_LIT_24", "CONF_24", "RISK_TOL_24", "ANXIETY_24",
        "S4_ORD_24", "COMPLEX_PLAN_24",
        "FEE_IMPORT_24", "FORMAT_IMPORT_24",
        "DIGITAL_24", "SOCIAL_24"]

# 2024 predictors asked only of advised respondents
RELATIONSHIP = ["TRUST_24", "ENGAGE_SELF_24", "TENURE_24"]

# 2024 demographic controls (categorical)
CATS = ["SEX_24", "AGE_24", "EDUC5_24", "INCOME_24"]

PRETTY = {
    "FIN_LIT_24": "Financial literacy (0-5)",
    "CONF_24": "Financial confidence (1-4)",
    "RISK_TOL_24": "Risk tolerance (0-10)",
    "ANXIETY_24": "Financial anxiety (1-4)",
    "S4_ORD_24": "Investable assets (1-10)",
    "COMPLEX_PLAN_24": "Complex financial plan",
    "FEE_IMPORT_24": "Importance of fee structure (1-4)",
    "FORMAT_IMPORT_24": "Importance of service format (1-4)",
    "DIGITAL_24": "Digital resources 2024",
    "SOCIAL_24": "Podcasts / social media 2024",
    "TRUST_24": "Trust in professional (1-3)",
    "TRUST_C_24": "Trust in professional (centred)",
    "ENGAGE_SELF_24": "Self-directed engagement (0-1)",
    "TENURE_24": "Relationship tenure (years)",
    "DIGxTRUST": "Digital 2024 x Trust (centred)",
    "DIGxLIT": "Digital 2024 x Financial literacy",
    "AI_25": "AI assistant 2025 (contemporaneous)",
    "COMPLEMENT": "Complement (retain + digital)",
    "SUBSTITUTE": "Substitute (exit + digital)",
    "DROPOUT": "Drop out (exit, no digital)",
    "CONF_24_LAG": "Financial confidence 2024",
    "ANXIETY_24_LAG": "Financial anxiety 2024",
}


# ==============================================================================
# 1.  I/O AND NORMALIZATION
# ==============================================================================

_DASHES = {"\u2013": "-", "\u2014": "-", "\u2212": "-", "\u2010": "-", "\u2011": "-"}
_QUOTES = {"\u2019": "'", "\u2018": "'", "\u201c": '"', "\u201d": '"', "\u00b4": "'"}
_NULLS = {"", "nan", "na", "n/a", ".", "none", "null", "-", "--"}

MISSING_TOKENS = {
    77, 98, 99, -1, -8, -9,
    "don't know", "dont know", "do not know", "i don't know", "dk", "not sure",
    "unsure", "skipped on web", "skipped", "refused", "ref", "no answer",
    "prefer not to say", "under 18", "unknown",
    "do not have a partner/spouse",
}


def read_any(path):
    if not os.path.exists(path):
        raise FileNotFoundError("File not found: %s" % path)
    if os.path.splitext(path)[1].lower() in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        df, err = None, None
        for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
            try:
                df = pd.read_csv(path, encoding=enc, low_memory=False)
                break
            except (UnicodeDecodeError, UnicodeError) as e:
                err = e
        if df is None:
            raise err
    n0 = len(df)
    df = df.dropna(how="all").reset_index(drop=True)
    if len(df) != n0:
        print("   [read] dropped %d all-missing row(s)" % (n0 - len(df)))
    return df


def norm(x):
    """Normalize a cell so one map matches codes, text labels and booleans."""
    if x is None:
        return None
    if isinstance(x, (bool, np.bool_)):
        return "true" if bool(x) else "false"
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        if math.isnan(float(x)):
            return None
        return int(x) if float(x).is_integer() else float(x)
    try:
        if pd.isna(x):
            return None
    except (TypeError, ValueError):
        pass
    s = str(x)
    for a, b in _DASHES.items():
        s = s.replace(a, b)
    for a, b in _QUOTES.items():
        s = s.replace(a, b)
    for sym in ("\u00ae", "\u2122", "\u00a9"):
        s = s.replace(sym, "")
    s = re.sub(r"\s+", " ", s).strip().lower().rstrip(".")
    if s in _NULLS:
        return None
    try:
        f = float(s)
        return int(f) if f.is_integer() else f
    except ValueError:
        return s


_MISS_N = set(norm(m) for m in MISSING_TOKENS)


class Audit(object):
    def __init__(self):
        self.rows = []

    def add(self, year, var, source, n_valid, n_missing, unmapped):
        self.rows.append({"year": year, "variable": var, "source": source,
                          "n_valid": n_valid, "n_missing": n_missing,
                          "unmapped_values": "; ".join(sorted(str(u) for u in unmapped)[:8])})

    def frame(self):
        return pd.DataFrame(self.rows)


def recode(df, col, mapping, year, name, audit):
    if col not in df.columns:
        audit.add(year, name, col + " (ABSENT)", 0, len(df), [])
        return pd.Series(np.nan, index=df.index, dtype=float)
    mapping = {norm(k): v for k, v in mapping.items()}
    out = np.full(len(df), np.nan)
    unmapped = set()
    for i, v in enumerate(df[col].tolist()):
        k = norm(v)
        if k is None:
            continue
        if k in mapping:
            m = mapping[k]
            out[i] = np.nan if m is None else float(m)
        elif k in _MISS_N:
            continue
        else:
            unmapped.add(k)
    s = pd.Series(out, index=df.index, dtype=float)
    audit.add(year, name, col, int(s.notna().sum()), int(s.isna().sum()), unmapped)
    return s


def recode_scale(df, col, lo, hi, year, name, audit):
    if col not in df.columns:
        audit.add(year, name, col + " (ABSENT)", 0, len(df), [])
        return pd.Series(np.nan, index=df.index, dtype=float)
    out = np.full(len(df), np.nan)
    unmapped = set()
    for i, v in enumerate(df[col].tolist()):
        k = norm(v)
        if k is None or k in _MISS_N:
            continue
        val = None
        if isinstance(k, (int, float)):
            val = float(k)
        else:
            m = re.match(r"^(-?\d+)", str(k))
            if m:
                val = float(m.group(1))
        if val is None or not (lo <= val <= hi):
            unmapped.add(k)
            continue
        out[i] = val
    s = pd.Series(out, index=df.index, dtype=float)
    audit.add(year, name, col, int(s.notna().sum()), int(s.isna().sum()), unmapped)
    return s


def clean_cat(df, col, year, audit):
    if col not in df.columns:
        audit.add(year, col, col + " (ABSENT)", 0, len(df), [])
        return pd.Series([np.nan] * len(df), index=df.index, dtype=object)
    vals = []
    for v in df[col].tolist():
        k = norm(v)
        vals.append(np.nan if (k is None or k in _MISS_N) else str(k))
    s = pd.Series(vals, index=df.index, dtype=object)
    audit.add(year, col, col, int(s.notna().sum()), int(s.isna().sum()), [])
    return s


def find_weight(df, preferred):
    for p in preferred:
        if p in df.columns:
            return p
    c = [x for x in df.columns if str(x).upper().startswith("WEIGHT")]
    return c[0] if c else None


# ==============================================================================
# 2.  VALUE MAPS
# ==============================================================================

YESNO = {1: 1, "yes": 1, "true": 1, 0: 0, 2: 0, "no": 0, "false": 0}

MAP_CONF = {1: 4, "strongly agree": 4, 2: 3, "agree": 3, 3: 2, "disagree": 2,
            4: 1, "strongly disagree": 1,
            5: None, "i don't have any financial goals": None}
MAP_ANX = {1: 1, "not at all": 1, 2: 2, "several days": 2,
           3: 3, "more than half the days": 3, 4: 4, "nearly every day": 4}

# Q48-Q54: the 2024 file uses "Not at all / Somewhat / A lot",
#          the 2025 file uses "Not at all well / Somewhat well / Very well"
MAP_REL = {1: 1.0, "not at all": 1.0, "not at all well": 1.0,
           2: 2.0, "somewhat": 2.0, "somewhat well": 2.0,
           3: 3.0, "a lot": 3.0, "very well": 3.0,
           "do not have a partner/spouse": None}

# Q24-Q30 importance items, recoded so HIGHER = MORE IMPORTANT
MAP_IMPORT = {1: 4, "very important": 4, 2: 3, "somewhat important": 3,
              3: 2, "not very important": 2, 4: 1, "not important at all": 1}

# Q34 (2024 only): HIGHER = client does more of the planning work
MAP_ENGAGE = {1: 1.0, "i mostly do financial planning on my own": 1.0,
              2: 0.5, "we equally share the financial planning activities": 0.5,
              3: 0.0, "my professional handles nearly all of my financial planning activities": 0.0}

MAP_TENURE = {1: 0.5, "less than 1 year": 0.5, 2: 1.5, "1-2 years": 1.5,
              3: 4.0, "3-5 years": 4.0, 4: 8.0, "6-10 years": 8.0,
              5: 13.0, "11-15 years": 13.0, 6: 18.0, "16-20 years": 18.0,
              7: 25.0, "more than 20 years": 25.0}

# Q21 (2024 only): 0 = no formal plan, 1 = simple, 2 = complex
MAP_Q21 = {1: 1, "simple - one or two planning-related services": 1,
           2: 2, "complex - integrated planning-related services": 2,
           3: 0, "i do not have a formal financial plan": 0}
# (norm() collapses whitespace and strips trailing dots, so the "..." labels match)

S4_LABELS = ["less than $30,000", "$30,000 to $49,999", "$50,000 to $99,999",
             "$100,000 to $249,999", "$250,000 to $499,999",
             "$500,000 to $749,999", "$750,000 to $999,999",
             "$1,000,000 to $1,999,999", "$2,000,000 to $2,999,999",
             "$3,000,000 or more"]
MAP_S4 = {}
for _i, _lab in enumerate(S4_LABELS):
    MAP_S4[_i + 1] = _i + 1
    MAP_S4[_lab] = _i + 1

# Codebook: 77 = Don't know, 98 = Skipped on web, 99 = Refused
LIT_CORRECT = {"Q56": {1, "more than $102"}, "Q57": {3, "less"},
               "Q58": {2, "fall"}, "Q59": {1, "true"}, "Q60": {2, "false"}}
LIT_DK = {77, "don't know", "dont know"}            # counted as INCORRECT (paper Sec. 4.2)
LIT_SKIP = {98, 99, "skipped on web", "refused"}    # missing


def build_literacy(df):
    items = []
    for q, correct in LIT_CORRECT.items():
        if q not in df.columns:
            continue
        corr = set(norm(c) for c in correct)
        out = np.full(len(df), np.nan)
        for i, v in enumerate(df[q].tolist()):
            k = norm(v)
            if k is None or k in LIT_SKIP:
                continue
            if k in LIT_DK:
                out[i] = 0.0                        # DK (77) counted as incorrect
                continue
            out[i] = 1.0 if k in corr else 0.0
        items.append(pd.Series(out, index=df.index))
    if not items:
        return pd.Series(np.nan, index=df.index, dtype=float)
    M = pd.concat(items, axis=1)
    return M.sum(axis=1, min_count=1).where(M.notna().sum(axis=1) >= 3).astype(float)


# ==============================================================================
# 3.  PERSON FRAME (one wave)
# ==============================================================================

def build_person(df, year, audit):
    out = pd.DataFrame(index=df.index)
    if "caseid" not in df.columns:
        raise KeyError("No 'caseid' column in the %s file." % year)
    out["caseid"] = df["caseid"]

    if year == 2025:
        wl = find_weight(df, ["WEIGHT1_W2"])
        out["W_LONG"] = pd.to_numeric(df[wl], errors="coerce") if wl else np.nan
        out["WAVECOMP"] = recode(df, "WAVECOMP",
                                 {1: 1, "completed wave 1": 1,
                                  0: 0, "did not complete wave 1": 0},
                                 year, "WAVECOMP", audit)
        out["AI"] = recode(df, "Q20_12", YESNO, year, "AI", audit)

    out["ADVISED"] = recode(df, "Q23", YESNO, year, "ADVISED", audit)
    out["HUMAN_Q20"] = recode(df, "Q20_11", YESNO, year, "HUMAN_Q20", audit)
    out["DIGITAL"] = recode(df, "Q20_9", YESNO, year, "DIGITAL", audit)
    out["SOCIAL"] = recode(df, "Q20_10", YESNO, year, "SOCIAL", audit)
    out["FAMFRIEND"] = recode(df, "Q20_8", YESNO, year, "FAMFRIEND", audit)

    out["CONF"] = recode(df, "S1", MAP_CONF, year, "CONF", audit)
    a1 = recode(df, "Q11", MAP_ANX, year, "ANX_Q11", audit)
    a2 = recode(df, "Q12_NEW", MAP_ANX, year, "ANX_Q12NEW", audit)
    A = pd.concat([a1, a2], axis=1)
    out["ANXIETY"] = A.mean(axis=1).where(A.notna().all(axis=1))   # both items required
    out["RISK_TOL"] = recode_scale(df, "Q12", 0, 10, year, "RISK_TOL", audit)
    out["FIN_LIT"] = build_literacy(df)
    out["S4_ORD"] = recode(df, "S4", MAP_S4, year, "S4_ORD", audit)

    out["FEE_IMPORT"] = recode(df, "Q26", MAP_IMPORT, year, "FEE_IMPORT", audit)
    out["FORMAT_IMPORT"] = recode(df, "Q29", MAP_IMPORT, year, "FORMAT_IMPORT", audit)

    trust = recode(df, "Q50", MAP_REL, year, "TRUST", audit)
    out["TRUST"] = trust
    rel = [recode(df, q, MAP_REL, year, q + "_rel", audit) for q in ("Q48", "Q51", "Q52")]
    R = pd.concat([trust] + rel, axis=1)
    out["RELATIONAL"] = R.mean(axis=1).where(R.notna().sum(axis=1) >= 3)
    out["SATIS_ADV"] = recode(df, "Q54", MAP_REL, year, "SATIS_ADV", audit)
    out["TENURE"] = recode(df, "Q33", MAP_TENURE, year, "TENURE", audit)

    if year == 2024:
        out["ENGAGE_SELF"] = recode(df, "Q34", MAP_ENGAGE, year, "ENGAGE_SELF", audit)
        q21 = recode(df, "Q21", MAP_Q21, year, "PLAN_TYPE", audit)
        if Q21_DK_AS_ZERO and "Q21" in df.columns:
            dk = df["Q21"].map(norm).isin([77, "i don't know", "don't know"])
            q21 = q21.where(~dk, 0.0)
        out["COMPLEX_PLAN"] = (q21 == 2).astype(float).where(q21.notna())

    # 2024 public file: GENDER (0 = Unknown, 1 = Male, 2 = Female); 2025 public file: SEX
    sexcol = "GENDER" if "GENDER" in df.columns else "SEX"
    out["SEX"] = clean_cat(df, sexcol, year, audit)
    out.loc[out["SEX"] == "0", "SEX"] = np.nan

    # AGE: public file = 3 birth cohorts (1 = 1980-2006, 2 = 1965-1979, 3 = <=1964; 99 = under 18).
    # If the file instead carries continuous age, bin it into cohorts.
    age_num = pd.to_numeric(df["AGE"], errors="coerce") if "AGE" in df.columns else pd.Series(np.nan, index=df.index)
    age_valid = age_num.where(~age_num.isin([77, 98, 99]))
    if age_valid.notna().sum() and age_valid.max() > 10:
        out["AGE"] = pd.cut(age_valid, [0, 34, 44, 54, 64, 200],
                            labels=["25-34", "35-44", "45-54", "55-64", "65+"]).astype(object)
        audit.add(year, "AGE", "AGE (continuous -> 5 cohorts)",
                  int(out["AGE"].notna().sum()), int(out["AGE"].isna().sum()), [])
    else:
        out["AGE"] = clean_cat(df, "AGE", year, audit)

    # EDUC5 and INCOME are already collapsed in the public file (3 and 4 levels)
    for c in ["EDUC5", "INCOME"]:
        out[c] = clean_cat(df, c, year, audit)
    return out


def build_panel(p24, p25):
    a = p24.copy()
    a.columns = ["caseid"] + [c + "_24" for c in a.columns[1:]]
    b = p25.copy()
    b.columns = ["caseid"] + [c + "_25" for c in b.columns[1:]]
    a = a.drop_duplicates("caseid")
    b = b.drop_duplicates("caseid")
    m = a.merge(b, on="caseid", how="inner")
    print("\n   panel: %d respondents in both waves" % len(m))
    if RESTRICT_WAVECOMP and "WAVECOMP_25" in m.columns and m["WAVECOMP_25"].notna().any():
        n0 = len(m)
        m = m[m["WAVECOMP_25"] == 1].copy()
        print("   restricted to WAVECOMP == 1: %d (from %d)" % (len(m), n0))
    # convenience copies for the outcome models
    m["CONF_24_LAG"] = m["CONF_24"]
    m["ANXIETY_24_LAG"] = m["ANXIETY_24"]
    return m.reset_index(drop=True)


def define_states(m, hdef):
    """Attach transition variables for one definition of the human state."""
    hcol = "ADVISED" if hdef == "REL" else "HUMAN_Q20"
    d = m.copy()
    d["H24"] = d[hcol + "_24"]
    d["H25"] = d[hcol + "_25"]
    d["D24"] = d["DIGITAL_24"]
    d["D25"] = d["DIGITAL_25"]

    def state(h, g):
        if pd.isna(h) or pd.isna(g):
            return np.nan
        return {(0, 0): "Neither", (0, 1): "Digital only",
                (1, 0): "Human only", (1, 1): "Human + Digital"}[(int(h), int(g))]

    d["STATE24"] = [state(h, g) for h, g in zip(d["H24"], d["D24"])]
    d["STATE25"] = [state(h, g) for h, g in zip(d["H25"], d["D25"])]

    ok = d["H24"].eq(1) & d["H25"].notna() & d["D25"].notna()
    o4 = np.full(len(d), np.nan)
    o4[ok & d["H25"].eq(1) & d["D25"].eq(0)] = 0
    o4[ok & d["H25"].eq(1) & d["D25"].eq(1)] = 1
    o4[ok & d["H25"].eq(0) & d["D25"].eq(1)] = 2
    o4[ok & d["H25"].eq(0) & d["D25"].eq(0)] = 3
    d["OUTCOME4"] = o4

    d["EXIT"] = np.where(d["H24"].eq(1) & d["H25"].notna(), (d["H25"] == 0).astype(float), np.nan)
    d["SUBST"] = np.where(ok & d["D25"].eq(1), (d["H25"] == 0).astype(float), np.nan)
    d["ENTRY"] = np.where(d["H24"].eq(0) & d["H25"].notna(), (d["H25"] == 1).astype(float), np.nan)

    for k, nm in [(1, "COMPLEMENT"), (2, "SUBSTITUTE"), (3, "DROPOUT")]:
        d[nm] = np.where(pd.notna(o4), (o4 == k).astype(float), np.nan)

    # interaction terms (trust centred within the at-risk set)
    tmean = d.loc[d["H24"] == 1, "TRUST_24"].mean()
    d.attrs["TRUST_MEAN"] = float(tmean)          # reused by the conditional effects
    d["TRUST_C_24"] = d["TRUST_24"] - tmean
    d["DIGxTRUST"] = d["DIGITAL_24"] * d["TRUST_C_24"]
    d["DIGxLIT"] = d["DIGITAL_24"] * (d["FIN_LIT_24"] - d["FIN_LIT_24"].mean())
    return d


# ==============================================================================
# 4.  ESTIMATION (numpy)
# ==============================================================================

def _pval(t, dfree):
    if not np.isfinite(t):
        return np.nan
    if HAVE_SCIPY:
        return float(2 * _st.t.sf(abs(t), max(int(dfree), 1)))
    return float(math.erfc(abs(t) / math.sqrt(2.0)))


def stars(p):
    if p is None or not np.isfinite(p):
        return ""
    return "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.10 else ""))


class Fit(object):
    def __init__(self, names, b, V, n, k, label, kind, ymean, fitstat, note=""):
        self.names, self.b, self.V = names, b, V
        self.se = np.sqrt(np.maximum(np.diag(V), 0))
        self.t = np.where(self.se > 0, b / np.where(self.se > 0, self.se, 1), np.nan)
        self.p = np.array([_pval(x, n - k) for x in self.t])
        self.n, self.k, self.label, self.kind = n, k, label, kind
        self.ymean, self.fitstat, self.note = ymean, fitstat, note

    def get(self, name):
        if name not in self.names:
            return None
        i = self.names.index(name)
        return float(self.b[i]), float(self.se[i]), float(self.p[i])

    def contrast(self, weights, label=""):
        """Linear combination sum_j w_j * b_j with its robust SE and 95% CI."""
        v = np.zeros(len(self.names))
        for nm, w in weights.items():
            if nm not in self.names:
                return None
            v[self.names.index(nm)] = w
        est = float(v.dot(self.b))
        se = float(np.sqrt(max(v.dot(self.V).dot(v), 0.0)))
        t = est / se if se > 0 else np.nan
        dfree = self.n - self.k
        crit = float(_st.t.ppf(0.975, max(dfree, 1))) if HAVE_SCIPY else 1.96
        return {"label": label, "estimate": est, "se": se, "t": t,
                "p": _pval(t, dfree), "ci_lo": est - crit * se, "ci_hi": est + crit * se}


def design(sub, nums, cats):
    cols = {"Intercept": np.ones(len(sub))}
    for v in nums:
        cols[v] = sub[v].astype(float).values
    for c in cats:
        s = sub[c].astype(str)
        vc = s.value_counts()
        rare = set(vc[vc < MIN_LEVEL_N].index)
        if rare:
            s = s.where(~s.isin(rare), "_pooled")
        vc = s.value_counts()
        if len(vc) < 2:
            continue
        for lv in vc.index[1:]:                       # most frequent = reference
            cols["%s[%s]" % (c, lv)] = (s == lv).astype(float).values
    X = pd.DataFrame(cols, index=sub.index)

    keep, A = [], np.empty((len(X), 0))
    for c in X.columns:
        v = X[c].values
        if c != "Intercept" and np.allclose(v.std(), 0):
            continue
        B = np.column_stack([A, v])
        if np.linalg.matrix_rank(B) > A.shape[1]:
            keep.append(c)
            A = B
    return X[keep]


def _ols(y, X, w=None):
    if w is not None:
        sw = np.sqrt(w)
        Xw, yw = X * sw[:, None], y * sw
    else:
        Xw, yw = X, y
    XtXi = np.linalg.pinv(Xw.T.dot(Xw))
    b = XtXi.dot(Xw.T.dot(yw))
    u = yw - Xw.dot(b)
    n, k = Xw.shape
    meat = (Xw * (u ** 2)[:, None]).T.dot(Xw)
    V = XtXi.dot(meat).dot(XtXi) * n / max(n - k, 1)        # HC1
    resid = y - X.dot(b)
    wt = np.ones(n) if w is None else w
    ybar = np.sum(wt * y) / np.sum(wt)
    r2 = 1 - np.sum(wt * resid ** 2) / np.sum(wt * (y - ybar) ** 2)
    return b, V, r2


def _logit(y, X, maxit=100):
    n, k = X.shape
    b = np.zeros(k)
    converged, note = False, ""
    for _ in range(maxit):
        p = 1.0 / (1.0 + np.exp(-np.clip(X.dot(b), -30, 30)))
        W = p * (1 - p)
        H = (X * W[:, None]).T.dot(X) + 1e-9 * np.eye(k)
        step = np.linalg.solve(H, X.T.dot(y - p))
        b = b + step
        if np.max(np.abs(step)) < 1e-8:
            converged = True
            break
    if not converged:
        note = "not converged"
    if np.any(np.abs(b[1:]) > 15):
        note = (note + "; " if note else "") + "possible separation"
    p = 1.0 / (1.0 + np.exp(-np.clip(X.dot(b), -30, 30)))
    Hi = np.linalg.pinv((X * (p * (1 - p))[:, None]).T.dot(X))
    s = X * (y - p)[:, None]
    V = Hi.dot(s.T.dot(s)).dot(Hi) * n / max(n - 1, 1)       # sandwich
    ll = np.sum(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12))
    ybar = y.mean()
    ll0 = n * (ybar * np.log(ybar + 1e-12) + (1 - ybar) * np.log(1 - ybar + 1e-12))
    return b, V, 1 - ll / ll0, note


def fit(df, y, nums, cats=(), kind="ols", weight=None, label="", sample=None):
    d = df if sample is None else df[sample]
    nums = [v for v in nums if v in d.columns]
    cats = [c for c in cats if c in d.columns]
    need = [y] + nums + cats + ([weight] if weight else [])
    sub = d[need].dropna()
    if weight:
        sub = sub[sub[weight] > 0]
    if len(sub) < 40 or sub[y].nunique() < 2:
        print("   !! %s : %d usable rows (or no variation in %s) -- skipped" % (label, len(sub), y))
        return None
    X = design(sub, nums, cats)
    yv = sub[y].astype(float).values
    if kind == "logit":
        b, V, stat, note = _logit(yv, X.values)
    else:
        w = sub[weight].astype(float).values if weight else None
        b, V, stat = _ols(yv, X.values, w)
        note = "weighted" if weight else ""
    return Fit(list(X.columns), b, V, len(sub), X.shape[1], label, kind,
               float(yv.mean()), float(stat), note)


# ==============================================================================
# 5.  OUTPUT HELPERS
# ==============================================================================

def show(obj, title):
    print("\n" + "=" * 104)
    print(title)
    print("=" * 104)
    with pd.option_context("display.width", 230, "display.max_columns", 40,
                           "display.max_rows", 400):
        print(obj.to_string() if isinstance(obj, (pd.DataFrame, pd.Series)) else obj)


def table(fits, rows, title):
    fits = [f for f in fits if f is not None]
    if not fits:
        return pd.DataFrame()
    body = []
    for r in rows:
        c_line, s_line, hit = {}, {}, False
        for f in fits:
            g = f.get(r)
            if g is None:
                c_line[f.label], s_line[f.label] = "", ""
            else:
                hit = True
                c_line[f.label] = "%.4f%s" % (g[0], stars(g[2]))
                s_line[f.label] = "(%.4f)" % g[1]
        if hit:
            body.append((PRETTY.get(r, r), c_line))
            body.append(("", s_line))
    out = pd.DataFrame([b[1] for b in body], index=[b[0] for b in body])
    tail = {"Model": {}, "Controls": {}, "Mean of DV": {}, "Observations": {},
            "R-sq / pseudo R-sq": {}, "Note": {}}
    for f in fits:
        tail["Model"][f.label] = "LPM" if f.kind == "ols" else "Logit"
        tail["Controls"][f.label] = "Yes" if any("[" in n for n in f.names) else "No"
        tail["Mean of DV"][f.label] = "%.4f" % f.ymean
        tail["Observations"][f.label] = "%d" % f.n
        tail["R-sq / pseudo R-sq"][f.label] = "%.4f" % f.fitstat
        tail["Note"][f.label] = f.note
    out = pd.concat([out, pd.DataFrame(tail).T])
    out.index.name = title
    return out


def transition_matrix(d, weight=None):
    s = d.dropna(subset=["STATE24", "STATE25"])
    order = ["Neither", "Digital only", "Human only", "Human + Digital"]
    cnt = pd.crosstab(s["STATE24"], s["STATE25"]).reindex(index=order, columns=order, fill_value=0)
    pct = cnt.div(cnt.sum(axis=1).replace(0, np.nan), axis=0) * 100
    out = cnt.astype(str) + " (" + pct.round(1).astype(str) + "%)"
    out["Row N"] = cnt.sum(axis=1)
    if weight and weight in s.columns and s[weight].notna().any():
        w = s[s[weight] > 0]
        wc = w.pivot_table(index="STATE24", columns="STATE25", values=weight,
                           aggfunc="sum", fill_value=0).reindex(index=order, columns=order, fill_value=0)
        wp = wc.div(wc.sum(axis=1).replace(0, np.nan), axis=0) * 100
        for c in order:
            out["w% " + c] = wp[c].round(1)
    return out


# ==============================================================================
# 6.  STAGES
# ==============================================================================

def stage1(d_rel, d_use):
    print("\n" + "#" * 104)
    print("# STAGE 1 -- TRANSITIONS AND CELL SIZES  (read this first)")
    print("#" * 104)
    res = {}
    for tag, d in [("REL (Q23)", d_rel), ("USE (Q20_11)", d_use)]:
        tm = transition_matrix(d, "W_LONG_25")
        show(tm, "TABLE 1 -- 2024 state (rows) x 2025 state (columns), human = %s" % tag)
        res["T1_transitions_" + tag.split()[0]] = tm

        at = d[d["H24"] == 1]
        c = at["OUTCOME4"].value_counts().reindex([0, 1, 2, 3]).fillna(0).astype(int)
        print("\n   Among H24 = 1 (n = %d):" % int(at["OUTCOME4"].notna().sum()))
        for k, nm in [(0, "Retain"), (1, "Complement"), (2, "Substitute"), (3, "Drop out")]:
            flag = "   <-- below %d" % WARN_CELL if c[k] < WARN_CELL else ""
            print("      %-11s %5d%s" % (nm, c[k], flag))
        n_new = int(((at["D24"] == 0) & at["SUBST"].notna()).sum())
        print("   SUBST sample (H24=1 & D25=1): %d ; of which newly digital (D24=0): %d"
              % (int(at["SUBST"].notna().sum()), n_new))

    # consistency between the two human definitions
    j = d_rel[["caseid", "H24", "H25"]].merge(
        d_use[["caseid", "H24", "H25"]], on="caseid", suffixes=("_REL", "_USE"))

    def lab(a, b):
        if pd.isna(a) or pd.isna(b):
            return np.nan
        return {(1, 1): "stay", (1, 0): "EXIT", (0, 1): "ENTER", (0, 0): "never"}[(int(a), int(b))]

    j["REL"] = [lab(a, b) for a, b in zip(j["H24_REL"], j["H25_REL"])]
    j["USE"] = [lab(a, b) for a, b in zip(j["H24_USE"], j["H25_USE"])]
    ct = pd.crosstab(j["REL"], j["USE"], margins=True)
    show(ct, "TABLE 1C -- Do the two human definitions agree?  rows = Q23, cols = Q20_11")
    print("""
   How to read 1C: if most Q20_11 'EXIT' cases are Q23 'stay', the Q20_11 exits are
   people who kept their professional but did not consult them this year.  That is
   a change in consultation intensity, not substitution, and the paper must use Q23.""")
    res["T1C_definition_agreement"] = ct
    return res


def stage2(d):
    print("\n" + "#" * 104)
    print("# STAGE 2 -- 2024 PROFILES BY 2025 OUTCOME  (H24 = 1)")
    print("#" * 104)
    at = d[d["OUTCOME4"].notna()].copy()
    at["Outcome"] = at["OUTCOME4"].map({0: "Retain", 1: "Complement", 2: "Substitute", 3: "Drop out"})
    vars_ = CORE + RELATIONSHIP
    prof = at.groupby("Outcome")[[v for v in vars_ if v in at.columns]].mean().T
    prof = prof[[c for c in ["Retain", "Complement", "Substitute", "Drop out"] if c in prof.columns]]
    prof["N non-missing"] = [int(at[v].notna().sum()) for v in prof.index]
    prof.index = [PRETTY.get(v, v) for v in prof.index]
    show(prof.round(3), "TABLE 2 -- 2024 means by 2025 outcome")
    return {"T2_profiles": prof}


def stage3(d):
    print("\n" + "#" * 104)
    print("# STAGE 3 -- WHO LEAVES HUMAN ADVICE?  EXIT among H24 = 1")
    print("#" * 104)
    s = d["H24"] == 1
    f1 = fit(d, "EXIT", CORE, CATS, label="(1) LPM", sample=s)
    f2 = fit(d, "EXIT", CORE + RELATIONSHIP, CATS, label="(2) + relationship", sample=s)
    f3 = fit(d, "EXIT", CORE + ["TRUST_C_24", "ENGAGE_SELF_24", "TENURE_24", "DIGxTRUST"],
             CATS, label="(3) digital x trust", sample=s)
    f4 = fit(d, "EXIT", CORE, CATS, kind="logit", label="(4) logit", sample=s)
    f5 = fit(d, "EXIT", CORE + RELATIONSHIP, CATS, kind="logit",
             label="(5) logit + relationship", sample=s)
    f6 = fit(d, "EXIT", CORE + ["TRUST_C_24", "ENGAGE_SELF_24", "TENURE_24", "DIGxTRUST"],
             CATS, kind="logit", label="(6) logit x trust", sample=s)
    rows = CORE + RELATIONSHIP + ["TRUST_C_24", "DIGxTRUST"]
    t = table([f1, f2, f3, f4, f5, f6], rows, "DV = 1 if no human advice in 2025")
    show(t, "TABLE 3 -- Exit from human advice")
    out = {"T3_exit": t}

    # ---- conditional effect of 2024 digital use on exit, by trust level ----
    if f3 is None:
        return out
    tmean = d.attrs.get("TRUST_MEAN", d.loc[d["H24"] == 1, "TRUST_24"].mean())
    print("\n   trust centred at %.4f (mean among H24 = 1)" % tmean)

    # cell sizes: who identifies each conditional effect
    at = d[(d["H24"] == 1) & d["EXIT"].notna() & d["TRUST_24"].notna() & d["DIGITAL_24"].notna()]
    cnt = pd.crosstab(at["TRUST_24"], at["DIGITAL_24"])

    recs = []
    for tv, lab in [(1.0, "Not at all (1)"), (2.0, "Somewhat (2)"),
                    (tmean, "Sample mean (%.2f)" % tmean), (3.0, "A lot (3)")]:
        c = f3.contrast({"DIGITAL_24": 1.0, "DIGxTRUST": tv - tmean},
                        "Effect of digital use | trust = %s" % lab)
        if c is None:
            continue
        n_dig = int(cnt.loc[tv, 1.0]) if (tv in cnt.index and 1.0 in cnt.columns) else np.nan
        n_all = int(cnt.loc[tv].sum()) if tv in cnt.index else np.nan
        recs.append({"Trust level": lab,
                     "Effect on P(exit)": round(c["estimate"], 4),
                     "Std. Error": round(c["se"], 4),
                     "95% CI low": round(c["ci_lo"], 4),
                     "95% CI high": round(c["ci_hi"], 4),
                     "p-value": round(c["p"], 4),
                     "": stars(c["p"]),
                     "N at this trust level": n_all,
                     "of whom digital users": n_dig})
    ce = pd.DataFrame(recs)
    show(ce, "TABLE 3B -- Conditional effect of 2024 digital use on exit, by trust "
             "(from column 3)")
    print("""
   Read the last two columns before interpreting a row.  An effect identified
   from a handful of digital users at a given trust level is an extrapolation
   of the linear interaction, not evidence about that group.""")
    out["T3B_conditional_effects"] = ce

    # difference between the effect at trust = 2 and trust = 3 equals -b(DIGxTRUST)
    diff = f3.contrast({"DIGxTRUST": -1.0}, "Effect at trust 2 minus effect at trust 3")
    if diff:
        print("\n   %s: %.4f (SE %.4f, p = %.4f)"
              % (diff["label"], diff["estimate"], diff["se"], diff["p"]))

    plot_conditional(f3, tmean, cnt, os.path.join(OUTDIR, "Fig_conditional_effect_trust.png"))

    # ---- Table 2C: observed 2025 outcomes by 2024 trust x 2024 digital use ----
    seg = d[(d["H24"] == 1) & d["OUTCOME4"].notna()
            & d["TRUST_24"].isin([2.0, 3.0]) & d["DIGITAL_24"].notna()].copy()
    seg["Segment"] = (seg["TRUST_24"].map({3.0: "High trust", 2.0: "Moderate trust"}) + ", "
                      + seg["DIGITAL_24"].map({1.0: "digital", 0.0: "no digital"}))
    lab4 = {0: "Retain", 1: "Complement", 2: "Substitute", 3: "Disengage"}
    order = ["High trust, no digital", "High trust, digital",
             "Moderate trust, no digital", "Moderate trust, digital"]
    t2c = (pd.crosstab(seg["Segment"], seg["OUTCOME4"].map(lab4), normalize="index") * 100)
    t2c = t2c.reindex(index=order, columns=["Retain", "Complement", "Substitute", "Disengage"], fill_value=0.0)
    t2c.insert(0, "Exit", t2c["Substitute"] + t2c["Disengage"])
    t2c.insert(0, "N", seg["Segment"].value_counts().reindex(order).fillna(0).astype(int))
    ex = seg[seg["OUTCOME4"].isin([2, 3])]
    t2c["Share of households (%)"] = t2c["N"] / t2c["N"].sum() * 100
    t2c["Share of exits (%)"] = (ex["Segment"].value_counts().reindex(order).fillna(0)
                                 / max(len(ex), 1) * 100)
    show(t2c.round(1), "TABLE 2C -- 2025 outcomes by 2024 trust x 2024 digital use (row %)")
    out["T2C_segments"] = t2c
    return out


def plot_conditional(f, tmean, cnt, path):
    """Marginal effect of 2024 digital use on exit across trust, with 95% CI."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print("   (figure skipped: %s)" % e)
        return

    grid = np.linspace(1.0, 3.0, 81)
    est, lo, hi = [], [], []
    for tv in grid:
        c = f.contrast({"DIGITAL_24": 1.0, "DIGxTRUST": tv - tmean})
        est.append(c["estimate"])
        lo.append(c["ci_lo"])
        hi.append(c["ci_hi"])

    fig, ax = plt.subplots(figsize=(6.5, 4.3))
    ax.fill_between(grid, lo, hi, color="0.80", label="95% confidence interval")
    ax.plot(grid, est, color="black", lw=1.6, label="Conditional effect")
    ax.axhline(0.0, color="0.35", lw=0.8, ls="--")

    # observed trust levels: point estimates with CIs and cell sizes
    for tv in (1.0, 2.0, 3.0):
        c = f.contrast({"DIGITAL_24": 1.0, "DIGxTRUST": tv - tmean})
        ax.errorbar(tv, c["estimate"],
                    yerr=[[c["estimate"] - c["ci_lo"]], [c["ci_hi"] - c["estimate"]]],
                    fmt="o", color="black", capsize=4, ms=5)
        if tv in cnt.index:
            nd = int(cnt.loc[tv, 1.0]) if 1.0 in cnt.columns else 0
            ax.annotate("n = %d\n(%d digital)" % (int(cnt.loc[tv].sum()), nd),
                        xy=(tv, c["ci_hi"]), xytext=(0, 8), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8)

    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["Not at all (1)", "Somewhat (2)", "A lot (3)"])
    ax.set_xlim(0.8, 3.2)
    ax.set_xlabel("Trust in financial professional, 2024")
    ax.set_ylabel("Effect of 2024 digital use\non probability of exit")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    fig.savefig(path.replace(".png", ".pdf"))
    plt.close(fig)
    print("   figure saved: %s (+ .pdf)" % path)


def stage4(d, d_alt):
    print("\n" + "#" * 104)
    print("# STAGE 4 -- SUBSTITUTE OR COMPLEMENT?  among H24 = 1 & D25 = 1   (HEADLINE)")
    print("#" * 104)
    print("   DV = 1 if the consumer dropped human advice (substitute), 0 if kept it (complement)")
    s = d["SUBST"].notna()
    f1 = fit(d, "SUBST", CORE, CATS, label="(1) LPM", sample=s)
    f2 = fit(d, "SUBST", CORE + RELATIONSHIP, CATS, label="(2) + relationship", sample=s)
    f3 = fit(d, "SUBST", CORE, CATS, weight="W_LONG_25", label="(3) weighted", sample=s)
    f4 = fit(d, "SUBST", CORE, CATS, kind="logit", label="(4) logit", sample=s)
    core_nodig = [v for v in CORE if v != "DIGITAL_24"]
    f5 = fit(d, "SUBST", core_nodig, CATS, label="(5) newly digital",
             sample=s & (d["D24"] == 0))
    f6 = fit(d_alt, "SUBST", CORE, CATS, label="(6) human = alt def",
             sample=d_alt["SUBST"].notna())
    rows = CORE + RELATIONSHIP
    t = table([f1, f2, f3, f4, f5, f6], rows, "DV = substitute (1) vs complement (0)")
    show(t, "TABLE 4 -- What makes digital a substitute rather than a complement?")
    return {"T4_substitute_vs_complement": t}


def stage5(d):
    print("\n" + "#" * 104)
    print("# STAGE 5 -- FOUR OUTCOMES AS SEPARATE LPMs  (H24 = 1; coefficients sum to 0)")
    print("#" * 104)
    s = d["OUTCOME4"].notna()
    fits = []
    for k, nm in [(0, "Retain"), (1, "Complement"), (2, "Substitute"), (3, "Drop out")]:
        d["_y%d" % k] = np.where(s, (d["OUTCOME4"] == k).astype(float), np.nan)
        fits.append(fit(d, "_y%d" % k, CORE, CATS, label=nm, sample=s))
    t = table(fits, CORE, "DV = 1[outcome]")
    show(t, "TABLE 5 -- Multinomial outcome, one LPM per category")
    return {"T5_four_outcomes": t}


def stage6(d):
    print("\n" + "#" * 104)
    print("# STAGE 6 -- DIGITAL AS A GATEWAY?  ENTRY among H24 = 0")
    print("#" * 104)
    s = d["H24"] == 0
    f1 = fit(d, "ENTRY", CORE, CATS, label="(1) LPM", sample=s)
    f2 = fit(d, "ENTRY", CORE + ["DIGxLIT"], CATS, label="(2) digital x literacy", sample=s)
    f3 = fit(d, "ENTRY", CORE, CATS, kind="logit", label="(3) logit", sample=s)
    t = table([f1, f2, f3], CORE + ["DIGxLIT"], "DV = 1 if human advice in 2025")
    show(t, "TABLE 6 -- Entry into human advice")
    return {"T6_entry": t}


def stage7(d):
    print("\n" + "#" * 104)
    print("# STAGE 7 -- DOES THE TRANSITION TYPE TRACK OUTCOMES?  (descriptive)")
    print("#" * 104)
    print("   Reference = Retain.  Lagged outcome and 2024 controls included.")
    s = d["OUTCOME4"].notna()
    base = ["COMPLEMENT", "SUBSTITUTE", "DROPOUT"]
    ctrl = ["FIN_LIT_24", "RISK_TOL_24", "S4_ORD_24"]
    f1 = fit(d, "CONF_25", base + ["CONF_24_LAG"] + ctrl, CATS,
             label="Confidence 2025", sample=s)
    f2 = fit(d, "ANXIETY_25", base + ["ANXIETY_24_LAG"] + ctrl, CATS,
             label="Anxiety 2025", sample=s)
    t = table([f1, f2], base + ["CONF_24_LAG", "ANXIETY_24_LAG"], "Outcome in 2025")
    show(t, "TABLE 7 -- Financial wellbeing by transition type")
    return {"T7_consequences": t}


def stage8(d):
    print("\n" + "#" * 104)
    print("# STAGE 8 -- AI EXTENSION  (AI measured in 2025 only)")
    print("#" * 104)
    at = d[d["OUTCOME4"].notna()].copy()
    at["Outcome"] = at["OUTCOME4"].map({0: "Retain", 1: "Complement", 2: "Substitute", 3: "Drop out"})
    ct = at.groupby("Outcome")["AI_25"].agg(["count", "sum", "mean"])
    ct.columns = ["N", "AI users 2025", "AI share"]
    show(ct.round(4), "TABLE 8A -- AI use in 2025 by transition type")
    s = d["SUBST"].notna()
    f = fit(d, "SUBST", CORE + ["AI_25"], CATS, label="SUBST + AI_25", sample=s)
    t = table([f], CORE + ["AI_25"], "AI_25 is contemporaneous with the outcome")
    show(t, "TABLE 8B -- Adding AI to the headline model (not causal)")
    return {"T8A_ai_by_outcome": ct, "T8B_ai_in_subst": t}


# ==============================================================================
# 7.  MAIN
# ==============================================================================

def main():
    print("=" * 104)
    print(" FPLS 2024 -> 2025 : HUMAN -> DIGITAL TRANSITIONS")
    print("=" * 104)
    if not os.path.isdir(OUTDIR):
        os.makedirs(OUTDIR)
    audit = Audit()

    print("\n[1] Loading ...")
    raw24 = read_any(PATH24)
    raw25 = read_any(PATH25)
    print("    2024: %d x %d   2025: %d x %d" % (raw24.shape + raw25.shape))
    p24 = build_person(raw24, 2024, audit)
    p25 = build_person(raw25, 2025, audit)

    aud = audit.frame()
    show(aud, "SECTION 0 -- CODING AUDIT  (unmapped_values should be empty)")
    if (aud["unmapped_values"].astype(str).str.len() > 0).any():
        print("\n  !! Some values were not recognized and were set to missing.  Check above.")

    panel = build_panel(p24, p25)
    d_rel = define_states(panel, "REL")
    d_use = define_states(panel, "USE")
    d, d_alt = (d_rel, d_use) if PRIMARY_HUMAN == "REL" else (d_use, d_rel)
    print("\n   primary human definition: %s" % PRIMARY_HUMAN)

    res = {"Coding_audit": aud}
    res.update(stage1(d_rel, d_use))
    res.update(stage2(d))
    res.update(stage3(d))
    res.update(stage4(d, d_alt))
    res.update(stage5(d))
    res.update(stage6(d))
    res.update(stage7(d))
    res.update(stage8(d))

    print("\n[2] Writing output to %s" % OUTDIR)
    for name, obj in res.items():
        if isinstance(obj, pd.DataFrame) and len(obj):
            obj.to_csv(os.path.join(OUTDIR, name + ".csv"), encoding="utf-8-sig")
    try:
        with pd.ExcelWriter(os.path.join(OUTDIR, "FPLS_IS_Tables.xlsx"), engine="openpyxl") as w:
            for name, obj in res.items():
                if isinstance(obj, pd.DataFrame) and len(obj):
                    obj.to_excel(w, sheet_name=name[:31])
    except Exception as e:
        print("    (Excel export skipped: %s)" % e)
    d.to_csv(os.path.join(OUTDIR, "panel_frame.csv"), index=False, encoding="utf-8-sig")
    print("\nDone.")


if __name__ == "__main__":
    main()
