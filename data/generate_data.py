"""
Generate synthetic FP&A data for the agent.
Run once: python data/generate_data.py
Swap these CSVs later with your real data — same column names, it'll just work.
"""

import pandas as pd
import numpy as np
import random
from datetime import date, timedelta

random.seed(42)
np.random.seed(42)

# ── 1. Monthly Revenue / MRR ──────────────────────────────────────────────────
months = pd.date_range("2023-01-01", periods=18, freq="MS")
base_mrr = 50_000
mrr_data = []
for i, m in enumerate(months):
    new_mrr       = round(np.random.normal(8000, 1500), 2)
    expansion_mrr = round(np.random.normal(3000, 800), 2)
    churned_mrr   = round(np.random.normal(2000, 600), 2)
    contraction   = round(np.random.normal(500, 200), 2)
    base_mrr      = base_mrr + new_mrr + expansion_mrr - churned_mrr - contraction
    mrr_data.append({
        "month":          m.strftime("%Y-%m"),
        "mrr":            round(base_mrr, 2),
        "new_mrr":        max(new_mrr, 0),
        "expansion_mrr":  max(expansion_mrr, 0),
        "churned_mrr":    max(churned_mrr, 0),
        "contraction_mrr":max(contraction, 0),
        "arr":            round(base_mrr * 12, 2),
    })
pd.DataFrame(mrr_data).to_csv("data/mrr.csv", index=False)
print("✓ mrr.csv")

# ── 2. Customers ──────────────────────────────────────────────────────────────
plans  = ["Starter", "Growth", "Enterprise"]
prices = {"Starter": 99, "Growth": 299, "Enterprise": 999}
customers = []
for cid in range(1, 201):
    plan       = random.choices(plans, weights=[50, 35, 15])[0]
    signup     = date(2022, 1, 1) + timedelta(days=random.randint(0, 730))
    churned    = random.random() < 0.18
    churn_date = signup + timedelta(days=random.randint(60, 540)) if churned else None
    customers.append({
        "customer_id":   f"C{cid:04d}",
        "plan":          plan,
        "mrr":           prices[plan] + random.randint(-10, 50),
        "signup_date":   signup,
        "churn_date":    churn_date,
        "is_churned":    int(churned),
        "country":       random.choice(["India","USA","UK","Germany","Canada","Australia"]),
        "industry":      random.choice(["SaaS","Fintech","Healthcare","E-commerce","EdTech"]),
    })
pd.DataFrame(customers).to_csv("data/customers.csv", index=False)
print("✓ customers.csv")

# ── 3. CAC / Marketing Spend ──────────────────────────────────────────────────
cac_data = []
for m in months:
    spend        = round(np.random.normal(25000, 4000), 2)
    new_custs    = random.randint(12, 30)
    cac_data.append({
        "month":             m.strftime("%Y-%m"),
        "marketing_spend":   max(spend, 0),
        "new_customers":     new_custs,
        "cac":               round(max(spend, 0) / new_custs, 2),
        "sales_spend":       round(np.random.normal(15000, 2000), 2),
    })
pd.DataFrame(cac_data).to_csv("data/cac.csv", index=False)
print("✓ cac.csv")

# ── 4. KPI Summary (pre-computed, agent can also compute live) ────────────────
latest     = mrr_data[-1]
prev       = mrr_data[-2]
total_custs   = len(customers)
active_custs  = sum(1 for c in customers if not c["is_churned"])
churned_custs = total_custs - active_custs
churn_rate    = round(churned_custs / total_custs * 100, 2)
avg_cac       = round(sum(r["cac"] for r in cac_data) / len(cac_data), 2)
avg_mrr_per   = round(sum(c["mrr"] for c in customers if not c["is_churned"]) / active_custs, 2)
ltv           = round(avg_mrr_per / (churn_rate / 100) if churn_rate else 0, 2)

kpis = [{
    "metric": "Current MRR",         "value": latest["mrr"],          "unit": "USD"},
    {"metric": "ARR",                 "value": latest["arr"],          "unit": "USD"},
    {"metric": "MoM MRR Growth",      "value": round((latest["mrr"]-prev["mrr"])/prev["mrr"]*100,2), "unit": "%"},
    {"metric": "New MRR (latest mo)", "value": latest["new_mrr"],      "unit": "USD"},
    {"metric": "Churned MRR",         "value": latest["churned_mrr"],  "unit": "USD"},
    {"metric": "Total Customers",     "value": total_custs,            "unit": "count"},
    {"metric": "Active Customers",    "value": active_custs,           "unit": "count"},
    {"metric": "Churn Rate",          "value": churn_rate,             "unit": "%"},
    {"metric": "Avg CAC",             "value": avg_cac,                "unit": "USD"},
    {"metric": "Avg MRR per Customer","value": avg_mrr_per,            "unit": "USD"},
    {"metric": "LTV (est.)",          "value": ltv,                    "unit": "USD"},
    {"metric": "LTV:CAC Ratio",       "value": round(ltv/avg_cac,2),   "unit": "x"},
]
pd.DataFrame(kpis).to_csv("data/kpi_summary.csv", index=False)
print("✓ kpi_summary.csv")
print("\nAll data generated. You'll find 4 CSVs in data/")
