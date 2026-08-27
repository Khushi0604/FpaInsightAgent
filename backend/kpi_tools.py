"""
KPI Tools — the LangGraph agent calls these as structured tools.
Each tool reads from the CSVs directly so numbers are always accurate.
"""

import os
import pandas as pd
from langchain_core.tools import tool

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def _load(fname: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(DATA_DIR, fname))


@tool
def get_mrr_summary(months: int = 3) -> str:
    """
    Returns MRR, ARR, and MoM growth for the last N months.
    Use this when the user asks about revenue, MRR, ARR, or growth trends.
    """
    df = _load("mrr.csv")
    df = df.tail(months)
    lines = [f"MRR Summary — last {months} months:"]
    for _, r in df.iterrows():
        lines.append(
            f"  {r['month']}: MRR=${r['mrr']:,.0f} | ARR=${r['arr']:,.0f} | "
            f"New MRR=${r['new_mrr']:,.0f} | Churned MRR=${r['churned_mrr']:,.0f}"
        )
    return "\n".join(lines)


@tool
def get_churn_analysis() -> str:
    """
    Returns churn rate, number of churned customers, and breakdown by plan.
    Use this when the user asks about churn, retention, or customer loss.
    """
    df = _load("customers.csv")
    total   = len(df)
    churned = df["is_churned"].sum()
    rate    = round(churned / total * 100, 2)
    by_plan = df.groupby("plan")["is_churned"].agg(["sum", "count"])
    by_plan["churn_rate_%"] = (by_plan["sum"] / by_plan["count"] * 100).round(2)
    lines = [
        f"Churn Analysis:",
        f"  Total Customers : {total}",
        f"  Churned         : {churned}",
        f"  Churn Rate      : {rate}%",
        "",
        "  By Plan:",
    ]
    for plan, row in by_plan.iterrows():
        lines.append(
            f"    {plan}: {int(row['sum'])}/{int(row['count'])} churned "
            f"({row['churn_rate_%']}%)"
        )
    return "\n".join(lines)


@tool
def get_cac_ltv() -> str:
    """
    Returns Customer Acquisition Cost (CAC), LTV, and LTV:CAC ratio.
    Use this when the user asks about CAC, LTV, unit economics, or marketing efficiency.
    """
    cac_df  = _load("cac.csv")
    cust_df = _load("customers.csv")

    avg_cac      = round(cac_df["cac"].mean(), 2)
    avg_mrr      = round(
        cust_df[cust_df["is_churned"] == 0]["mrr"].mean(), 2
    )
    churn_rate   = round(cust_df["is_churned"].mean(), 4)
    ltv          = round(avg_mrr / churn_rate if churn_rate else 0, 2)
    ltv_cac      = round(ltv / avg_cac, 2)
    total_spend  = round(cac_df["marketing_spend"].sum() + cac_df["sales_spend"].sum(), 2)

    return (
        f"Unit Economics:\n"
        f"  Avg CAC            : ${avg_cac:,.2f}\n"
        f"  Avg MRR/Customer   : ${avg_mrr:,.2f}\n"
        f"  Estimated LTV      : ${ltv:,.2f}\n"
        f"  LTV:CAC Ratio      : {ltv_cac}x  "
        f"({'Healthy ✓' if ltv_cac >= 3 else 'Needs improvement ⚠️'})\n"
        f"  Total S&M Spend    : ${total_spend:,.2f}"
    )


@tool
def get_kpi_dashboard() -> str:
    """
    Returns a full KPI dashboard snapshot with all key metrics.
    Use this when the user asks for an overview, summary, or dashboard.
    """
    df    = _load("kpi_summary.csv")
    lines = ["📊 FP&A KPI Dashboard:", ""]
    for _, r in df.iterrows():
        unit = r["unit"]
        val  = r["value"]
        if unit == "USD":
            formatted = f"${val:,.2f}"
        elif unit == "%":
            formatted = f"{val}%"
        elif unit == "x":
            formatted = f"{val}x"
        else:
            formatted = str(int(val))
        lines.append(f"  {r['metric']:<28} {formatted}")
    return "\n".join(lines)


@tool
def get_customer_segments() -> str:
    """
    Returns customer breakdown by plan, country, and industry.
    Use this when the user asks about customer segments, distribution, or breakdown.
    """
    df = _load("customers.csv")
    active = df[df["is_churned"] == 0]

    by_plan     = active.groupby("plan")["mrr"].agg(["count", "sum", "mean"]).round(2)
    by_country  = active["country"].value_counts().head(5)
    by_industry = active["industry"].value_counts().head(5)

    lines = ["Customer Segments (Active Only):", "", "  By Plan:"]
    for plan, row in by_plan.iterrows():
        lines.append(
            f"    {plan}: {int(row['count'])} customers | "
            f"Total MRR=${row['sum']:,.0f} | Avg MRR=${row['mean']:,.0f}"
        )
    lines += ["", "  Top Countries:"]
    for country, cnt in by_country.items():
        lines.append(f"    {country}: {cnt}")
    lines += ["", "  Top Industries:"]
    for ind, cnt in by_industry.items():
        lines.append(f"    {ind}: {cnt}")

    return "\n".join(lines)


# Expose all tools as a list for the agent
ALL_TOOLS = [
    get_mrr_summary,
    get_churn_analysis,
    get_cac_ltv,
    get_kpi_dashboard,
    get_customer_segments,
]
