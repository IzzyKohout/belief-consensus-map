#!/usr/bin/env python3
"""
build_data.py — preprocesses GD aggregate data into data.json for the map.

Usage:
    python preprocess/build_data.py --gd-path ./global-dialogues --round 8 --output ./app/data.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── Exact question text as it appears in GD8 aggregate file ──────────────────
# Maps question_text -> (question_code, category_code, scale_type)

INDICATOR_QUESTIONS = [
    ("Overall, would you say the increased use of artificial intelligence (AI) in daily life makes you feel\u2026",
     "ai_feeling_excited_concerned", "AI Perception", "excited_concerned", "Overall feeling about AI in daily life"),
    ("Thinking about the last three months, how often, if at all, have you noticed AI systems in your daily life?",
     "ai_notice_frequency", "AI Exposure", "frequency", "Frequency of noticing AI systems"),
    ("Thinking about the last three months, how often, if at all, have you noticed human interactions which have been replaced with automated systems?",
     "automation_notice_frequency", "AI Exposure", "frequency", "Frequency of noticing automation replacing humans"),
    ("Thinking about the last three months, how often, if at all, have you been expected to use an AI system at work?",
     "ai_use_work_expected_freq", "AI Use", "frequency", "Expected AI use at work"),
    ("Thinking about the last three months, how often, if at all, have you personally chosen to use an AI system at work?",
     "ai_use_work_chosen_freq", "AI Use", "frequency", "Chosen AI use at work"),
    ("Thinking about the last three months, how often, if at all, have you personally chosen to use an AI system in your personal life?",
     "ai_use_personal_chosen_freq", "AI Use", "frequency", "Chosen AI use in personal life"),
    ("Thinking about the last three months, how often, if at all, have you interacted with AI systems to get advice on a sensitive personal issue or to get emotional support?",
     "ai_use_sensitive_advice_freq", "AI Use", "frequency", "AI use for emotional support or sensitive advice"),
    ("Thinking about the last three months, how often, if at all, have you interacted with AI systems to complete an action in the real world on your behalf without your supervision?",
     "ai_use_real_world_action_freq", "AI Use", "frequency", "AI use for unsupervised real-world actions"),
    ("Considering both potential benefits and risks, how do you assess the overall impact on society of AI chatbots?",
     "impact_society_ai_chatbots", "Societal Impact Assessment", "benefits_risks", "Perceived societal impact of AI chatbots"),
    ("Considering both potential benefits and risks, how do you assess the overall impact on society of AI systems that can perform tasks in the real world without human supervision?",
     "impact_society_ai_unsupervised", "Societal Impact Assessment", "benefits_risks", "Perceived societal impact of unsupervised AI"),
    ("Considering both potential benefits and risks, how do you assess the overall impact on society of AI systems that can outperform humans on most economically valuable work?",
     "impact_society_ai_outperform", "Societal Impact Assessment", "benefits_risks", "Perceived societal impact of AI outperforming humans"),
    ("To what extent, if at all, do you generally trust social media companies to do what is right?",
     "trust_social_media_co", "Trust in Institutions", "trust", "Trust in social media companies"),
    ("To what extent, if at all, do you generally trust companies building AI to do what is right?",
     "trust_ai_co", "Trust in Institutions", "trust", "Trust in AI companies"),
    ("To what extent, if at all, do you generally trust public research institutions to do what is right?",
     "trust_research_inst", "Trust in Institutions", "trust", "Trust in public research institutions"),
    ("To what extent, if at all, do you trust your AI chatbot (eg ChatGPT) to act in your best interest?",
     "trust_personal_ai_chatbot", "Trust in Specific Actors", "trust", "Personal trust in AI chatbot"),
    ("Do you agree or disagree with this statement? AI could make better decisions on my behalf than my government representatives.",
     "ai_vs_govt_decisions", "AI Governance", "agree_disagree_unsure", "AI vs government decision-making"),
    ("Do you think the increased use of AI across society is likely to make your cost of living better, worse or stay the same in the next 10 years?",
     "ai_impact_future_cost_living", "Future Impact — Personal", "impact", "AI impact on cost of living"),
    ("Do you think the increased use of AI across society is likely to make the amount of free time you have better, worse or stay the same in the next 10 years?",
     "ai_impact_future_free_time", "Future Impact — Personal", "impact", "AI impact on free time"),
    ("Do you think the increased use of AI across society is likely to make your community's well-being better, worse or stay the same in the next 10 years?",
     "ai_impact_future_community", "Future Impact — Societal", "impact", "AI impact on community well-being"),
    ("Do you think the increased use of AI across society is likely to make the availability of good jobs better, worse or stay the same in the next 10 years?",
     "ai_impact_future_jobs", "Future Impact — Societal", "impact", "AI impact on job availability"),
    ("Do you think the increased use of AI across society is likely to make your sense of purpose better, worse or stay the same in the next 10 years?",
     "ai_impact_future_purpose", "Future Impact — Personal", "impact", "AI impact on sense of purpose"),
    ("So far, what has been the overall impact of AI on your daily life?",
     "ai_impact_current_daily_life", "Current AI Impact", "impact", "Current AI impact on daily life"),
    ("Do you think your job is likely to be automated in the next 10 years?",
     "job_automation_risk", "Work & Automation", "yes_no", "Likelihood of job automation"),
    ("Do you think your job should be automated in the next 10 years?",
     "job_automation_desire", "Work & Automation", "yes_no", "Desire for job automation"),
    ("So far, how has your community been affected by job loss from automation?",
     "community_automation_impact", "Work & Automation", "automation_impact", "Community impact of automation job loss"),
]

# ── Scale type definitions ────────────────────────────────────────────────────

SCALE_CONFIGS = {
    "trust": {
        "positions":      {"Strongly Distrust": 0.0, "Somewhat Distrust": 0.25,
                           "Neither Trust Nor Distrust": 0.5, "Somewhat Trust": 0.75, "Strongly Trust": 1.0},
        "agree_label":    "Trust",
        "disagree_label": "Distrust",
    },
    "benefits_risks": {
        "positions":      {"Risks far outweigh benefits": 0.0, "Risks slightly outweigh benefits": 0.25,
                           "Risks and benefits are equal": 0.5, "Benefits slightly outweigh risks": 0.75,
                           "Benefits far outweigh risks": 1.0},
        "agree_label":    "Benefits outweigh risks",
        "disagree_label": "Risks outweigh benefits",
    },
    "impact": {
        "positions":      {"Profoundly Worse": 0.0, "Noticeably Worse": 0.25,
                           "No Major Change": 0.5, "Noticeably Better": 0.75, "Profoundly Better": 1.0},
        "agree_label":    "Better",
        "disagree_label": "Worse",
    },
    "frequency": {
        "positions":      {"never": 0.0, "annually": 0.25,
                           "monthly": 0.5, "weekly": 0.75, "daily": 1.0},
        "agree_label":    "Frequently (daily/weekly)",
        "disagree_label": "Rarely/Never",
    },
    "excited_concerned": {
        "positions":      {"More concerned than excited": 0.0,
                           "Equally concerned and excited": 0.5,
                           "More excited than concerned": 1.0},
        "agree_label":    "More excited",
        "disagree_label": "More concerned",
    },
    "agree_disagree_unsure": {
        "positions":      {"Disagree": 0.0, "Unsure": 0.5, "Agree": 1.0},
        "agree_label":    "Agree",
        "disagree_label": "Disagree",
    },
    "yes_no": {
        "positions":      {"No": 0.0, "Don't Know": 0.5, "Yes": 1.0},
        "agree_label":    "Yes",
        "disagree_label": "No",
    },
    "automation_impact": {
        "positions":      {"Not at all": 0.0,
                           "I know someone who has lost their job": 0.33,
                           "I know several people who have lost their job": 0.67,
                           "I know many people who have lost their job": 1.0},
        "agree_label":    "Knows someone affected",
        "disagree_label": "Not affected",
    },
}


def load_country_map(preprocess_dir: Path) -> dict:
    path = preprocess_dir / "country_name_map.json"
    if not path.exists():
        print(f"  ❌ country_name_map.json not found at {path}")
        sys.exit(1)
    with open(path) as f:
        return json.load(f)


def load_aggregate(gd_path: Path, round_num: int) -> pd.DataFrame:
    path = gd_path / "Data" / f"GD{round_num}" / f"GD{round_num}_aggregate_standardized.csv"
    if not path.exists():
        print(f"\n❌ Could not find: {path}")
        sys.exit(1)
    df = pd.read_csv(path, low_memory=False)
    print(f"  Loaded aggregate file: {len(df)} rows")
    return df


def load_counts(gd_path: Path, round_num: int) -> pd.DataFrame:
    """Load segment counts — gives us n per country per question."""
    path = gd_path / "Data" / f"GD{round_num}" / f"GD{round_num}_segment_counts_by_question.csv"
    if not path.exists():
        print(f"  ⚠️  Segment counts file not found — n will not be available")
        return None
    df = pd.read_csv(path, low_memory=False)
    df["Question Text"] = df["Question Text"].str.strip()
    print(f"  Loaded segment counts: {len(df)} questions")
    return df


def get_country_columns(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c.startswith("O7:")]


def compute_agree_rate(distribution: dict, scale_type: str) -> float | None:
    """Weighted average position across response options (0=negative, 1=positive)."""
    config = SCALE_CONFIGS.get(scale_type)
    if not config:
        return None
    positions = config["positions"]
    total = sum(distribution.values())
    if total == 0:
        return None
    weighted_sum = sum(
        positions.get(response, 0.5) * proportion
        for response, proportion in distribution.items()
    )
    return round(weighted_sum / total, 4)


def process_question(q_text: str, q_code: str, category: str, scale_type: str, short_label: str,
                     agg_df: pd.DataFrame, country_cols: list,
                     country_map: dict, counts_df: pd.DataFrame = None) -> dict | None:

    config = SCALE_CONFIGS[scale_type]

    # Build n lookup from counts file
    n_lookup = {}
    if counts_df is not None:
        count_row = counts_df[counts_df["Question Text"] == q_text.strip()]
        if not count_row.empty:
            for col in [c for c in counts_df.columns if c.startswith("O7:")]:
                country_name = col.replace("O7: ", "").strip()
                try:
                    n_lookup[country_name] = int(count_row.iloc[0][col])
                except (ValueError, TypeError):
                    pass

    # Get all rows for this question
    q_rows = agg_df[agg_df["Question"].str.strip() == q_text.strip()]
    if q_rows.empty:
        print(f"    ⚠️  No rows found for: {q_text[:60]}...")
        return None

    # Get response options in order
    response_options = q_rows["Response"].dropna().tolist()

    countries_data = {}

    for col in country_cols:
        country_name = col.replace("O7: ", "").strip()
        iso = country_map.get(country_name)
        if not iso:
            continue

        # Build distribution for this country
        distribution = {}
        total_pct = 0.0
        for _, row in q_rows.iterrows():
            response = row.get("Response", "")
            try:
                raw = str(row.get(col, '0') or '0').replace('%', '').strip()
                val = float(raw) if raw else 0.0
            except (TypeError, ValueError):
                val = 0.0
            if response:
                distribution[response] = round(val / 100.0, 4)
                total_pct += val

        # Skip countries with effectively no data
        if total_pct < 10:
            continue

        agree_rate = compute_agree_rate(distribution, scale_type)
        if agree_rate is None:
            continue

        n = n_lookup.get(country_name)
        # Apply min-n filter
        if n is not None and n < 5:
            continue
        countries_data[iso] = {
            "agree_rate":   agree_rate,
            "distribution": distribution,
            "country_name": country_name,
            "n":            n,
        }

    return {
        "id":               q_code,
        "text":             q_text,
        "short_label":      short_label,
        "category_label":   category,
        "scale_name":       scale_type,
        "agree_label":      config["agree_label"],
        "disagree_label":   config["disagree_label"],
        "response_options": response_options,
        "countries":        countries_data,
    }


def build_data(gd_path: Path, round_num: int, output_path: Path,
               preprocess_dir: Path):

    print(f"\n  Round: GD{round_num}")
    country_map  = load_country_map(preprocess_dir)
    agg_df       = load_aggregate(gd_path, round_num)
    counts_df    = load_counts(gd_path, round_num)
    country_cols = get_country_columns(agg_df)
    print(f"  Found {len(country_cols)} country columns")
    print(f"  Processing {len(INDICATOR_QUESTIONS)} indicator questions...\n")

    questions = []
    skipped   = 0

    for (q_text, q_code, category, scale_type, short_label) in INDICATOR_QUESTIONS:
        q_data = process_question(
            q_text, q_code, category, scale_type, short_label,
            agg_df, country_cols, country_map, counts_df
        )
        if q_data and q_data["countries"]:
            questions.append(q_data)
            print(f"    ✓ {q_code} — {len(q_data['countries'])} countries")
        else:
            skipped += 1

    output = {
        "round":       f"GD{round_num}",
        "round_label": f"Global Dialogues Round {round_num}",
        "questions":   questions,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, separators=(",", ":"))

    size_kb = output_path.stat().st_size // 1024
    print(f"\n  ✓ Wrote {output_path} ({size_kb} KB)")
    print(f"  ✓ {len(questions)} questions, {skipped} skipped")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gd-path", required=True, type=Path)
    parser.add_argument("--round",   required=True, type=int)
    parser.add_argument("--output",  required=True, type=Path)
    parser.add_argument("--min-n",   default=5, type=int)
    args = parser.parse_args()

    preprocess_dir = Path(__file__).parent
    build_data(args.gd_path, args.round, args.output, preprocess_dir)