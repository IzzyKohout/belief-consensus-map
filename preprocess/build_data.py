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
     "ai_feeling_excited_concerned", "AI Perception", "excited_concerned"),
    ("Thinking about the last three months, how often, if at all, have you noticed AI systems in your daily life?",
     "ai_notice_frequency", "AI Exposure", "frequency"),
    ("Thinking about the last three months, how often, if at all, have you noticed human interactions which have been replaced with automated systems?",
     "automation_notice_frequency", "AI Exposure", "frequency"),
    ("Thinking about the last three months, how often, if at all, have you been expected to use an AI system at work?",
     "ai_use_work_expected_freq", "AI Use", "frequency"),
    ("Thinking about the last three months, how often, if at all, have you personally chosen to use an AI system at work?",
     "ai_use_work_chosen_freq", "AI Use", "frequency"),
    ("Thinking about the last three months, how often, if at all, have you personally chosen to use an AI system in your personal life?",
     "ai_use_personal_chosen_freq", "AI Use", "frequency"),
    ("Thinking about the last three months, how often, if at all, have you interacted with AI systems to get advice on a sensitive personal issue or to get emotional support?",
     "ai_use_sensitive_advice_freq", "AI Use", "frequency"),
    ("Thinking about the last three months, how often, if at all, have you interacted with AI systems to complete an action in the real world on your behalf without your supervision?",
     "ai_use_real_world_action_freq", "AI Use", "frequency"),
    ("Considering both potential benefits and risks, how do you assess the overall impact on society of messaging apps?",
     "impact_society_msg_apps", "Societal Impact Assessment", "benefits_risks"),
    ("Considering both potential benefits and risks, how do you assess the overall impact on society of social media apps?",
     "impact_society_social_media", "Societal Impact Assessment", "benefits_risks"),
    ("Considering both potential benefits and risks, how do you assess the overall impact on society of AI chatbots?",
     "impact_society_ai_chatbots", "Societal Impact Assessment", "benefits_risks"),
    ("Considering both potential benefits and risks, how do you assess the overall impact on society of AI systems that can perform tasks in the real world without human supervision?",
     "impact_society_ai_unsupervised", "Societal Impact Assessment", "benefits_risks"),
    ("Considering both potential benefits and risks, how do you assess the overall impact on society of AI systems that can outperform humans on most economically valuable work?",
     "impact_society_ai_outperform", "Societal Impact Assessment", "benefits_risks"),
    ("To what extent, if at all, do you generally trust governments to do what is right?",
     "trust_govt", "Trust in Institutions", "trust"),
    ("To what extent, if at all, do you generally trust small businesses to do what is right?",
     "trust_smb", "Trust in Institutions", "trust"),
    ("To what extent, if at all, do you generally trust large corporations to do what is right?",
     "trust_large_corp", "Trust in Institutions", "trust"),
    ("To what extent, if at all, do you generally trust social media companies to do what is right?",
     "trust_social_media_co", "Trust in Institutions", "trust"),
    ("To what extent, if at all, do you generally trust companies building AI to do what is right?",
     "trust_ai_co", "Trust in Institutions", "trust"),
    ("To what extent, if at all, do you generally trust public utility companies to do what is right?",
     "trust_utility_co", "Trust in Institutions", "trust"),
    ("To what extent, if at all, do you generally trust public research institutions to do what is right?",
     "trust_research_inst", "Trust in Institutions", "trust"),
    ("To what extent, if at all, do you trust your family doctor to act in your best interest?",
     "trust_personal_doctor", "Trust in Specific Actors", "trust"),
    ("To what extent, if at all, do you trust your social media feed (eg TikTok, Facebook) to act in your best interest?",
     "trust_personal_sm_feed", "Trust in Specific Actors", "trust"),
    ("To what extent, if at all, do you trust your elected representatives to act in your best interest?",
     "trust_personal_elected_reps", "Trust in Specific Actors", "trust"),
    ("To what extent, if at all, do you trust your faith or community leader to act in your best interest?",
     "trust_personal_faith_leader", "Trust in Specific Actors", "trust"),
    ("To what extent, if at all, do you trust the civil servants in your government to act in your best interest?",
     "trust_personal_civil_servants", "Trust in Specific Actors", "trust"),
    ("To what extent, if at all, do you trust your AI chatbot (eg ChatGPT) to act in your best interest?",
     "trust_personal_ai_chatbot", "Trust in Specific Actors", "trust"),
    ("Do you agree or disagree with this statement? AI could make better decisions on my behalf than my government representatives.",
     "ai_vs_govt_decisions", "AI Governance", "agree_disagree_unsure"),
    ("Do you think the increased use of AI across society is likely to make your cost of living better, worse or stay the same in the next 10 years?",
     "ai_impact_future_cost_living", "Future Impact — Personal", "impact"),
    ("Do you think the increased use of AI across society is likely to make the amount of free time you have better, worse or stay the same in the next 10 years?",
     "ai_impact_future_free_time", "Future Impact — Personal", "impact"),
    ("Do you think the increased use of AI across society is likely to make your community's well-being better, worse or stay the same in the next 10 years?",
     "ai_impact_future_community", "Future Impact — Societal", "impact"),
    ("Do you think the increased use of AI across society is likely to make the availability of good jobs better, worse or stay the same in the next 10 years?",
     "ai_impact_future_jobs", "Future Impact — Societal", "impact"),
    ("Do you think the increased use of AI across society is likely to make your sense of purpose better, worse or stay the same in the next 10 years?",
     "ai_impact_future_purpose", "Future Impact — Personal", "impact"),
    ("So far, what has been the overall impact of AI on your daily life?",
     "ai_impact_current_daily_life", "Current AI Impact", "impact"),
    ("Is your job making a meaningful contribution to the world?",
     "job_meaningful", "Work & Automation", "yes_no"),
    ("Do you think your job is likely to be automated in the next 10 years?",
     "job_automation_risk", "Work & Automation", "yes_no"),
    ("Do you think your job should be automated in the next 10 years?",
     "job_automation_desire", "Work & Automation", "yes_no"),
    ("So far, how has your community been affected by job loss from automation?",
     "community_automation_impact", "Work & Automation", "automation_impact"),
]

# ── Scale type definitions ────────────────────────────────────────────────────

SCALE_CONFIGS = {
    "trust": {
        "positive":       ["Somewhat Trust", "Strongly Trust"],
        "agree_label":    "Trust",
        "disagree_label": "Distrust",
    },
    "benefits_risks": {
        "positive":       ["Benefits slightly outweigh risks", "Benefits far outweigh risks"],
        "agree_label":    "Benefits outweigh risks",
        "disagree_label": "Risks outweigh benefits",
    },
    "impact": {
        "positive":       ["Noticeably Better", "Profoundly Better"],
        "agree_label":    "Better",
        "disagree_label": "Worse",
    },
    "frequency": {
        "positive":       ["daily", "weekly"],
        "agree_label":    "Frequently (daily/weekly)",
        "disagree_label": "Rarely/Never",
    },
    "excited_concerned": {
        "positive":       ["More excited than concerned"],
        "agree_label":    "More excited",
        "disagree_label": "More concerned",
    },
    "agree_disagree_unsure": {
        "positive":       ["Agree"],
        "agree_label":    "Agree",
        "disagree_label": "Disagree",
    },
    "yes_no": {
        "positive":       ["Yes"],
        "agree_label":    "Yes",
        "disagree_label": "No",
    },
    "automation_impact": {
        "positive":       [
            "I know someone who has lost their job",
            "I know several people who have lost their job",
            "I know many people who have lost their job",
        ],
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
    config = SCALE_CONFIGS.get(scale_type)
    if not config:
        return None
    total = sum(distribution.values())
    if total == 0:
        return None
    positive_sum = sum(distribution.get(opt, 0) for opt in config["positive"])
    return round(positive_sum / total, 4)


def process_question(q_text: str, q_code: str, category: str, scale_type: str,
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

    for (q_text, q_code, category, scale_type) in INDICATOR_QUESTIONS:
        q_data = process_question(
            q_text, q_code, category, scale_type,
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