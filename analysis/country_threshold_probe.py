"""Freeze per-country thresholds on development and test on held-out validation."""

import json
import sys
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "business_entity_resolution" / "src"))
from number_features import NUMBER_FEATURE_NAMES, number_pair_features
from validation import f05_for_query
from calibrate_on_development import sampled_truth as development_truth
from evaluate_first_matcher import sampled_truth as validation_truth

MODEL = ROOT / "code" / "business_entity_resolution" / "number_model.joblib"


def load_and_score(filename, truth_function, model):
    frame = pd.read_parquet(ROOT / "analysis" / filename).fillna("")
    feature_matrix = np.empty((len(frame), len(NUMBER_FEATURE_NAMES)), dtype=np.float32)
    for i, row in enumerate(frame.itertuples(index=False)):
        feature_matrix[i] = number_pair_features(
            row.q_name, row.t_name, row.q_address, row.t_address, row.target_source
        )
    frame["score"] = model.predict_proba(feature_matrix)[:, 1]
    return frame, truth_function()


def country_query_ids(frame):
    return {country: set(group.s1_id) for country, group in frame.groupby("country")}


def evaluate(frame, truth, thresholds):
    selected = frame.score.to_numpy() >= np.array([
        thresholds[country] for country in frame.country
    ])
    predicted = defaultdict(set)
    for s1, target in frame.loc[selected, ["s1_id", "target_id"]].itertuples(index=False, name=None):
        predicted[s1].add(target)
    result = {"macro_f05": float(np.mean([
        f05_for_query(actual, predicted[s1]) for s1, actual in truth.items()
    ]))}
    by_country = country_query_ids(frame)
    result["countries"] = {country: float(np.mean([
        f05_for_query(truth[s1], predicted[s1]) for s1 in ids
    ])) for country, ids in by_country.items()}
    result["pairs"] = int(selected.sum())
    return result


def main():
    model = joblib.load(MODEL)
    assert list(model.feature_name_) == NUMBER_FEATURE_NAMES
    development, dev_truth = load_and_score(
        "full_development_combined_pairs.parquet", development_truth, model
    )
    validation, val_truth = load_and_score(
        "full_validation_combined_pairs.parquet", validation_truth, model
    )
    baseline = {country: 0.75 for country in development.country.unique()}
    grid = np.arange(0.5, 0.901, 0.025)
    chosen = baseline.copy()
    curves = {}
    for country in chosen:
        curve = []
        for threshold in grid:
            settings = baseline | {country: float(threshold)}
            measured = evaluate(development, dev_truth, settings)
            curve.append({"threshold": round(float(threshold), 3),
                          "country_f05": measured["countries"][country]})
        best = max(curve, key=lambda row: row["country_f05"])
        chosen[country] = best["threshold"]
        curves[country] = curve
    result = {
        "selected_on_development": chosen,
        "development_global_075": evaluate(development, dev_truth, baseline),
        "development_by_country": evaluate(development, dev_truth, chosen),
        "validation_global_075": evaluate(validation, val_truth, baseline),
        "validation_by_country": evaluate(validation, val_truth, chosen),
        "development_curves": curves,
    }
    path = ROOT / "analysis" / "country_threshold_probe_results.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items()
                      if key != "development_curves"}, indent=2))


if __name__ == "__main__":
    main()
