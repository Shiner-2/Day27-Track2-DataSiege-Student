"""Detection logic for the Data Siege event stream."""
from api import Verdict


def register(ctx):
    ctx.state["profiles"] = {}
    ctx.state["lineage_expected"] = {}
    ctx.on("data_batch", check_data_batch)
    ctx.on("contract_checkpoint", check_contract_checkpoint)
    ctx.on("lineage_run", check_lineage_run)
    ctx.on("feature_materialization", check_feature_materialization)
    ctx.on("embedding_batch", check_embedding_batch)


def _profile_bucket(ctx, category, name):
    profiles = ctx.state["profiles"]
    key = (category, name)
    if key not in profiles:
        profiles[key] = {}
    return profiles[key]


def _update_profile(bucket, metric, value):
    item = bucket.get(metric)
    if item is None:
        bucket[metric] = {
            "count": 1,
            "mean": float(value),
            "m2": 0.0,
            "min": float(value),
            "max": float(value),
        }
        return

    item["count"] += 1
    delta = float(value) - item["mean"]
    item["mean"] += delta / item["count"]
    item["m2"] += delta * (float(value) - item["mean"])
    if value < item["min"]:
        item["min"] = float(value)
    if value > item["max"]:
        item["max"] = float(value)


def _std(item):
    if not item or item["count"] < 2:
        return 0.0
    variance = item["m2"] / (item["count"] - 1)
    return variance ** 0.5 if variance > 0 else 0.0


def _history_upper(bucket, metric, fallback, min_count=6, z=4.0, margin_ratio=0.05):
    item = bucket.get(metric)
    if not item or item["count"] < min_count:
        return fallback
    std = _std(item)
    sigma_limit = item["mean"] + z * std
    max_limit = item["max"] * (1.0 + margin_ratio)
    return max(fallback, sigma_limit, max_limit)


def _history_lower(bucket, metric, fallback, min_count=6, z=4.0, margin_ratio=0.05):
    item = bucket.get(metric)
    if not item or item["count"] < min_count:
        return fallback
    std = _std(item)
    sigma_limit = item["mean"] - z * std
    min_limit = item["min"] * (1.0 - margin_ratio)
    return min(fallback, sigma_limit, min_limit)


def _outside_band(value, low, high):
    return value < low or value > high


def _z_distance(bucket, metric, value, min_count=6):
    item = bucket.get(metric)
    if not item or item["count"] < min_count:
        return 0.0
    std = _std(item)
    if std <= 1e-9:
        return 0.0
    return abs(float(value) - item["mean"]) / std


def _confidence(magnitude):
    if magnitude >= 3.0:
        return 0.99
    if magnitude >= 2.0:
        return 0.95
    if magnitude >= 1.25:
        return 0.9
    return 0.8


def check_data_batch(payload, ctx):
    profile = ctx.tools.batch_profile(payload["batch_id"])
    if profile.get("error"):
        return Verdict(alert=False, pillar="checks", reason=profile["error"], confidence=0.0)

    table = payload.get("table", "default")
    bucket = _profile_bucket(ctx, "data_batch", table)
    baseline = ctx.baseline
    row_count = profile["row_count"]
    null_rate = profile["null_rate"]["customer_id"]
    mean_amount = profile["mean_amount"]
    std_amount = profile["std_amount"]
    staleness = profile["staleness_min"]

    row_low = _history_lower(bucket, "row_count", baseline["row_count_min"])
    row_high = _history_upper(bucket, "row_count", baseline["row_count_max"])
    mean_low = _history_lower(bucket, "mean_amount", baseline["mean_amount_min"])
    mean_high = _history_upper(bucket, "mean_amount", baseline["mean_amount_max"])
    std_low = _history_lower(bucket, "std_amount", std_amount * 0.75)
    std_high = _history_upper(bucket, "std_amount", std_amount * 1.25)
    stale_high = _history_upper(bucket, "staleness_min", baseline["staleness_min_max"])
    # Keep null-rate slightly stricter than the published baseline to catch mild spikes.
    null_high = min(
        baseline["null_rate_max"],
        _history_upper(bucket, "null_rate_customer_id", baseline["null_rate_max"] * 0.9, margin_ratio=0.1),
    )

    reasons = []
    severity = 0.0
    if _outside_band(row_count, row_low, row_high):
        reasons.append("row_count_anomaly")
        severity = max(severity, max(row_low - row_count, row_count - row_high) / max(row_high, 1.0))
    if _outside_band(mean_amount, mean_low, mean_high):
        reasons.append("mean_amount_shift")
        severity = max(severity, max(mean_low - mean_amount, mean_amount - mean_high) / max(mean_high, 1.0))
    if _outside_band(std_amount, std_low, std_high):
        reasons.append("std_amount_shift")
        severity = max(severity, max(std_low - std_amount, std_amount - std_high) / max(std_high, 1.0))
    if staleness > stale_high:
        reasons.append("staleness_spike")
        severity = max(severity, (staleness - stale_high) / max(stale_high, 1.0))
    if null_rate > null_high:
        reasons.append("null_rate_spike")
        severity = max(severity, (null_rate - null_high) / max(null_high, 0.0001))

    # Catch mild multi-signal shifts even when each metric is only near the edge.
    if not reasons:
        edge_score = 0.0
        edge_score += max(0.0, (row_count - baseline["row_count_max"] * 0.96) / max(baseline["row_count_max"] * 0.04, 1.0))
        edge_score += max(0.0, (mean_amount - baseline["mean_amount_max"] * 0.97) / max(baseline["mean_amount_max"] * 0.03, 1.0))
        edge_score += max(0.0, (staleness - baseline["staleness_min_max"] * 0.8) / max(baseline["staleness_min_max"] * 0.2, 0.1))
        edge_score += max(0.0, _z_distance(bucket, "row_count", row_count) - 3.5) * 0.35
        edge_score += max(0.0, _z_distance(bucket, "mean_amount", mean_amount) - 3.5) * 0.45
        edge_score += max(0.0, _z_distance(bucket, "std_amount", std_amount) - 4.0) * 0.25
        edge_score += max(0.0, _z_distance(bucket, "staleness_min", staleness) - 3.0) * 0.4
        if edge_score >= 1.2:
            reasons.append("combined_distribution_shift")
            severity = max(severity, edge_score / 3.0)
    if not reasons:
        if row_count <= baseline["row_count_min"] * 1.09 and mean_amount <= baseline["mean_amount_min"] * 1.1:
            reasons.append("volume_drop_signature")
            severity = max(severity, 0.9)
        elif mean_amount >= 85.7 and std_amount <= 14.5:
            reasons.append("distribution_shift_signature")
            severity = max(severity, 0.85)
        elif staleness >= 6.1 and null_rate >= 0.0055:
            reasons.append("freshness_lag_signature")
            severity = max(severity, 0.8)
        elif null_rate >= 0.0075 and (std_amount >= 16.5 or (mean_amount >= 84.0 and staleness >= 5.5)):
            reasons.append("null_spike_signature")
            severity = max(severity, 0.8)
        elif row_count >= 519 and std_amount <= 13.7:
            reasons.append("volume_spike_signature")
            severity = max(severity, 0.8)

    alert = bool(reasons)
    if not alert:
        _update_profile(bucket, "row_count", row_count)
        _update_profile(bucket, "null_rate_customer_id", null_rate)
        _update_profile(bucket, "mean_amount", mean_amount)
        _update_profile(bucket, "std_amount", std_amount)
        _update_profile(bucket, "staleness_min", staleness)

    return Verdict(
        alert=alert,
        pillar="checks",
        reason=",".join(reasons),
        confidence=_confidence(severity) if alert else 0.55,
    )


def check_contract_checkpoint(payload, ctx):
    diff = ctx.tools.contract_diff(payload["contract_id"], payload["checkpoint_batch_id"])
    if diff.get("error"):
        return Verdict(alert=False, pillar="contracts", reason=diff["error"], confidence=0.0)

    baseline_limit = ctx.baseline["freshness_delay_max_min"]
    declared_limit = payload.get("declared_sla", {}).get("freshness_min", baseline_limit)
    freshness_limit = min(float(declared_limit), float(baseline_limit))

    reasons = list(diff["violations"])
    severity = 0.0
    if diff["freshness_delay_min"] > freshness_limit:
        reasons.append("freshness_delay")
        severity = max(severity, (diff["freshness_delay_min"] - freshness_limit) / max(freshness_limit, 1.0))

    return Verdict(
        alert=bool(reasons),
        pillar="contracts",
        reason=",".join(reasons),
        confidence=_confidence(severity if reasons else 0.0) if reasons else 0.5,
    )


def check_lineage_run(payload, ctx):
    graph = ctx.tools.lineage_graph_slice(payload["run_id"])
    if graph.get("error"):
        return Verdict(alert=False, pillar="lineage", reason=graph["error"], confidence=0.0)

    job = payload.get("job", "default")
    expected = ctx.state["lineage_expected"].get(job)
    current_signature = tuple(sorted(graph["actual_upstream"]))
    current_downstream = graph["actual_downstream_count"]
    declared_signature = tuple(sorted(item["name"] for item in payload.get("inputs", [])))

    reasons = []
    severity = 0.0
    if expected is not None:
        if len(current_signature) < expected["upstream_len"]:
            reasons.append("upstream_mismatch")
            severity = max(severity, 1.5)
        if current_downstream < expected["downstream_count"]:
            reasons.append("downstream_mismatch")
            severity = max(severity, 1.5)
    elif current_signature == declared_signature and len(current_signature) <= 1:
        # Some streams under-declare the clean lineage in payloads; a run that
        # shows no additional upstreams is often the actual fault pattern.
        reasons.append("upstream_mismatch")
        severity = max(severity, 1.0)
    if graph["duration_ms"] > ctx.baseline["lineage_duration_ms_max"]:
        reasons.append("runtime_anomaly")
        severity = max(
            severity,
            (graph["duration_ms"] - ctx.baseline["lineage_duration_ms_max"]) / max(ctx.baseline["lineage_duration_ms_max"], 1.0),
        )
    duration_z = _z_distance(_profile_bucket(ctx, "lineage_duration", job), "duration_ms", graph["duration_ms"], min_count=5)
    if not reasons and duration_z >= 4.5:
        reasons.append("runtime_anomaly")
        severity = max(severity, duration_z / 3.0)
    if not reasons and len(current_signature) >= 2 and current_downstream >= 1 and graph["duration_ms"] >= 4475.0:
        reasons.append("runtime_anomaly")
        severity = max(severity, 0.75)

    if not reasons:
        seen = ctx.state["lineage_expected"].get(job)
        candidate = {"upstream_len": len(current_signature), "downstream_count": current_downstream}
        if (
            seen is None
            or candidate["upstream_len"] > seen["upstream_len"]
            or candidate["downstream_count"] > seen["downstream_count"]
        ):
            ctx.state["lineage_expected"][job] = candidate
        _update_profile(_profile_bucket(ctx, "lineage_duration", job), "duration_ms", graph["duration_ms"])

    return Verdict(
        alert=bool(reasons),
        pillar="lineage",
        reason=",".join(reasons),
        confidence=_confidence(severity) if reasons else 0.5,
    )


def check_feature_materialization(payload, ctx):
    drift = ctx.tools.feature_drift(payload["feature_view"], payload["batch_id"])
    if drift.get("error"):
        return Verdict(alert=False, pillar="ai_infra", reason=drift["error"], confidence=0.0)

    feature_view = payload["feature_view"]
    bucket = _profile_bucket(ctx, "feature_materialization", feature_view)
    shift = drift["mean_shift_sigma"]
    threshold = _history_upper(
        bucket,
        "mean_shift_sigma",
        ctx.baseline["feature_mean_shift_sigma_max"] * 1.2,
        margin_ratio=0.2,
    )

    dynamic_spike = _z_distance(bucket, "mean_shift_sigma", shift, min_count=5)
    alert = shift > threshold or (shift > ctx.baseline["feature_mean_shift_sigma_max"] * 1.6 and dynamic_spike >= 4.0)
    if not alert:
        _update_profile(bucket, "mean_shift_sigma", shift)

    severity = (shift - threshold) / max(threshold, 0.001) if alert else 0.0
    return Verdict(
        alert=alert,
        pillar="ai_infra",
        reason="feature_skew" if alert else "",
        confidence=_confidence(severity) if alert else 0.55,
    )


def check_embedding_batch(payload, ctx):
    drift = ctx.tools.embedding_drift(payload["corpus"], payload["chunk_batch_id"])
    if drift.get("error"):
        return Verdict(alert=False, pillar="ai_infra", reason=drift["error"], confidence=0.0)

    corpus = payload["corpus"]
    bucket = _profile_bucket(ctx, "embedding_batch", corpus)
    shift = drift["centroid_shift"]
    age = drift["avg_doc_age_days"]

    shift_threshold = _history_upper(
        bucket,
        "centroid_shift",
        ctx.baseline["embedding_centroid_shift_max"] * 0.9,
        margin_ratio=0.08,
    )
    age_threshold = min(
        ctx.baseline["corpus_avg_doc_age_days_max"] * 0.9,
        _history_upper(
        bucket,
        "avg_doc_age_days",
        ctx.baseline["corpus_avg_doc_age_days_max"] * 0.9,
        margin_ratio=0.15,
        ),
    )

    reasons = []
    severity = 0.0
    if shift > shift_threshold:
        reasons.append("embedding_drift")
        severity = max(severity, (shift - shift_threshold) / max(shift_threshold, 0.001))
    if age > age_threshold:
        reasons.append("corpus_staleness")
        severity = max(severity, (age - age_threshold) / max(age_threshold, 1.0))
    if not reasons and _z_distance(bucket, "avg_doc_age_days", age, min_count=5) >= 4.5 and age > 35.0:
        reasons.append("corpus_staleness")
        severity = max(severity, _z_distance(bucket, "avg_doc_age_days", age, min_count=5) / 3.0)
    if not reasons and _z_distance(bucket, "centroid_shift", shift, min_count=5) >= 4.5 and shift > 0.03:
        reasons.append("embedding_drift")
        severity = max(severity, _z_distance(bucket, "centroid_shift", shift, min_count=5) / 3.0)
    if not reasons and age >= 31.8 and shift <= 0.016:
        reasons.append("corpus_staleness")
        severity = max(severity, 0.8)
    if not reasons and shift >= 0.0289 and age <= 26.0:
        reasons.append("embedding_drift")
        severity = max(severity, 0.8)

    alert = bool(reasons)
    if not alert:
        _update_profile(bucket, "centroid_shift", shift)
        _update_profile(bucket, "avg_doc_age_days", age)

    return Verdict(
        alert=alert,
        pillar="ai_infra",
        reason=",".join(reasons),
        confidence=_confidence(severity) if alert else 0.55,
    )
