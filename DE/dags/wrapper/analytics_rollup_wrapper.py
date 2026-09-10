"""Wrapper for dag_analytics_rollup — python_callables for PythonOperator.

DAG file only contains Airflow wiring (DAG, tasks, dependencies).
All callable logic lives here, bridging Airflow context → src/.
"""
from __future__ import annotations
from datetime import date

from src.user_analytics.domain.models.chord_mastery import ChordMastery
from src.user_analytics.infrastructure.repositories.chord_attempt_repository import ChordAttemptRepository
from src.user_analytics.infrastructure.repositories.chord_mastery_repository import ChordMasteryRepository


def aggregate_attempts(**context) -> None:
    """Compute accuracy per (user_id, chord, date) from today's chord_attempts."""
    today = date.today()

    repo = ChordAttemptRepository()
    all_attempts = repo.list_by_date(today)

    groups: dict[tuple[str, str], list] = {}
    for attempt in all_attempts:
        key = (attempt.user_id, attempt.target_chord)
        groups.setdefault(key, []).append(attempt)

    accuracies = []
    for (user_id, chord), attempts in groups.items():
        correct  = sum(1 for a in attempts if a.is_correct)
        accuracy = correct / len(attempts) if attempts else 0.0
        accuracies.append({
            "user_id":       user_id,
            "chord":         chord,
            "accuracy_today": accuracy,
            "attempt_count": len(attempts),
        })

    context["ti"].xcom_push(key="accuracies", value=accuracies)


def rolling_accuracy(**context) -> None:
    """Compute 3-day rolling accuracy using today + last 2 days from mastery table."""
    ti         = context["ti"]
    accuracies = ti.xcom_pull(key="accuracies") or []
    today      = date.today()

    mastery_repo = ChordMasteryRepository()
    rolled = []

    unique_users = {row["user_id"] for row in accuracies}
    user_history: dict[str, list] = {}
    for uid in unique_users:
        user_history[uid] = mastery_repo.get_by_user(uid)

    for row in accuracies:
        user_id = row["user_id"]
        chord   = row["chord"]

        past = [
            r for r in user_history[user_id]
            if r.chord == chord and r.date < today
        ]
        past_sorted = sorted(past, key=lambda r: r.date, reverse=True)[:2]

        weights = [row["accuracy_today"]] + [r.accuracy_today for r in past_sorted]
        rolling = sum(weights) / len(weights) if weights else 0.0
        rolled.append({**row, "rolling_accuracy_3d": rolling})

    ti.xcom_push(key="rolled", value=rolled)


def update_mastery(**context) -> None:
    """Upsert user_chord_mastery rows with new rolling accuracy."""
    ti     = context["ti"]
    rolled = ti.xcom_pull(key="rolled") or []
    today  = date.today()

    repo = ChordMasteryRepository()
    for row in rolled:
        entity = ChordMastery(
            user_id=row["user_id"],
            chord=row["chord"],
            date=today,
            accuracy_today=row["accuracy_today"],
            rolling_accuracy_3d=row["rolling_accuracy_3d"],
        )
        entity.recompute_mastery()
        repo.upsert(entity)
