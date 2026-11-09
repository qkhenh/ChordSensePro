"""Re-map labels of already-stored training samples after a HarteMapper change — no re-crawl needed.

Reads the distinct ``label_raw`` strings, runs ``map_harte`` again and updates
``label_status / root_id / quality_id / ext_id`` where the result differs.

    docker exec chordsense_airflow python /opt/airflow/scripts/remap_labels.py --dry-run
    docker exec chordsense_airflow python /opt/airflow/scripts/remap_labels.py
"""
from __future__ import annotations
import argparse
import sys

sys.path.insert(0, "/opt/airflow")

from src.data_processing.application.mappers.harte_mapper import map_harte   # noqa: E402
from src.shared.infrastructure.postgres.client import get_connection         # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    changed_rows = 0
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT label_raw, label_status, root_id, quality_id, ext_id, COUNT(*) "
                    "FROM training_samples GROUP BY 1,2,3,4,5")
        for raw, status, root, quality, ext, n in cur.fetchall():
            m = map_harte(raw)
            new = (m.status.value, m.root_id, m.quality_id, m.ext_id)
            if new == (status, root, quality, ext):
                continue
            print(f"{raw!r:<24} {status}/{root},{quality},{ext}  ->  {new[0]}/{new[1]},{new[2]},{new[3]}   ({n} rows)")
            changed_rows += n
            if not args.dry_run:
                cur.execute(
                    "UPDATE training_samples SET label_status=%s, root_id=%s, quality_id=%s, ext_id=%s "
                    "WHERE label_raw=%s AND label_status=%s "
                    "AND root_id IS NOT DISTINCT FROM %s AND quality_id IS NOT DISTINCT FROM %s "
                    "AND ext_id IS NOT DISTINCT FROM %s",
                    (*new, raw, status, root, quality, ext),
                )
    print(f"\n{changed_rows} rows {'would change' if args.dry_run else 'updated'}.")


if __name__ == "__main__":
    main()
