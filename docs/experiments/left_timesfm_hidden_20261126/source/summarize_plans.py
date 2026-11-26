"""CPU-only exposure accounting from the sealed source-plan bytes."""

import gzip
import io
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent


def summarize(rows, identities):
    groups = []
    for group_id, records in enumerate(identities):
        x = rows[rows[:, 0] == group_id]
        pairs, counts = np.unique(x[:, :3], axis=0, return_counts=True)
        unique_input = len(np.unique(x, axis=0))
        record_rows = []
        for record_index in np.unique(x[:, 1]):
            q = x[x[:, 1] == record_index]
            ends, repetitions = np.unique(q[:, 2], return_counts=True)
            record_rows.append(
                dict(
                    record_id=records[int(record_index)]["record_id"],
                    window_presentations=len(q),
                    distinct_current_starts=len(ends),
                    distinct_inputs_including_ordered_channels=len(np.unique(q, axis=0)),
                    max_presentations_per_start=int(repetitions.max()),
                )
            )
        groups.append(
            dict(
                source_domain=records[0]["record_id"].split("/")[0],
                window_presentations=len(x),
                distinct_records=len(record_rows),
                distinct_record_current_starts=len(pairs),
                repeat_presentations_ignoring_channels=len(x) - len(pairs),
                distinct_inputs_including_ordered_channels=unique_input,
                presentations_per_record_start_quantiles=np.quantile(
                    counts, [0, 0.25, 0.5, 0.75, 1]
                ).tolist(),
                channel_window_presentations=int(len(x) * (x.shape[1] - 4)),
                current_value_presentations=int(len(x) * 192 * (x.shape[1] - 4)),
                by_record=record_rows,
            )
        )
    return groups


if __name__ == "__main__":
    output = {}
    for fold in ("PSM", "SWAT", "SMD"):
        identities = json.loads((ROOT / fold / "SOURCE.json").read_text())
        data = gzip.decompress((ROOT / fold / "PLANS.npz.gz").read_bytes())
        with np.load(io.BytesIO(data)) as plan:
            output[fold] = {
                role: summarize(plan[role], identities[role])
                for role in ("fit", "select")
            }
            for role in ("fit", "select"):
                current, donor = plan[role], plan[f"{role}_donor"]
                shift = current[:, 2] - donor[:, 2]
                output[fold][f"{role}_history_reassignment"] = dict(
                    presentations=len(current),
                    identity_presentations=int(np.count_nonzero(shift == 0)),
                    shift_quantiles=np.quantile(shift, [0, 0.25, 0.5, 0.75, 1]).tolist(),
                    preserved_record_role_channel_mapping=bool(
                        np.array_equal(current[:, :2], donor[:, :2])
                        and np.array_equal(current[:, 3:], donor[:, 3:])
                    ),
                    future_history_presentations=int(np.count_nonzero(shift < 0)),
                )
    (ROOT / "plan_statistics.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8"
    )
    for fold, roles in output.items():
        for row in roles["fit"]:
            print(fold, row["source_domain"], row["window_presentations"], row["distinct_records"], row["distinct_record_current_starts"], row["distinct_inputs_including_ordered_channels"])
