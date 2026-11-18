import numpy as np

from hybrid_tsad.left_exposure_evaluation import numerical_keys
from hybrid_tsad.left_exposure_repeats_evaluation import fixed_summary, contrasts_of


def test_fixed_three_seed_summary_uses_sample_sd_and_no_pooled_points():
    rows=[dict(domain="D",method="R",condition="B",stream="strict",seed=s,AP=x)
          for s,x in zip([2025,2026,2027],[.2,.4,.9])]
    out=fixed_summary(rows,numerical_keys,["domain","method","condition","stream"])
    assert len(out)==1 and out[0]["n"]==3
    assert np.isclose(out[0]["mean"],.5)
    assert np.isclose(out[0]["sample_sd"],np.std([.2,.4,.9],ddof=1))


def test_fixed_reference_and_undefined_seed_never_get_fake_sd():
    reference=[dict(method="noU",seed=None,AP=.5)]
    row=fixed_summary(reference,numerical_keys,["method"])[0]
    assert row["n"]==1 and row["sample_sd"]is None
    assert row["note"]=="FIXED_REFERENCE_NO_SEED_SD"
    rows=[dict(method="R",seed=s,AP=x)for s,x in zip([2025,2026,2027],[.1,None,.3])]
    row=fixed_summary(rows,numerical_keys,["method"])[0]
    assert row["mean"]is None and row["sample_sd"]is None and row["n_valid"]==2


def test_all_contrasts_are_same_seed_differences():
    rows=[]
    for domain in ["SWAT","PSM"]:
        for seed in [2025,2026,2027]:
            for stream in ["strict","confirmed3"]:
                for method,values in [("TT",[.7,.4,.6]),("R",[.8,.75,.79])]:
                    for condition,value in zip(["A","B","C"],values):
                        rows.append(dict(domain=domain,seed=seed,stream=stream,method=method,
                                         condition=condition,AP=value+(seed-2025)/100))
    paired,interaction=contrasts_of(rows)
    assert len(paired)==84 and len(interaction)==12
    assert all(np.isclose(row["AP"],.16)for row in interaction)
    assert all(np.isclose(row["AP"],.35)for row in paired if row["contrast"]=="RB−TTB")
