from pathlib import Path
import runpy

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def entry():
    return runpy.run_path(str(ROOT / "scripts/run_m2.py"))


def test_threshold_feedback_counts_are_independent_of_models():
    module=entry()
    data=module["load_datasets"]()
    protocol=yaml.safe_load((ROOT/"protocol_v1.yaml").read_text())
    table=module["threshold_diagnostics"](data[("V_raw","full")]["sigma"], protocol)
    assert len(table)==40
    assert (table.true_0+table.true_1 == 33).all()
    main=table.query('threshold_name == "T_main"')
    assert set(main.seed)=={0,1,2,3,4}
    stratified=main.query('split_kind == "stratified"')
    assert set(stratified.true_1).issubset({16,17})
    assert not stratified.matches_paper.any()
    from llzo_pdp.models import split_ids
    sigma=data[("V_raw","full")]["sigma"]
    y=(sigma>=protocol["target"]["threshold_main"]).astype(int)
    for row in main.itertuples():
        ids=split_ids(y,row.seed,stratified=row.split_kind=="stratified")["test"]
        assert row.true_1==int(y.loc[ids].sum())


def test_job_plan_covers_all_variants_and_disjoint_group_folds():
    module=entry()
    data=module["load_datasets"]()
    protocol=yaml.safe_load((ROOT/"protocol_v1.yaml").read_text())
    jobs=module["build_jobs"](data,protocol)
    regular=[j for j in jobs if j["method"]=="holdout"]
    assert len(regular)==80+10+4
    assert all(j["imputation"]=="A0" for j in jobs)
    for name in ("stratified_cv","group_kfold","loso"):
        subset=[j for j in jobs if j["method"]==name and j["model"]=="DT"]
        test_ids=[]
        for job in subset:
            split=job["splits"]
            assert not set(split["train"]) & set(split["test"])
            test_ids.extend(split["test"])
            if name!="stratified_cv":
                sources=data[("V_raw","full")]["sources"]
                assert not set(sources.loc[split["train"]]) & set(sources.loc[split["test"]])
        assert len(test_ids)==len(set(test_ids))==218


def test_job_roundtrip_resume_and_tamper_rejection(tmp_path):
    module=entry()
    data=module["load_datasets"]()
    protocol=yaml.safe_load((ROOT/"protocol_v1.yaml").read_text())
    job=next(j for j in module["build_jobs"](data,protocol) if j["model"]=="DT")
    dataset=data[(job["data_version"],job["cohort"])]
    folder=module["run_job"](tmp_path,job,dataset,protocol,"test-signature",grid={"max_depth":[2]})
    metrics=pd.read_csv(folder/"metrics.csv")
    assert set(metrics.split)=={"val","test"}
    assert metrics.reload_verified.all()
    folds=pd.read_csv(folder/"inner_fold_ids.csv")
    assert set(folds.record_id)==set(job["splits"]["train"])
    assert not set(folds.record_id)&set(job["splits"]["test"])
    assert module["verify_job"](folder,dataset,"test-signature")
    marker=(folder/"complete.json").read_bytes()
    module["run_job"](tmp_path,job,dataset,protocol,"test-signature",grid={"max_depth":[2]})
    assert (folder/"complete.json").read_bytes()==marker
    with (folder/"metrics.csv").open("a") as out:
        out.write("tampered\n")
    with pytest.raises(ValueError,match="hash"):
        module["verify_job"](folder,dataset,"test-signature")


def test_aggregation_reloads_models_and_pools_each_record_once(tmp_path):
    module=entry()
    data=module["load_datasets"]()
    protocol=yaml.safe_load((ROOT/"protocol_v1.yaml").read_text())
    jobs=module["build_jobs"](data,protocol)
    selected=[next(j for j in jobs if j["model"]=="DT")]
    selected += [j for j in jobs if j["model"]=="DT" and j["method"]=="stratified_cv"]
    for job in selected:
        module["run_job"](tmp_path,job,data[("V_raw","full")],protocol,"integration",
                          grid={"max_depth":[2]})
    aggregate=module["aggregate"]
    aggregate.__globals__["OUTPUT"]=tmp_path
    aggregate(data,selected,"integration")
    group=pd.read_csv(tmp_path/"group_cv.csv")
    pooled=group.query('split == "pooled_oof"').iloc[0]
    assert pooled.n==218
    assert len(group)==6
    assert (tmp_path/"importance_vs_paper.csv").is_file()
    assert pd.read_csv(tmp_path/"completion_summary.csv").set_index("key").at["all_reload_verified","value"]==1
