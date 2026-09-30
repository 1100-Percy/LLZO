"""Prepare, resume and verify every frozen M2 experiment (no later milestone)."""
from pathlib import Path
import argparse
import hashlib
from importlib.metadata import version
import json
import os
import pickle
import subprocess
import sys
import tempfile
import time
import traceback
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, ParameterGrid, StratifiedKFold

from llzo_pdp.features import build_features
from llzo_pdp.io import load_raw, TARGET_COL
from llzo_pdp.meta import write_meta
from llzo_pdp.models import ModelBundle, classification_metrics, fit_grid, make_estimator, safe_columns, split_ids
from llzo_pdp.preprocess import filter_rows, impute_features, threshold_table

OUTPUT = ROOT / "outputs/M2"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_text(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      default=lambda x: x.item() if isinstance(x, np.generic) else list(x))


def save(value, path, context=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    if isinstance(value, pd.DataFrame):
        value.to_csv(temp, index=False)
    elif path.suffix == ".pkl":
        temp.write_bytes(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))
    else:
        temp.write_text(json_text(value) + "\n")
    temp.replace(path)
    context = context or {}
    write_meta(path, seed=context.get("seed", [0,1,2,3,4]),
               data_version=context.get("data_version", "V_raw+V_corr"),
               script="scripts/run_m2.py", run_context=context)


def load_datasets():
    frozen = pd.read_csv(ROOT / "outputs/M1/input_manifest.csv").set_index("path").sha256
    for name in ("protocol_v1.yaml", "config/known_issues.csv", "config/dopant_properties.csv",
                 "data/raw/Machine_Learning_Guided_Design_dataset.xlsx"):
        if digest(ROOT / name) != frozen[name]:
            raise ValueError(f"Frozen input changed: {name}")
    raw = load_raw()
    data = {}
    ga = (raw[["dopant_1","dopant_2","dopant_3"]] == "Ga").any(axis=1)
    for version_name in ("V_raw", "V_corr"):
        features = build_features(raw, version_name)
        kept, _ = filter_rows(features.raw_features)
        for cohort, ids in (("full",kept), ("ga",kept[ga.loc[kept]])):
            data[(version_name,cohort)] = {
                "X": features.X.loc[ids], "numeric": features.numeric_columns,
                "sigma": raw.loc[ids,TARGET_COL], "sources": raw.loc[ids,"source_id"],
            }
    return data


def thresholds(sigma, protocol):
    return threshold_table(sigma, protocol["target"]["threshold_main"],
                           protocol["target"]["threshold_variants"]["T_1e4"])


def threshold_diagnostics(sigma, protocol):
    rows = []
    for threshold in thresholds(sigma,protocol).itertuples():
        y = (sigma >= threshold.threshold).astype(int)
        for seed in protocol["split_and_tuning"]["seeds"]:
            for stratified in (True,False):
                ids = split_ids(y,seed,stratified)["test"]
                rows.append({"threshold_name":threshold.threshold_name,"threshold":threshold.threshold,
                             "seed":seed,"split_kind":"stratified" if stratified else "unstratified",
                             "role":"protocol" if stratified else "diagnostic_only",
                             "n_test":len(ids),"true_0":int((y.loc[ids]==0).sum()),
                             "true_1":int((y.loc[ids]==1).sum())})
    result = pd.DataFrame(rows)
    # Paper values enter only this comparison, after all splits have been made.
    target = yaml.safe_load((ROOT/"paper_targets.yaml").read_text())["performance_test_set"]["confusion_all_four_models"]
    result["paper_true_0"], result["paper_true_1"] = target["true_0"],target["true_1"]
    result["count_distance_l1"] = abs(result.true_0-target["true_0"]) + abs(result.true_1-target["true_1"])
    result["matches_paper"] = result.count_distance_l1 == 0
    result["closest_within_split_kind"] = result.count_distance_l1 == result.groupby("split_kind").count_distance_l1.transform("min")
    return result


def build_jobs(data, protocol):
    jobs = []
    values = thresholds(data[("V_raw","full")]["sigma"],protocol).set_index("threshold_name").threshold.to_dict()
    seeds, main_seed = protocol["split_and_tuning"]["seeds"],protocol["split_and_tuning"]["main_seed"]
    settings = [("V_raw","full",t,s,model) for t in values for s in seeds for model in protocol["models"]["full_data"]]
    settings += [("V_raw","ga","T_main",s,model) for s in seeds for model in protocol["models"]["ga_subset"]]
    settings += [("V_corr","full","T_main",main_seed,model) for model in protocol["models"]["full_data"]]
    for v,c,t,s,model in settings:
        y = (data[(v,c)]["sigma"] >= values[t]).astype(int)
        jobs.append({"job_id":f"{c}_{model}_{v}_A0_{t}_seed{s}","cohort":c,"model":model,
                     "data_version":v,"imputation":"A0","threshold_name":t,"threshold":values[t],
                     "seed":s,"method":"holdout","fold":-1,"splits":split_ids(y,s)})
    dataset = data[("V_raw","full")]
    y = (dataset["sigma"] >= values["T_main"]).astype(int)
    outer = {"stratified_cv":StratifiedKFold(5,shuffle=True,random_state=main_seed),
             "group_kfold":GroupKFold(5),"loso":LeaveOneGroupOut()}
    for method, splitter in outer.items():
        iterator = splitter.split(dataset["X"],y) if method == "stratified_cv" else splitter.split(dataset["X"],y,dataset["sources"])
        for fold,(train,test) in enumerate(iterator):
            for model in protocol["models"]["full_data"]:
                jobs.append({"job_id":f"{method}_{model}_fold{fold}","cohort":"full","model":model,
                             "data_version":"V_raw","imputation":"A0","threshold_name":"T_main",
                             "threshold":values["T_main"],"seed":main_seed,"method":method,"fold":fold,
                             "splits":{"train":list(y.index[train]),"test":list(y.index[test])}})
    return jobs


def verify_job(folder, dataset, signature):
    marker = json.loads((folder/"complete.json").read_text())
    if marker["signature"] != signature:
        raise ValueError(f"Input/source signature changed: {folder.name}")
    root = folder.parent.parent
    for name, expected in marker["hashes"].items():
        if digest(root/name) != expected:
            raise ValueError(f"Output hash mismatch: {name}")
    bundle = pickle.loads((root/marker["model_path"]).read_bytes())
    predictions = pd.read_csv(folder/"predictions.csv")
    for split, group in predictions.groupby("split",sort=False):
        X = dataset["X"].loc[group.record_id]
        np.testing.assert_allclose(bundle.predict_proba(X)[:,1],group.p_high,rtol=0,atol=1e-12)
        np.testing.assert_array_equal(bundle.predict(X),group.y_pred)
        computed = classification_metrics(group.y_true,group.y_pred,group.p_high)
        saved = pd.read_csv(folder/"metrics.csv").query('split == @split').iloc[0]
        for key,value in computed.items():
            if key != "auc_reason":
                np.testing.assert_allclose(saved[key],value,rtol=0,atol=1e-12,equal_nan=True)
    return True


def run_job(output, job, dataset, protocol, signature, grid=None, imputation=None):
    folder = Path(output)/"jobs"/job["job_id"]
    if (folder/"complete.json").exists():
        verify_job(folder,dataset,signature)
        return folder
    start = time.perf_counter()
    X, y = dataset["X"], (dataset["sigma"] >= job["threshold"]).astype(int)
    imputation = imputation or impute_features(X,dataset["numeric"],"A0",job["seed"])
    split = job["splits"]
    search = fit_grid(imputation.X.loc[split["train"]],y.loc[split["train"]],
                      job["model"],job["seed"],grid or protocol["split_and_tuning"]["grids"][job["model"]])
    context = {k:v for k,v in job.items() if k != "splits"}
    context.update(best_params=search.best_params_,train_ids=split["train"],
                   imputer_fit_ids=imputation.fit_ids,imputer_warnings=imputation.convergence_warnings)
    bundle = ModelBundle(search.best_estimator_,imputation,list(X),context)
    filename = f"{job['model']}_{job['data_version']}_A0_{job['threshold_name']}_seed{job['seed']}.pkl"
    if job["method"]=="holdout":
        model_path = Path(output)/"models"/job["cohort"]/filename
    else:
        model_path = Path(output)/"models"/job["method"]/f"fold{job['fold']}_{filename}"
    save(bundle,model_path,context)
    restored = pickle.loads(model_path.read_bytes())
    identity = {k:v for k,v in job.items() if k != "splits"}
    metric_rows, prediction_rows = [],[]
    for kind,ids in split.items():
        if kind=="train":
            continue
        probability = bundle.predict_proba(X.loc[ids])[:,1]
        predicted = bundle.predict(X.loc[ids])
        np.testing.assert_allclose(restored.predict_proba(X.loc[ids])[:,1],probability,rtol=0,atol=1e-12)
        np.testing.assert_array_equal(restored.predict(X.loc[ids]),predicted)
        metric_rows.append({**identity,"split":kind,**classification_metrics(y.loc[ids],predicted,probability),
                            "reload_verified":True,"best_cv_f1":search.best_score_,
                            "best_params":json_text(search.best_params_),"elapsed_seconds":time.perf_counter()-start})
        prediction_rows.extend({**identity,"split":kind,"record_id":int(rid),
                                "source_id":int(dataset["sources"].loc[rid]),"y_true":int(y.loc[rid]),
                                "y_pred":int(pred),"p_high":float(prob)}
                               for rid,pred,prob in zip(ids,predicted,probability))
    artifacts = {"metrics.csv":pd.DataFrame(metric_rows),"predictions.csv":pd.DataFrame(prediction_rows)}
    artifacts["split_ids.csv"] = pd.DataFrame([{"split":kind,"record_id":rid,
                                                "source_id":int(dataset["sources"].loc[rid]),"label":int(y.loc[rid])}
                                               for kind,ids in split.items() for rid in ids])
    inner = []
    train_ids = np.asarray(split["train"])
    for fold,(fit,valid) in enumerate(search.cv.split(imputation.X.loc[train_ids],y.loc[train_ids])):
        inner.extend({"fold":fold,"role":role,"record_id":int(rid)}
                     for role,ids in (("fit",train_ids[fit]),("validation",train_ids[valid])) for rid in ids)
    artifacts["inner_fold_ids.csv"] = pd.DataFrame(inner)
    artifacts["cv_results.csv"] = pd.DataFrame({
        "params":[json_text(p) for p in search.cv_results_["params"]],
        **{key:search.cv_results_[key] for key in ("mean_test_score","std_test_score","rank_test_score","mean_fit_time")},
    })
    importance = pd.DataFrame({"feature":list(X),"importance":bundle.estimator.feature_importances_})
    importance = importance.sort_values("importance",ascending=False,kind="stable").reset_index(drop=True)
    importance["rank"] = np.arange(1,len(importance)+1)
    artifacts["importance.csv"] = importance
    paths = [model_path]
    for name,frame in artifacts.items():
        path = folder/name
        save(frame,path,context)
        paths.append(path)
    save({"signature":signature,"model_path":str(model_path.relative_to(output)),
          "job":job,"hashes":{str(p.relative_to(output)):digest(p) for p in paths}},folder/"complete.json",context)
    verify_job(folder,dataset,signature)
    return folder


def input_signature():
    names = ["protocol_v1.yaml","paper_targets.yaml","config/known_issues.csv","config/dopant_properties.csv",
             "data/raw/Machine_Learning_Guided_Design_dataset.xlsx","scripts/run_m2.py"]
    names += [str(p.relative_to(ROOT)) for p in sorted((ROOT/"src/llzo_pdp").glob("*.py"))]
    values = {name:digest(ROOT/name) for name in names}
    values.update({f"package:{p}":version(p) for p in ("numpy","pandas","scikit-learn","lightgbm","catboost")})
    return hashlib.sha256(json_text(values).encode()).hexdigest(),values


def run_tests():
    with tempfile.TemporaryDirectory() as temp:
        junit = Path(temp)/"tests.xml"
        subprocess.run([sys.executable,"-m","pytest","-q",f"--junitxml={junit}"],cwd=ROOT,check=True)
        suite = ET.parse(junit).getroot().find("testsuite")
        save(pd.DataFrame([{"key":k,"value":int(suite.attrib[k])} for k in ("tests","failures","errors","skipped")]),
             OUTPUT/"test_validation.csv")


def prepare(data,protocol,jobs):
    diag = threshold_diagnostics(data[("V_raw","full")]["sigma"],protocol)
    save(diag,OUTPUT/"threshold_check.csv")
    distribution = diag.query('threshold_name == "T_main"').groupby(["split_kind","true_0","true_1"]).size().reset_index(name="n_seeds")
    save(distribution,OUTPUT/"main_seed_class_distribution.csv")
    plan = pd.DataFrame([{**{k:v for k,v in j.items() if k!="splits"},
                           "n_train":len(j["splits"]["train"]),"n_test":len(j["splits"]["test"]),
                           "expected_fits":5*len(ParameterGrid(protocol["split_and_tuning"]["grids"][j["model"]]))+1}
                          for j in jobs])
    save(plan,OUTPUT/"task_plan.csv")
    dataset = data[("V_raw","full")]
    imp = impute_features(dataset["X"],dataset["numeric"],"A0",protocol["split_and_tuning"]["main_seed"])
    y = (dataset["sigma"]>=protocol["target"]["threshold_main"]).astype(int)
    ids = split_ids(y,protocol["split_and_tuning"]["main_seed"])["train"]
    probes = []
    for model in protocol["split_and_tuning"]["grids"]:
        grid = list(ParameterGrid(protocol["split_and_tuning"]["grids"][model]))
        estimator = make_estimator(model,protocol["split_and_tuning"]["main_seed"]).set_params(**grid[-1])
        start = time.perf_counter()
        estimator.fit(safe_columns(imp.X.loc[ids]),y.loc[ids])
        seconds = time.perf_counter()-start
        probes.append({"model":model,"parameters":json_text(grid[-1]),"fit_seconds":seconds,
                       "grid_candidates":len(grid),"expected_fits":int(plan.query('model == @model').expected_fits.sum()),
                       "estimated_seconds":seconds*plan.query('model == @model').expected_fits.sum()})
    probes = pd.DataFrame(probes)
    save(probes,OUTPUT/"runtime_probe.csv")
    estimate = probes.estimated_seconds.sum()
    save(pd.DataFrame([{"key":k,"value":v} for k,v in {
        "jobs":len(jobs),"fits":plan.expected_fits.sum(),"estimate_minutes":estimate/60,
        "conservative_minutes":estimate*1.8/60,"test_sample_percentage_points":100/diag.n_test.iloc[0],
        "n_test":diag.n_test.iloc[0]}.items()]),OUTPUT/"budget.csv")
    signature,inputs = input_signature()
    save({"signature":signature,"inputs":inputs},OUTPUT/"input_manifest.json")


def aggregate(data,jobs,signature):
    metric_parts, prediction_parts, importance_parts = [],[],[]
    paper = yaml.safe_load((ROOT/"paper_targets.yaml").read_text())
    for job in jobs:
        folder = OUTPUT/"jobs"/job["job_id"]
        verify_job(folder,data[(job["data_version"],job["cohort"])],signature)
        metrics = pd.read_csv(folder/"metrics.csv")
        predictions = pd.read_csv(folder/"predictions.csv")
        metric_parts.append(metrics)
        prediction_parts.append(predictions)
        if job["method"]=="holdout" and job["cohort"]=="full" and job["data_version"]=="V_raw" and job["threshold_name"]=="T_main" and job["seed"]==0:
            imp = pd.read_csv(folder/"importance.csv")
            target = paper["top5_features_fig7"][job["model"]]
            imp["paper_rank"] = imp.feature.map({f:i+1 for i,f in enumerate(target)})
            imp["own_top15"] = imp["rank"]<=15
            imp["paper_top5"] = imp.feature.isin(target)
            importance_parts.append(imp[imp.own_top15|imp.paper_top5].assign(model=job["model"]))
    metrics = pd.concat(metric_parts,ignore_index=True)
    predictions = pd.concat(prediction_parts,ignore_index=True)
    save(metrics.query('method == "holdout" and cohort == "full"'),OUTPUT/"metrics.csv")
    save(metrics.query('method == "holdout" and cohort == "ga"'),OUTPUT/"ga_metrics.csv")
    group = metrics.query('method != "holdout"').copy()
    pooled = []
    for (method,model),frame in predictions.query('method != "holdout"').groupby(["method","model"]):
        if frame.record_id.duplicated().any() or set(frame.record_id)!=set(data[("V_raw","full")]["X"].index):
            raise ValueError("OOF predictions must cover each retained record exactly once")
        pooled.append({"method":method,"model":model,"fold":-1,"split":"pooled_oof",
                       "data_version":"V_raw","cohort":"full","imputation":"A0","threshold_name":"T_main",
                       **classification_metrics(frame.y_true,frame.y_pred,frame.p_high),"reload_verified":True})
    save(pd.concat([group,pd.DataFrame(pooled)],ignore_index=True),OUTPUT/"group_cv.csv")
    save(pd.concat(importance_parts,ignore_index=True),OUTPUT/"importance_vs_paper.csv")
    save(predictions,OUTPUT/"predictions.csv")
    save(pd.DataFrame([{"key":"completed_jobs","value":len(jobs)},
                       {"key":"all_reload_verified","value":int(metrics.reload_verified.all())},
                       {"key":"single_class_test_folds","value":int((group.auc_reason=="single_class_test").sum())}]),OUTPUT/"completion_summary.csv")


def report(stage):
    diag = pd.read_csv(OUTPUT/"threshold_check.csv")
    budget = pd.read_csv(OUTPUT/"budget.csv").set_index("key").value
    lines = ["# M2 阶段检查", "", f"当前状态：{stage}。本研究选择见 DECISIONS.md。", "",
             "## 人类反馈：主阈值的类别计数", "",
             "以下仅回答协议种子中是否观察到原文类别数；不分层仅作诊断，不用于主模型。", "",
             "| 划分 | seed | 测试低导数 | 测试高导数 | 与原文一致 |", "|---|---:|---:|---:|---|"]
    claims=[]
    for row in diag.itertuples():
        key=f"threshold_name={row.threshold_name}|seed={row.seed}|split_kind={row.split_kind}"
        for col in ("true_0","true_1","n_test","count_distance_l1"):
            claims.append({"claim_id":f"M2-{row.threshold_name}-{row.seed}-{row.split_kind}-{col}",
                           "statement":f"类别计数诊断 {key} {col}","artifact_path":"outputs/M2/threshold_check.csv",
                           "key":f"{key}|{col}","value":getattr(row,col)})
        if row.threshold_name=="T_main":
            lines.append(f"| {row.split_kind} | {row.seed} | {row.true_0} | {row.true_1} | {row.matches_paper} |")
    lines += ["", "## 计算与验收", "",
              f"计划 {int(budget['jobs'])} 个网格搜索任务，共 {int(budget['fits'])} 次候选/最终拟合。",
              f"短计时外推 {budget['estimate_minutes']:.1f} 分钟，留余量估计 {budget['conservative_minutes']:.1f} 分钟；不是完成时限承诺。",
              f"主实验测试集 {int(budget['n_test'])} 条，改变一条判定即 {budget['test_sample_percentage_points']:.2f} 个百分点，不对模型排名下结论。"]
    for name in ("budget.csv","test_validation.csv","completion_summary.csv"):
        path=OUTPUT/name
        if path.exists():
            for row in pd.read_csv(path).itertuples():
                claims.append({"claim_id":f"M2-{name}-{row.key}","statement":f"M2 {row.key}",
                               "artifact_path":f"outputs/M2/{name}","key":f"key={row.key}|value","value":row.value})
    if (OUTPUT/"test_validation.csv").exists():
        tests=pd.read_csv(OUTPUT/"test_validation.csv").set_index("key").value
        lines.append(f"程序测试 {tests['tests']} 项，失败 {tests['failures']}，错误 {tests['errors']}，跳过 {tests['skipped']}。")
    lines += ["", "## 人类需要看", "",
              "- threshold_check.csv 与 main_seed_class_distribution.csv：反馈要求的全部类别计数。",
              "- task_plan.csv、runtime_probe.csv 与 budget.csv：任务覆盖和预算依据。",
              "- run.log、run_state.json 与 jobs/*/complete.json：实际进度及失败位置。",
              "- 完成后查看 metrics.csv、ga_metrics.csv、group_cv.csv 与 importance_vs_paper.csv；完整预测和模型均可重载核验。",
              "", "## 未解决问题", "",
              "A0 使用全部保留行拟合插补，分组测试也不能称为无泄漏；A1 属后续敏感性分析。",
              "尚未完成的任务不能据已有部分结果得出模型比较结论。下一会话先检查后台状态，不启动 M3。"]
    checkpoint=ROOT/"reports/checkpoints/M2.md"
    checkpoint.write_text("\n".join(lines)+"\n")
    write_meta(checkpoint,seed=[0,1,2,3,4],data_version="V_raw+V_corr",script="scripts/run_m2.py")
    path=ROOT/"reports/claims.csv"
    existing=pd.read_csv(path)
    existing=existing[~existing.claim_id.str.startswith("M2-")]
    pd.concat([existing,pd.DataFrame(claims)],ignore_index=True).to_csv(path,index=False)
    write_meta(path,seed=[0,1,2,3,4],data_version="V_raw+V_corr",script="scripts/run_m2.py")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--prepare",action="store_true")
    parser.add_argument("--verify",action="store_true")
    args=parser.parse_args()
    protocol=yaml.safe_load((ROOT/"protocol_v1.yaml").read_text())
    data=load_datasets()
    jobs=build_jobs(data,protocol)
    if args.prepare or not (OUTPUT/"input_manifest.json").exists():
        prepare(data,protocol,jobs)
    signature,_=input_signature()
    manifest=json.loads((OUTPUT/"input_manifest.json").read_text())
    if signature!=manifest["signature"]:
        raise ValueError("Prepared input/source signature changed; inspect before rerunning --prepare")
    if args.prepare:
        run_tests()
        report("程序和诊断已验收；完整计算待后台运行")
        return
    if args.verify:
        aggregate(data,jobs,signature)
        run_tests()
        report("全部计算和重载验证完成；待会话更新里程碑状态并提交")
        return
    cache={}
    started=time.time()
    current=None
    try:
        for i,job in enumerate(jobs):
            current=job["job_id"]
            if time.time()-started>protocol["compute_budget"]["max_wall_hours_per_milestone"]*3600:
                raise TimeoutError("Protocol wall-time budget reached; no grid reduction allowed for M2")
            save({"status":"running","pid":os.getpid(),"completed_jobs":i,"total_jobs":len(jobs),
                  "current_job":current,"started_unix":started},OUTPUT/"run_state.json")
            print(f"[{i+1}/{len(jobs)}] {current}",flush=True)
            dataset=data[(job["data_version"],job["cohort"])]
            key=(job["data_version"],job["cohort"],job["seed"])
            if key not in cache:
                cache[key]=impute_features(dataset["X"],dataset["numeric"],"A0",job["seed"])
            run_job(OUTPUT,job,dataset,protocol,signature,imputation=cache[key])
        aggregate(data,jobs,signature)
        run_tests()
        report("全部计算和重载验证完成；待会话更新里程碑状态并提交")
        save({"status":"complete","pid":os.getpid(),"completed_jobs":len(jobs),"total_jobs":len(jobs),
              "elapsed_seconds":time.time()-started},OUTPUT/"run_state.json")
        print("M2 COMPUTATION COMPLETE; all saved models reloaded and checked.",flush=True)
    except Exception:
        save({"status":"failed","pid":os.getpid(),"current_job":current,"traceback":traceback.format_exc()},OUTPUT/"run_state.json")
        raise
    finally:
        if (OUTPUT/"run.log").exists():
            sys.stdout.flush()
            sys.stderr.flush()
            write_meta(OUTPUT/"run.log",seed=[0,1,2,3,4],data_version="V_raw+V_corr",script="scripts/run_m2.py")


if __name__=="__main__":
    main()
