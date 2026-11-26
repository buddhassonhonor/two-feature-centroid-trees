"""Post-primary diagnostic fixed-depth evaluation; never selects by outer test score."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
from concurrent.futures import ProcessPoolExecutor,as_completed
import json,time
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from run_experiments import ROOT,candidates,preprocessor,metrics,structure,atomic_json
from run_supplement import GEOMS,run_task

def evaluate(did,seed,method):
    threadpool_limits(1)
    meta=json.loads((ROOT/f'exp/data/{did}.json').read_text());X=pd.read_pickle(ROOT/f'exp/data/{did}_X.pkl');y=np.load(ROOT/f'exp/data/{did}_y.npz')['y']
    split=np.load(ROOT/f'exp/splits/{did}_{seed}.npz');full=np.r_[split['train'],split['val']];te=split['test']
    prep=preprocessor(meta);xf=np.asarray(prep.fit_transform(X.iloc[full]),float);xe=np.asarray(prep.transform(X.iloc[te]),float)
    for params,model in candidates(method,seed,xf.shape[1]):
        path=ROOT/f'exp/matched/{did}/{seed}/{method}/{params["depth"]}';path.mkdir(parents=True,exist_ok=True)
        if (path/'result.json').exists():continue
        t=time.perf_counter();model.fit(xf,y[full]);seconds=time.perf_counter()-t
        pred=model.predict(xe);prob=model.predict_proba(xe)
        np.savez_compressed(path/'predictions.npz',y=y[te],pred=pred,proba=prob,test_rows=te)
        atomic_json(path/'result.json',dict(dataset=did,name=meta['name'],seed=seed,method=method,depth_limit=params['depth'],
                       fit_seconds=seconds,**metrics(y[te],pred,prob),**structure(model,xe)))
    return did,seed,method

def main():
    ids=pd.read_csv(ROOT/'exp/data/manifest.csv').id.tolist()
    with ProcessPoolExecutor(max_workers=4) as ex:
        fs=[ex.submit(evaluate,d,s,m) for d in ids for s in range(5) for m in ['P2','P2-H','CART','Oblique-2']]
        for i,f in enumerate(as_completed(fs),1):print('depth',i,len(fs),f.result(),flush=True)
    with ProcessPoolExecutor(max_workers=2) as ex:
        fs=[ex.submit(run_task,g,s,'replication',0,m) for g in GEOMS for s in range(10,15)
            for m in ['P2','P2-H','CART','Oblique-2','P2-Tuned','P-Dense','P2-Random']]
        for i,f in enumerate(as_completed(fs),1):print('replication',i,len(fs),f.result(),flush=True)
    d=pd.DataFrame([json.loads(p.read_text()) for p in (ROOT/'exp/matched').rglob('result.json')]);d.to_csv(ROOT/'exp/matched_depth.csv',index=False)
    r=pd.DataFrame([json.loads(p.read_text()) for p in (ROOT/'exp/supplement').rglob('result.json')
                    if json.loads(p.read_text()).get('factor')=='replication']);r.to_csv(ROOT/'exp/replication.csv',index=False)

if __name__=='__main__':main()
