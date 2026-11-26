from __future__ import annotations
import os
for _k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_k] = '1'
import argparse, json, time, traceback, hashlib, platform, sys, importlib.metadata
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import joblib
import psutil
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from obliquetree import Classifier as ObliqueClassifier
from obliquetree.utils import export_tree
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from models import ProjectionTree

ROOT = Path(__file__).resolve().parents[1]
DEPTHS = [1,2,3,4,6]
METHODS = ['P2','P2-H','CART','CART-Entropy','ExtraTree','Oblique-2','Oblique-Dense',
           'LDA-Tree','Logistic-Tree','Random-Projection','Logistic','RBF-SVM','kNN',
           'RandomForest','ExtraTrees','HistGB','AdaBoost','GaussianNB','XGBoost','LightGBM']
ABLATIONS = ['P2-Tuned','P-Dense','P2-Random','P2-Unscaled']

def preprocessor(meta, scale=True):
    transformers = []
    if meta['numerical']:
        transformers.append(('num', make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True),
                                                  StandardScaler() if scale else 'passthrough'), meta['numerical']))
    if meta['categorical']:
        transformers.append(('cat', make_pipeline(SimpleImputer(strategy='most_frequent', keep_empty_features=True),
                                                  OneHotEncoder(handle_unknown='ignore', sparse_output=False)), meta['categorical']))
    return ColumnTransformer(transformers, sparse_threshold=0)

def group_split(X, y, seed):
    groups = pd.util.hash_pandas_object(X, index=False).to_numpy()
    unique, inverse = np.unique(groups, return_inverse=True)
    # Stratify groups by their majority label; contradictory-label groups stay intact.
    gy = np.array([np.bincount(y[inverse==g]).argmax() for g in range(len(unique))])
    gi = np.arange(len(unique))
    strat = gy if np.bincount(gy).min() >= 4 else None
    for attempt in range(50):
        full, test = train_test_split(gi, test_size=.25, stratify=strat, random_state=120+seed+attempt*997)
        gfull = gy[full]
        stratfull = gfull if np.bincount(gfull, minlength=len(np.unique(y))).min() >= 2 else None
        train, val = train_test_split(full, test_size=.25, stratify=stratfull, random_state=260+seed+attempt*997)
        parts = [np.flatnonzero(np.isin(inverse, g)) for g in (train,val,test)]
        if all(len(np.unique(y[i])) == len(np.unique(y)) for i in parts):
            assert not (set(groups[parts[0]]) & set(groups[parts[1]]))
            assert not (set(groups[parts[0]]) & set(groups[parts[2]]))
            return parts
    raise RuntimeError('Could not split groups with all labels present')

def candidates(method, seed, p, leaf=8):
    mode = {'P2':'p2','P2-H':'hybrid','P2-Tuned':'tuned','P-Dense':'dense','P2-Random':'random_pair',
            'P2-Unscaled':'p2','LDA-Tree':'lda','Logistic-Tree':'logistic','Random-Projection':'random'}
    if method in mode:
        return [(dict(depth=d, leaf=leaf, mode=mode[method]),
                 ProjectionTree(d, leaf, mode[method], seed)) for d in DEPTHS]
    if method in ('CART','CART-Entropy','ExtraTree'):
        cls = ExtraTreeClassifier if method == 'ExtraTree' else DecisionTreeClassifier
        criterion = 'entropy' if method == 'CART-Entropy' else 'gini'
        return [(dict(depth=d, leaf=leaf), cls(max_depth=d, min_samples_leaf=leaf,
                                            criterion=criterion, random_state=seed)) for d in DEPTHS]
    if method in ('Oblique-2','Oblique-Dense'):
        support = min(p, 2) if method == 'Oblique-2' else p
        return [(dict(depth=d, leaf=leaf, support=support),
                 ObliqueClassifier(max_depth=d, min_samples_leaf=leaf, n_pair=support,
                                   max_iter=100, random_state=seed)) for d in DEPTHS]
    if method == 'Logistic':
        return [(dict(C=c), LogisticRegression(C=c, max_iter=1000, random_state=seed)) for c in [.01,.1,1,10,100]]
    if method == 'RBF-SVM':
        return [(dict(C=c), SVC(C=c, kernel='rbf', probability=True, random_state=seed)) for c in [.1,1,10,100]]
    if method == 'kNN':
        return [(dict(k=k), KNeighborsClassifier(n_neighbors=k, weights='distance')) for k in [3,5,11,21]]
    if method in ('RandomForest','ExtraTrees'):
        cls = RandomForestClassifier if method == 'RandomForest' else ExtraTreesClassifier
        return [(dict(leaf=l), cls(n_estimators=100, min_samples_leaf=l, n_jobs=1, random_state=seed)) for l in [1,2,4,8]]
    if method == 'HistGB':
        return [(dict(leaves=l), HistGradientBoostingClassifier(max_iter=150, max_leaf_nodes=l,
                                                              early_stopping=False, random_state=seed)) for l in [7,15,31,63]]
    if method == 'AdaBoost':
        return [(dict(rate=r), AdaBoostClassifier(n_estimators=100, learning_rate=r, random_state=seed)) for r in [.1,.3,1,2]]
    if method == 'GaussianNB':
        return [(dict(smoothing=v), GaussianNB(var_smoothing=v)) for v in [1e-9,1e-7,1e-5,1e-3]]
    if method == 'XGBoost':
        return [(dict(depth=d), XGBClassifier(max_depth=d, n_estimators=150, learning_rate=.1,
                                             tree_method='hist', n_jobs=1, random_state=seed)) for d in [2,3,4,6]]
    if method == 'LightGBM':
        return [(dict(leaves=l), LGBMClassifier(num_leaves=l, n_estimators=150, learning_rate=.1,
                                               verbosity=-1, n_jobs=1, random_state=seed)) for l in [7,15,31,63]]
    raise ValueError(method)

def structure(model, X):
    if isinstance(model, ProjectionTree): return model.complexity(X)
    if isinstance(model, (DecisionTreeClassifier, ExtraTreeClassifier)):
        length = model.decision_path(X).sum(axis=1).A1 - 1
        return dict(leaves=int(model.get_n_leaves()), depth=int(model.get_depth()), path_tests=float(length.mean()),
                    path_burden=float(length.mean()), radius_median=None)
    if isinstance(model, ObliqueClassifier):
        tree = export_tree(model)['tree']
        sizes, depths, burdens, tests = [], [], [], []
        def walk(n, rows, depth, burden):
            if n['is_leaf']:
                sizes.append(1); depths.append(depth)
                burdens.extend([burden]*len(rows)); tests.extend([depth]*len(rows)); return
            if n['is_oblique']:
                features = n['features']; w = np.asarray(n['weights'])
                z = X[rows][:,features] @ w
                support = int(np.count_nonzero(np.abs(w)>1e-12))
            else:
                feature = n.get('feature_idx', n.get('feature', None))
                if feature is None: raise ValueError(str(n)[:300])
                z = X[rows,feature]; support = 1
            mask = z <= n['threshold']
            walk(n['left'], rows[mask], depth+1, burden+support)
            walk(n['right'], rows[~mask], depth+1, burden+support)
        walk(tree, np.arange(len(X)), 0, 0)
        return dict(leaves=len(sizes), depth=max(depths), path_tests=float(np.mean(tests)),
                    path_burden=float(np.mean(burdens)), radius_median=None)
    return dict(leaves=None, depth=None, path_tests=None, path_burden=None, radius_median=None)

def metrics(y, pred, proba):
    result = dict(accuracy=float(accuracy_score(y,pred)), balanced_accuracy=float(balanced_accuracy_score(y,pred)),
                  macro_f1=float(f1_score(y,pred,average='macro',zero_division=0)),
                  brier=float(np.mean(np.sum((np.eye(proba.shape[1])[y]-proba)**2,axis=1))))
    result['auc'] = float(roc_auc_score(y, proba[:,1])) if proba.shape[1]==2 else float(roc_auc_score(y,proba,multi_class='ovr',average='macro'))
    return result

def atomic_json(path, data):
    tmp = path.with_suffix('.tmp'); tmp.write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf-8'); tmp.replace(path)

def evaluate(did, seed, method, output='main'):
    threadpool_limits(1)
    path = ROOT/f'exp/runs/{output}/{did}/{seed}/{method}'
    path.mkdir(parents=True, exist_ok=True)
    if (path/'result.json').exists(): return str(path), 'cached'
    started = time.perf_counter()
    try:
        meta = json.loads((ROOT/f'exp/data/{did}.json').read_text())
        X = pd.read_pickle(ROOT/f'exp/data/{did}_X.pkl')
        y = np.load(ROOT/f'exp/data/{did}_y.npz')['y']
        split_path = ROOT/f'exp/splits/{did}_{seed}.npz'
        split = np.load(split_path); tr, va, te = [split[k] for k in ('train','val','test')]
        scale = method != 'P2-Unscaled'
        prep = preprocessor(meta, scale=scale)
        xt = np.asarray(prep.fit_transform(X.iloc[tr]),dtype=np.float64)
        xv = np.asarray(prep.transform(X.iloc[va]),dtype=np.float64)
        search_start = time.perf_counter(); scores = []; best_key = None; best_params = None
        for params, model in candidates(method, seed, xt.shape[1]):
            t = time.perf_counter(); model.fit(xt, y[tr])
            score = float(balanced_accuracy_score(y[va],model.predict(xv)))
            comp = structure(model, xv)
            key = (score, -(comp['leaves'] or 0), -(comp['depth'] or 0))
            scores.append(dict(params=params,score=score,seconds=time.perf_counter()-t,**comp))
            if best_key is None or key > best_key:
                best_key, best_params = key, params
        search_seconds = time.perf_counter()-search_start
        full = np.concatenate([tr,va]); prep = preprocessor(meta, scale=scale)
        xf = np.asarray(prep.fit_transform(X.iloc[full]),dtype=np.float64)
        xe = np.asarray(prep.transform(X.iloc[te]),dtype=np.float64)
        # One-hot width can change on refit; dense support must follow the refit width.
        matching=[(p,m) for p,m in candidates(method,seed,xf.shape[1])
                  if {k:v for k,v in p.items() if k!='support'} ==
                     {k:v for k,v in best_params.items() if k!='support'}]
        refit_params, model = matching[0]
        t = time.perf_counter(); model.fit(xf,y[full]); fit_seconds = time.perf_counter()-t
        t = time.perf_counter(); pred = model.predict(xe); proba = model.predict_proba(xe); pred_seconds = time.perf_counter()-t
        np.savez_compressed(path/'predictions.npz',y=y[te],pred=pred,proba=proba,test_rows=te)
        joblib.dump(prep, path/'preprocessor.joblib')
        if isinstance(model, ObliqueClassifier): atomic_json(path/'model.json',export_tree(model))
        else: joblib.dump(model,path/'model.joblib')
        if isinstance(model, ProjectionTree): atomic_json(path/'tree.json',dict(nodes=model.nodes,features=prep.get_feature_names_out().tolist()))
        row = dict(dataset=did,name=meta['name'],seed=seed,method=method,suite=output,
                   n_train=len(full),n_test=len(te),p=xf.shape[1],selected=best_params,refit_params=refit_params,
                   validation_score=best_key[0], search_seconds=search_seconds, fit_seconds=fit_seconds,
                   prediction_seconds=pred_seconds,total_seconds=time.perf_counter()-started,
                   rss_mb=psutil.Process().memory_info().rss/2**20,status='ok',
                   **metrics(y[te],pred,proba), **structure(model,xe))
        atomic_json(path/'validation.json',scores)
        atomic_json(path/'result.json',row)
        if (path/'error.txt').exists(): (path/'error.txt').rename(path/'resolved_error.txt')
        return str(path), 'ok'
    except Exception:
        (path/'error.txt').write_text(traceback.format_exc(),encoding='utf-8')
        return str(path), 'error'

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--seeds',default='0,1,2,3,4'); parser.add_argument('--datasets',default='')
    parser.add_argument('--methods',default=','.join(METHODS)); parser.add_argument('--suite',default='main')
    args=parser.parse_args()
    ids=pd.read_csv(ROOT/'exp/data/manifest.csv')['id'].tolist() if not args.datasets else list(map(int,args.datasets.split(',')))
    seeds=list(map(int,args.seeds.split(','))); methods=args.methods.split(',')
    (ROOT/'exp/splits').mkdir(parents=True,exist_ok=True)
    (ROOT/'exp/metadata').mkdir(parents=True,exist_ok=True)
    environment=dict(python=sys.version, executable=sys.executable,platform=platform.platform(),
                     cpu='Intel Core Ultra 9 275HX', logical_cpus=os.cpu_count(), memory_gb=psutil.virtual_memory().total/2**30,
                     workers=args.workers,blas_threads=1, gpu_used=False,
                     versions={p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','scikit-learn','obliquetree','xgboost','lightgbm','psutil']},
                     command=sys.argv,source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'src').glob('*.py')},
                     time_limit_policy='No per-fit timeout; errors logged explicitly; checkpoints permit resume.')
    env_path=ROOT/f'exp/metadata/{args.suite}_environment.json'
    if env_path.exists():
        invocation_hash=hashlib.sha256(json.dumps(environment,sort_keys=True).encode()).hexdigest()[:12]
        env_path=ROOT/f'exp/metadata/{args.suite}_invocation_{invocation_hash}.json'
    atomic_json(env_path,environment)
    for did in ids:
        X=pd.read_pickle(ROOT/f'exp/data/{did}_X.pkl'); y=np.load(ROOT/f'exp/data/{did}_y.npz')['y']
        for seed in seeds:
            p=ROOT/f'exp/splits/{did}_{seed}.npz'
            if not p.exists():
                tr,va,te=group_split(X,y,seed); np.savez_compressed(p,train=tr,val=va,test=te)
    tasks=[(d,s,m) for d in ids for s in seeds for m in methods]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures={ex.submit(evaluate,*t,args.suite):t for t in tasks}
        for count,f in enumerate(as_completed(futures),1):
            path,status=f.result(); print(f'{count}/{len(tasks)} {futures[f]} {status}',flush=True)
    failures=list((ROOT/f'exp/runs/{args.suite}').rglob('error.txt'))
    print('ERROR_FILES', len(failures),flush=True)

if __name__=='__main__': main()
