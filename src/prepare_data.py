from pathlib import Path
import gzip, json, hashlib
import numpy as np
import pandas as pd
import arff
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
CACHE = Path('D:/data/openml/openml.org')
# Named tasks specified before seeing any method test results. No duplicate versions.
IDS = [11,13,15,18,21,23,24,29,31,32,36,37,40,43,44,50,53,54,56,59,61,
       1063,1067,1068,1120,1461,1462,1464,1480,1489,1497,1510,1524,187,40983]

def main():
    output = ROOT/'exp/data'; output.mkdir(parents=True, exist_ok=True)
    rows = []
    for did in IDS:
        meta_path = CACHE/f'api/v1/json/data/{did}.gz'
        d = json.loads(gzip.decompress(meta_path.read_bytes()))['data_set_description']
        paths = list((CACHE/f'data/v1/download/{d["file_id"]}').glob('*.arff.gz'))
        if len(paths) != 1: raise RuntimeError((did, paths))
        path = paths[0]
        with gzip.open(path, 'rt', encoding='utf-8') as f:
            obj = arff.load(f)
        columns = [a[0] for a in obj['attributes']]
        df = pd.DataFrame(obj['data'], columns=columns)
        target = d['default_target_attribute']
        df = df.loc[df[target].notna()].reset_index(drop=True)
        y, labels = pd.factorize(df[target].astype(str), sort=True)
        X = df.drop(columns=target)
        removed = d.get('ignore_attribute', [])
        if isinstance(removed, str): removed = [removed]
        if d.get('row_id_attribute'): removed = list(removed) + [d['row_id_attribute']]
        X = X.drop(columns=[c for c in removed if c in X])
        types = {a[0]:a[1] for a in obj['attributes']}
        cats = [c for c in X if isinstance(types[c], list) or str(types[c]).lower() not in ('numeric','real','integer')]
        nums = [c for c in X if c not in cats]
        for c in nums: X[c] = pd.to_numeric(X[c], errors='coerce')
        for c in cats: X[c] = X[c].map(lambda v: str(v) if pd.notna(v) else np.nan)
        original_rows = np.arange(len(y)); original_n = len(y)
        if len(y) > 5000:
            original_rows, _ = train_test_split(original_rows, train_size=5000, stratify=y, random_state=9183)
            original_rows = np.sort(original_rows)
            X = X.iloc[original_rows].reset_index(drop=True); y = y[original_rows]
        # No global imputation, normalization or feature filtering.
        name = d['name']
        X.to_pickle(output/f'{did}_X.pkl')
        np.savez_compressed(output/f'{did}_y.npz', y=y, original_rows=original_rows)
        manifest = dict(id=did, name=name, n=len(y), original_n=original_n, p=X.shape[1],
                        numerical=nums, categorical=cats, labels=labels.tolist(),
                        class_counts=np.bincount(y).tolist(), missing=int(X.isna().sum().sum()),
                        source=f'https://www.openml.org/d/{did}', source_file=str(path),
                        source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                        ignore_attributes=d.get('ignore_attribute', []), row_id_attribute=d.get('row_id_attribute'))
        (output/f'{did}.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        rows.append({k:manifest[k] for k in ('id','name','n','original_n','p','missing','source','source_sha256')})
        print(did, name, X.shape, np.bincount(y), flush=True)
    pd.DataFrame(rows).to_csv(output/'manifest.csv', index=False)

if __name__ == '__main__': main()
