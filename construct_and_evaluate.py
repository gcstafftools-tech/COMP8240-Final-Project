"""Build a derived Wine stress-test dataset and evaluate it without target leakage.
New means a constructed perturbation of existing UCI Wine records; it is NOT fresh sampling.
"""
import csv, json, platform
from pathlib import Path
import numpy as np
import sklearn
from sklearn.datasets import load_wine
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
OUT=Path(__file__).resolve().parent
src=load_wine(); X=src.data.copy(); y=src.target.copy(); rng=np.random.default_rng(8240)
# Stress-test acquisition simulation: additive 1% feature-standard-deviation noise,
# followed by 8% missing measurements, independently of the labels.
scales=np.std(X,axis=0,ddof=1)
noise=rng.normal(size=X.shape)*scales[None,:]*0.01
mask=rng.random(X.shape)<0.08
X_derived=X+noise; X_derived[mask]=np.nan
with (OUT/'constructed_wine.csv').open('w',newline='') as f:
 w=csv.writer(f); w.writerow([*src.feature_names,'class_id'])
 for features,label in zip(X_derived,y): w.writerow(['' if np.isnan(x) else format(x,'.8g') for x in features]+[int(label)])
cv=RepeatedStratifiedKFold(n_splits=5,n_repeats=10,random_state=42)
results={'source':'sklearn.datasets.load_wine (UCI Wine), 178 samples, 13 features', 'constructed_data':'one derived record per original; Gaussian measurement perturbation at 1% of original per-feature sample SD; 8% independently masked cells; no new biological/chemical samples', 'seed':8240,'missing_cells':int(mask.sum()),'total_cells':int(mask.size),'missing_fraction':float(mask.mean()),'environment':{'python':platform.python_version(),'numpy':np.__version__,'sklearn':sklearn.__version__}}
for title,data in [('original_wine',X),('constructed_wine',X_derived)]:
 scores=[]; f1s=[]; baseline=[]
 for train,test in cv.split(data,y):
  model=make_pipeline(SimpleImputer(strategy='median'),LinearDiscriminantAnalysis())
  model.fit(data[train],y[train]); pred=model.predict(data[test]); scores.append(accuracy_score(y[test],pred)); f1s.append(f1_score(y[test],pred,average='macro'))
  b=DummyClassifier(strategy='most_frequent').fit(data[train],y[train]); baseline.append(accuracy_score(y[test],b.predict(data[test])))
 results[title]={'accuracy_mean':float(np.mean(scores)),'accuracy_sd':float(np.std(scores,ddof=1)),'macro_f1_mean':float(np.mean(f1s)),'macro_f1_sd':float(np.std(f1s,ddof=1)),'majority_accuracy_mean':float(np.mean(baseline))}
with (OUT/'constructed_data_results.json').open('w') as f:json.dump(results,f,indent=2)
print(json.dumps(results,indent=2))
