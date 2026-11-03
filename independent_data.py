"""Curate a genuinely separately sourced (not synthetic) external dataset.

Wisconsin Diagnostic Breast Cancer (WDBC) samples are NOT new patient records.
They are independent of the Iris and Wine collections. The curated CSV is newly
assembled from the original licensed benchmark via scikit-learn.

Re-run: python independent_data.py
"""
from pathlib import Path
import json, platform, sklearn, numpy as np, pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.metrics import confusion_matrix

base=Path(__file__).resolve().parent
ds=load_breast_cancer(as_frame=True)
X=ds.data.copy(); y=ds.target.copy()
assert X.shape==(569,30) and y.shape==(569,)
assert not X.isna().any().any()
assert set(y.unique())=={0,1}
# Preserve class definitions; no engineered predictors; no patient identifiers.
curated=X.copy(); curated.insert(0,'record_id',[f'WDBC_{n:04d}' for n in range(1,len(y)+1)])
curated['target_diagnosis']=y.map({0:'malignant',1:'benign'})
curated.to_csv(base/'independent_wdbc_curated.csv',index=False,float_format='%.10g')
cv=RepeatedStratifiedKFold(n_splits=5,n_repeats=10,random_state=42)
score={'accuracy':'accuracy','macro_f1':'f1_macro'}
output={}
for name,est in [('lda',LinearDiscriminantAnalysis()),('majority',DummyClassifier(strategy='most_frequent'))]:
  a=cross_validate(est,X,y,cv=cv,scoring=score, n_jobs=1)
  output[name]={k:{'mean':float(np.mean(a['test_'+k])),'sd':float(np.std(a['test_'+k]))} for k in score}
output['dataset']={'name':'Wisconsin Diagnostic Breast Cancer','source':'UCI Machine Learning Repository, scikit-learn load_breast_cancer','records':569,'features':30,'classes':{'malignant':212,'benign':357},'newly_collected':False,'independent_of_iris_wine':True,'license_check':'See UCI dataset page; verify dataset conditions before redistribution'}
output['protocol']={'cv':'RepeatedStratifiedKFold','n_splits':5,'n_repeats':10,'random_state':42,'python':platform.python_version(),'numpy':np.__version__,'sklearn':sklearn.__version__}
(base/'independent_wdbc_results.json').write_text(json.dumps(output,indent=2)+'\n')
print(json.dumps(output,indent=2))
