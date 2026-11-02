"""Reproducible educational experiment: Fisher-style linear discriminant classification."""
import json, platform
from pathlib import Path
OUT = Path(__file__).resolve().parent
import numpy as np
import sklearn
from sklearn.datasets import load_iris, load_wine
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate, cross_val_predict, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

cv=RepeatedStratifiedKFold(n_splits=5,n_repeats=10,random_state=42)
models={'majority':DummyClassifier(strategy='most_frequent'), 'lda':LinearDiscriminantAnalysis(), 'logistic':make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000))}
results={'environment':{'python':platform.python_version(),'numpy':np.__version__,'sklearn':sklearn.__version__},'protocol':'RepeatedStratifiedKFold(5 splits, 10 repeats, random_state=42); accuracy and macro F1'}
for key,loader in [('iris',load_iris),('wine',load_wine)]:
    bunch=loader(); X,y=bunch.data,bunch.target
    d={'n':int(len(y)),'features':int(X.shape[1]),'classes':np.bincount(y).tolist(),'models':{},'first_record':[round(float(v),3) for v in X[0]],'first_label':int(y[0])}
    for name,model in models.items():
        out=cross_validate(model,X,y,cv=cv,scoring={'accuracy':'accuracy','macro_f1':'f1_macro'},n_jobs=1)
        d['models'][name]={k:{'mean':float(np.mean(v)),'std':float(np.std(v,ddof=1))} for k,v in [('accuracy',out['test_accuracy']),('macro_f1',out['test_macro_f1'])]}
    pred=cross_val_predict(LinearDiscriminantAnalysis(),X,y,cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=42))
    cm=confusion_matrix(y,pred)
    d['lda_5fold_confusion_matrix']=cm.tolist()
    fig,ax=plt.subplots(figsize=(4.0,3.3)); im=ax.imshow(cm,cmap='Blues'); ax.set(xticks=range(3),yticks=range(3),xlabel='Predicted class',ylabel='True class',title=f'{key.title()}: LDA out-of-fold predictions');
    for (i,j),v in np.ndenumerate(cm): ax.text(j,i,str(v),ha='center',va='center',color='white' if v>cm.max()/2 else 'black')
    fig.tight_layout(); fig.savefig(OUT / f'{key}_confusion.pdf'); plt.close(fig)
    results[key]=d
with open(OUT / 'results.json','w') as f:json.dump(results,f,indent=2)
for key in ['iris','wine']:
 print(key,results[key]['models'],results[key]['lda_5fold_confusion_matrix'])
