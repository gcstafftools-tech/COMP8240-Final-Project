"""Numerically replicate Fisher's 1936 two-Iris-species discriminant.

Original source: Fisher (1936), Tables II--IV, pp. 181--182.
Table entries and reported coefficients are transcribed from the historical paper.
"""
import json
from pathlib import Path
import numpy as np
from sklearn.datasets import load_iris

root=Path(__file__).resolve().parent
X,y=load_iris(return_X_y=True)
A=X[y==0]; B=X[y==1]
assert A.shape==B.shape==(50,4)
mA=A.mean(axis=0); mB=B.mean(axis=0); d=mB-mA
S=(A-mA).T@(A-mA)+(B-mB).T@(B-mB)
w=np.linalg.solve(S,d)
w_norm=w/w[0]
reported_d=np.array([.930,-.658,2.798,1.080])
reported_S=np.array([[19.1434,9.0356,9.7634,3.2394],[9.0356,11.8658,4.6232,2.4746],[9.7634,4.6232,12.2978,3.8794],[3.2394,2.4746,3.8794,2.4604]])
reported_w=np.array([-.0311511,-.1839075,.2221044,.3147370])
reported_normal=np.array([1.,5.9037,-7.1299,-10.1036])
observed_a=float(mA@w_norm); observed_b=float(mB@w_norm)
original_a=12.3345;original_b=-21.4815
result={
 'source':'Fisher 1936, Annals of Eugenics 7(2), pp. 181-183, Tables II-IV',
 'comparison':'Iris setosa versus versicolor, 50 observations each',
 'published_mean_difference':reported_d.tolist(),
 'recomputed_mean_difference':d.tolist(),
 'max_absolute_mean_difference':float(np.max(np.abs(d-reported_d))),
 'published_within_species_sum_of_products':reported_S.tolist(),
 'recomputed_within_species_sum_of_products':S.tolist(),
 'max_absolute_matrix_difference':float(np.max(np.abs(S-reported_S))),
 'published_unscaled_coefficients':reported_w.tolist(),
 'recomputed_unscaled_coefficients':w.tolist(),
 'max_absolute_unscaled_coefficient_difference':float(np.max(np.abs(w-reported_w))),
 'published_sepal_normalized_coefficients':reported_normal.tolist(),
 'recomputed_sepal_normalized_coefficients':w_norm.tolist(),
 'max_absolute_normalized_coefficient_difference':float(np.max(np.abs(w_norm-reported_normal))),
 'published_group_scores':{'setosa':original_a,'versicolor':original_b},
 'recomputed_group_scores':{'setosa':observed_a,'versicolor':observed_b},
 'qualifier':'Numerical reproduction of published summary statistics and discriminant, not a rerun of absent 1936 code; computed from the scikit-learn Iris copy.'
}
(root/'historical_replication_results.json').write_text(json.dumps(result,indent=2))
for k in ['max_absolute_mean_difference','max_absolute_matrix_difference','max_absolute_unscaled_coefficient_difference','max_absolute_normalized_coefficient_difference','recomputed_sepal_normalized_coefficients','recomputed_group_scores']:
 print(k,result[k])
assert np.allclose(d,reported_d,atol=1e-10)
assert np.allclose(S,reported_S,atol=1e-8)
assert np.max(np.abs(w_norm-reported_normal)) < 0.001
