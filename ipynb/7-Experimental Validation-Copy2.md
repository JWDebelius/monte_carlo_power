# Experimental Validation

The goal of this notebook is to apply the power method validated in notebooks 1 - X to real data. We've chosen to perform a meta analysis of the effect size associated with a lean Body Mass Index () to an obese BMI ().

We've collected data from the following studies...

All studies were processed through Qiita ([www.qiita.ucsd.edu](www.qiita.ucsd.edu)). All samples were picked closed reference using SortMeRNA [[*]()] against the August 2013 released of Greengenes [*]().

```python
>>> import os
>>> from functools import partial
...
>>> import biom
>>> import matplotlib.pyplot as plt
>>> import numpy as np
>>> import pandas as pd
>>> import seaborn as sn
>>> import scipy
>>> import skbio
...
>>> import machivellian.plot as plot
>>> import machivellian.summarize as summarize
>>> from machivellian.bootstrap import bootstrap_permanova
>>> from machivellian.power import subsample_power, confidence_bound
>>> from machivellian.effects import z_power, z_effect
...
>>> from machivellian.effects import z_power
```

We

```python
>>> bmi_map = pd.read_csv('./data/merged_otu_table_and_mapping_bmi/merged_bmi_mapping_final2.txt',
...                       sep='\t', dtype=str)
>>> bmi_map.set_index('#SampleID', inplace=True)
```

```python
>>> bmi_map = bmi_map.loc[bmi_map['COUNTRY'] == 'GAZ:United States of America']
>>> bmi_map = bmi_map.loc[bmi_map['AGE'].apply(lambda x: float(x) > 20)]
```

```python
>>> bmi_dm = skbio.DistanceMatrix.read('data/merged_otu_table_and_mapping_bmi/1k/unweighted_unifrac_dm.txt')
```

```python
>>> map_ids = set(bmi_map.index)
>>> dm_ids = set(bmi_dm.ids)
```

```python
>>> rep_ids = map_ids.intersection(dm_ids)
```

```python
...
>>> bmi_map = bmi_map.loc[rep_ids]
>>> bmi_dm = bmi_dm.filter(rep_ids)
```

```python
>>> bmi_map['AGE'] = bmi_map['AGE'].astype(float)
```

```python
>>> bmi_map.groupby(('original_study', 'bmi_group_coded')).mean().unstack()
>>> # ['Normal', 'Overweight', 'Obese']
                             AGE                      
bmi_group_coded           Normal      Obese Overweight
original_study                                        
COMBO_Wu               29.386364  39.222222  29.956522
HMP                    26.903614  28.714286  27.145833
Turnbaugh_mz_dz_twins  28.000000  28.117647  29.312500
Yatsunenko_GG          38.559322  44.305556  46.285714
amish_Fraser           43.303030  53.082090  45.112245
```

```python
>>> bmi_map.columns
Index(['BarcodeSequence', 'LinkerPrimerSequence', 'BMI', 'bmi_group_binned',
       'bmi_group_coded', 'original_study', 'combined_study_bmi_group',
       'merged_category_bmi', 'merged_weight_cats_study', 'PCR_PRIMERS',
       'TARGET_SUBFRAGMENT', 'AGE', 'AGE_CHECK', 'ELEVATION', 'LONGITUDE',
       'COUNTRY', 'SEQUENCING_METH', 'SAMPLE_CENTER', 'Description_duplicate',
       'ReversePrimer', 'COLLECTION_DATE', 'SEX', 'FAMILY_RELATIONSHIP_GG',
       'STUDY_CENTER', 'EXPERIMENT_CENTER', 'bmi_group_amish', 'RUN_CENTER',
       'LATITUDE', 'Description'],
      dtype='object')
```

```python
>>> summaries = {}
```

```python
>>> for study_name, study_map in bmi_map.groupby('original_study'):
...     if study_name == 'COMBO_Wu':
...         continue
...     print(study_name)
...     study_dm = bmi_dm.filter(study_map.index)
...     study_groups = [study_map.groupby('bmi_group_coded').groups[group] for group in ['Normal', 'Obese']]
...     min_group_size = min([len(group) for group in study_groups])
...     study_counts = np.arange(15, min_group_size*1.1, 5)
...
...     def study_test(ids):
...         """A quick wraper for bootstrap_peramnaova"""
...         ids = np.hstack(ids)
...         res = bootstrap_permanova(ids, dm=study_dm, grouping=study_map['bmi_group_coded'])
...         return res['p-value']
...
...     study_pwr = subsample_power(test=study_test,
...                                 samples=study_groups,
...                                 counts=study_counts,
...                                 num_iter=500,
...                                 num_runs=3,
...                                 alpha=0.05
...                                 )
...
...     study_eff = z_effect(study_counts, study_pwr, alpha=0.05)
...
...     summaries[study_name] = {'study_pwr': study_pwr,
...                              'study_eff': study_eff.reshape(study_pwr.shape),
...                              'study_count': study_counts,
...                              }
HMP
```

```python
>>> study_eff
(0.34826452263810492, 0.00011436907099306287, 6)
```

```python
>>> summarie
```

```python
>>> for study_name, study_summary in summaries.items():
...     print('%s: %1.2f +/- %1.2f' % (study_name,
...           np.nanmean(study_summary['study_eff']),
...           confidence_bound(study_summary['study_eff'])))
```

```python
>>> plt.plot(summaries['Yatsunenko_GG']['study_count'],
...          summaries['Yatsunenko_GG']['study_eff'].T, 'o')
```

```python
>>> summaries['Yatsunenko_GG']['study_eff']
array([[ 0.23450755,  0.27578986,  0.3104443 ,  0.32870689,  0.35223206,
         0.38565073,  0.4085863 ,  0.4382363 ,  0.44973182],
       [ 0.23450755,  0.28656457,  0.31157971,  0.3334177 ,  0.33357403,
         0.38736182,  0.38007182,  0.41252595,  0.43671876],
       [ 0.24665236,  0.29840778,  0.30240247,  0.32396509,  0.36157012,
         0.38994236,  0.42026227,  0.41989613,  0.43402234]])
```

```python
>>> z_effect(summaries['Yatsunenko_GG']['study_count'],
...          summaries['Yatsunenko_GG']['study_pwr']).reshape(summaries['Yatsunenko_GG']['study_pwr'].shape)
array([[ 0.12893125,  0.1843581 ,  0.22866525,  0.25405317,  0.28311615,
         0.32099871,  0.34763179,  0.38040978,  0.3945964 ],
       [ 0.12893125,  0.1951328 ,  0.22980066,  0.25876397,  0.26445812,
         0.3227098 ,  0.31911731,  0.35469942,  0.38158334],
       [ 0.14107605,  0.20697602,  0.22062341,  0.24931137,  0.29245421,
         0.32529034,  0.35930776,  0.3620696 ,  0.37888691]])
```

```python

```
