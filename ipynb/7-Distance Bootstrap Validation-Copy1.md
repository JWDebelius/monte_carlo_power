The goal with this notebook (for now) is to test and see if I can handle power better by bootstrapping at the OTU level, and then performing the power calculation. I also want to know if I need to bootstrap the OTU table, or if its suffecient to bootstrap and rarify.

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
>>> from scipy.spatial.distance import braycurtis, jaccard
>>> import skbio
>>> from skbio.diversity.beta import unweighted_unifrac, weighted_unifrac
...
>>> import machivellian.beta as beta
...
>>> # import machivellian.plot as plot
... # import machivellian.summarize as summarize
... # from machivellian.bootstrap import bootstrap_permanova
... from machivellian.power import subsample_power, confidence_bound
>>> from machivellian.effects import z_power, z_effect
...
>>> % matplotlib inline
```

```python
>>> np.random.seed(5)
```

We start by loading the OTU table, and convert it to a dataframe for easier handling. We exclude singleton OTUs and samples with less than 1250 sequences per sample.

```python
>>> bmi_otu = biom.load_table('./data/turnbaugh_twins/219_otu_table.biom')
...
>>> bmi_otu.filter(lambda v, id_, md: v.sum() > 1, axis='observation', inplace=True)
>>> bmi_otu.filter(lambda v, id_, md: v.sum() > 1250, axis='sample', inplace=True)
>>> bmi_otu = pd.DataFrame(bmi_otu.matrix_data.todense(),
...                        columns=bmi_otu.ids('sample'),
...                        index=bmi_otu.ids('observation')).T
```

We'll also load the mapping file, and filter so the map only includes the samples in the OTU table.

```python
>>> bmi_map = pd.read_csv('./data/turnbaugh_twins/77_prep_534_qiime_20161216-090019.txt', sep='\t', dtype=str)
>>> bmi_map.set_index('#SampleID', inplace=True)
>>> bmi_map = bmi_map.loc[bmi_otu.index]
```

To look at phylogenetic distance, we also need to load a phylogenetic tree.

```python
>>> bmi_tree = skbio.TreeNode.read('./data/turnbaugh_twins/97_otu_filt.tree')
```

Let's compare the effect of BMI on the microbiome between lean and obese individuals.

```python
>>> bmi_map['obesitycat'].value_counts()
Obese         186
Lean           57
Overweight     24
Name: obesitycat, dtype: int64
```

```python
>>> sample_ids = [bmi_map.loc[bmi_map['obesitycat'] == g].index for g in ['Lean', 'Obese']]
```

Let's also define the test using a partial function. We'll use the same OTU table, map and tree. We'll also focus on unweighted UniFrac distance.

```python
>>> test = lambda x: beta.bootstrap_permanova(np.hstack(x),
...                                           obs=bmi_otu,
...                                           grouping=bmi_map['obesitycat'],
...                                           metric=unweighted_unifrac,
...                                           depth=1000,
...                                           metric_kws={'tree': bmi_tree,
...                                                       'otu_ids': bmi_otu.columns},
...                                           )[0]['p-value']
```

```python
>>> %%time
...
... test(np.hstack(sample_ids))
/Users/jdebelius/Repositories/monte_carlo_power/machivellian/beta.py:28: VisibleDeprecationWarning: using a non-integer number instead of an integer will result in an error in the future
  for (count, id_) in zip(list(counts), list(ids))])
```

Let's apply the function to our lean and obese individuals with a rarefaction depth of 1000 sequences/sample. We'll look at between 10 and 50 samples with

```python
>>> counts = np.arange(10, 51, 10)
>>> power = subsample_power(test=test,
...                         samples=sample_ids,
...                         counts=np.arange(10, 51, 10),
...                         num_iter=100,
...                         num_runs=3
...                         )
```

```python
>>> help(subsample_power)
```

```python

```
