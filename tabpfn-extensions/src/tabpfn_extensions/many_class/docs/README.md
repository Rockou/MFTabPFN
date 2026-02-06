# ManyClassClassifier

This extension allows TabPFN to handle classification problems with more classes than TabPFN's default limit (typically 10 classes). It works through an error-correcting output code (ECOC) approach that:

1. Encodes the multi-class problem into multiple binary/small-class problems
2. Trains the base classifier on these simpler problems
3. Decodes the predictions back to the original multi-class space

## Basic Usage

```python
from tabpfn_extensions.many_class import ManyClassClassifier
from tabpfn import TabPFNClassifier  # or from tabpfn_client
import numpy as np

# Create a dataset with many classes (more than TabPFN's default limit)
X = np.random.rand(100, 5)
y = np.random.randint(0, 20, 100)  # 20 classes

# Create the base classifier
base_classifier = TabPFNClassifier()

# Wrap it with ManyClassClassifier
many_class_clf = ManyClassClassifier(
    estimator=base_classifier,
    alphabet_size=8,    # Size of subproblems (should be <= TabPFN's class limit)
    n_estimators=15     # Number of subproblems to create
)

# Use like any scikit-learn classifier
many_class_clf.fit(X, y)
y_pred = many_class_clf.predict(X)
```

## Features

- **Large Class Support**: Handle dozens or hundreds of classes
- **Scalable**: Computational complexity scales well with number of classes
- **Robust**: Error-correcting codes provide robustness
- **Backend Compatibility**: Works with both TabPFN and TabPFN-client
- **scikit-learn Compatible**: Follows scikit-learn classifier API

## Parameters

- **estimator**: Base classifier (must support `predict_proba`)
- **alphabet_size**: Size of each subproblem, typically <= base classifier's class limit
- **n_estimators**: Number of subproblems to create
- **n_estimators_redundancy**: How many estimators per class (higher = more robust)
- **codebook_type**: Type of encoding ("random" or "dense")

## Tips

- Increase `n_estimators` for better accuracy on many-class problems
- Decrease `alphabet_size` if your base classifier has a low class limit
- Use `codebook_type="dense"` for better performance with many classes

## Reference

For more detailed information, see the [example](../../../../examples/many_class/) or [source code](../many_class_classifier.py).
