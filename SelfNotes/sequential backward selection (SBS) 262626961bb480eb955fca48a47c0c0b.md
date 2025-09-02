# sequential backward selection (SBS)

![Screenshot 2025-09-02 at 10.39.20.png](sequential%20backward%20selection%20(SBS)%20262626961bb480eb955fca48a47c0c0b/Screenshot_2025-09-02_at_10.39.20.png)

```python
from sklearn.base import clone
from itertools import combinations
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

class SBS():
	def __init__(self, estimator, k_features, scoring=accuracy_score, test_size=0.25, random_state=1):
		self.scoring = scoring
		self.estimator = clone(estimator)
		self.k_features = k_features
		self.test_size = test_size
		self.random_state = random_state
		
	def fit(self, X, y):
		X_train, X_test, y_train, y_test = \
			train_test_split(X, y, test_size=self.test_size, random_state.random_state)
			
			dim = X_train.shape[1]
			self.indices_ = tuple(range(dim))
			self.subsets = [self.indices_]
			score = self._calc_score(X_train, y_train, X_test, y_test, self.indices_)
			
			self.scores_ = [score]
			
			while dim > self.k_features:
				score = []
				subsets = []
				
				for p in combinations(self.indices_, r=dim - 1):
					score = self._calc_score(X_train, y_train, X_test, y_test, p)
					
					scores.append(score)
					subsets.append(p)
					
				best = np.argmax(score)
				self.indices_ = subset[best]
				self.subsets_.append(self.indices_)
				dim -= 1
				
				self.scores_.append(scores[best])
			self.k_score_ = self.scores_[-1]
			
			return self
			
		def transform(self, X):
			return X[:, self.indices_]
			
		def _calc_score(self, X_train, y_train, X_test, y_test, indices):
			self.estimator.fit(X_train[:, indices], y_train)
			y_pred = self.estimator.predict(X_test[:, indices])
			score = self.scoring(y_test, y_pred)
			return score
			
			
```

<aside>
💡

test set → **validation dataset**

</aside>

- (pg 166 (138))