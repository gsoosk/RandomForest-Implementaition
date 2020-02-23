<h1>Table of Contents<span class="tocSkip"></span></h1>
<div class="toc"><ul class="toc-item"><li><span><a href="#Training-Decision-Tree" data-toc-modified-id="Training-Decision-Tree-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Training Decision Tree</a></span></li><li><span><a href="#Making-Random-Forest" data-toc-modified-id="Making-Random-Forest-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Making Random Forest</a></span><ul class="toc-item"><li><span><a href="#Bootstaping" data-toc-modified-id="Bootstaping-2.1"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>Bootstaping</a></span><ul class="toc-item"><li><span><a href="#What-is-bootstraping-:" data-toc-modified-id="What-is-bootstraping-:-2.1.1"><span class="toc-item-num">2.1.1&nbsp;&nbsp;</span>What is bootstraping :</a></span></li></ul></li><li><span><a href="#Creating-Classifier-With-Bagging" data-toc-modified-id="Creating-Classifier-With-Bagging-2.2"><span class="toc-item-num">2.2&nbsp;&nbsp;</span>Creating Classifier With Bagging</a></span></li><li><span><a href="#Deleting-Every-Attribute" data-toc-modified-id="Deleting-Every-Attribute-2.3"><span class="toc-item-num">2.3&nbsp;&nbsp;</span>Deleting Every Attribute</a></span></li><li><span><a href="#Create-Random-Forest-Classifier" data-toc-modified-id="Create-Random-Forest-Classifier-2.4"><span class="toc-item-num">2.4&nbsp;&nbsp;</span>Create Random Forest Classifier</a></span></li></ul></li><li><span><a href="#Questions" data-toc-modified-id="Questions-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Questions</a></span><ul class="toc-item"><li><span><a href="#What-is-bootstrapping-and-what-is-it's-compact-on-variance-?" data-toc-modified-id="What-is-bootstrapping-and-what-is-it's-compact-on-variance-?-3.1"><span class="toc-item-num">3.1&nbsp;&nbsp;</span>What is bootstrapping and what is it's compact on variance ?</a></span></li><li><span><a href="#What-is-overfiting?-Why-decision-tree-overfit?-Bagging-trying-to-solve-what-problem?" data-toc-modified-id="What-is-overfiting?-Why-decision-tree-overfit?-Bagging-trying-to-solve-what-problem?-3.2"><span class="toc-item-num">3.2&nbsp;&nbsp;</span>What is overfiting? Why decision tree overfit? Bagging trying to solve what problem?</a></span></li><li><span><a href="#What-is-random-forest-and-bagging-relation?-Random-forest-trying-to-solve-what-problem?" data-toc-modified-id="What-is-random-forest-and-bagging-relation?-Random-forest-trying-to-solve-what-problem?-3.3"><span class="toc-item-num">3.3&nbsp;&nbsp;</span>What is random forest and bagging relation? Random forest trying to solve what problem?</a></span></li><li><span><a href="#Concolusion-from-accuracies" data-toc-modified-id="Concolusion-from-accuracies-3.4"><span class="toc-item-num">3.4&nbsp;&nbsp;</span>Concolusion from accuracies</a></span></li></ul></li></ul></div>

# Computer Assignment 4 - Ensemble Classification
**Farzad Habibi - 810195383**

In this project we will make a Random Forest classifier using `Bagging and Aggregation` methodology on Decision Tree classifier.


# Loading Data



```python
import pandas as pd
import gc
train_set = pd.read_csv('data.csv')
```

We select 20% of datas as test set. 


```python
from sklearn.model_selection import train_test_split
X, y = train_set.drop(['target'], axis=1), train_set['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
del X, y
```


```python
X_train.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 242 entries, 132 to 102
    Data columns (total 13 columns):
    age         242 non-null int64
    sex         242 non-null int64
    cp          242 non-null int64
    trestbps    242 non-null int64
    chol        242 non-null int64
    fbs         242 non-null int64
    restecg     242 non-null int64
    thalach     242 non-null int64
    exang       242 non-null int64
    oldpeak     242 non-null float64
    slope       242 non-null int64
    ca          242 non-null int64
    thal        242 non-null int64
    dtypes: float64(1), int64(12)
    memory usage: 26.5 KB


## Training Decision Tree


```python
from sklearn.tree import DecisionTreeClassifier
decision_tree_classifier = DecisionTreeClassifier()
decision_tree_classifier.fit(X_train, y_train)
```




    DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='gini',
                           max_depth=None, max_features=None, max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, presort='deprecated',
                           random_state=None, splitter='best')




```python
y_pred = decision_tree_classifier.predict(X_test)
```

We will compute accuracy for a single decision tree.


```python
from sklearn.metrics import accuracy_score
print(f'Accuracy for Decision Tree is \x1b[1;33m{accuracy_score(y_test, y_pred)}\x1b[0m')
```

    Accuracy for Decision Tree is [1;33m0.819672131147541[0m


## Making Random Forest

### Bootstaping

#### What is bootstraping : 

We can simply bootstraping by random selection between numbers. We select every samples as new train data.


```python
np.random.randint?
```

    Object `np.random.randint` not found.



```python
import numpy as np
np.random.seed(10)
samples_indidces = [np.random.randint(0, len(X_train), 150) for i in range(5)]
```

### Creating Classifier With Bagging

We implement `RandomForestClassifier` class. This class follow base estimator model of sklearn.
* In `fit` method we compute and make all of our decision trees according to our hyper-parameters in constructor.
* In `predict` method we predict values with each model, and then select maximum number repeated for every of them.


```python
from scipy.stats import mode 
class BaggingClassifier():
    def __init__(self, n_estimators=5, max_samples=150):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
    def fit(self, X, y):
        self.samples_indices = [np.random.randint(0, len(X), self.max_samples) for i in range(self.n_estimators)]
        self.models = [DecisionTreeClassifier().fit(X.iloc[sample_indices], y.iloc[sample_indices])
                       for sample_indices in self.samples_indices]
        return self
    def predict(self, X):
        self.all_predictions = np.array([model.predict(X) for model in self.models])
        val, count = mode(self.all_predictions, axis = 0) 
        return val.ravel()
```


```python
bagging_classifier = BaggingClassifier()
bagging_classifier.fit(X_train, y_train)
```




    <__main__.BaggingClassifier at 0x130ac0bd0>




```python
y_pred = bagging_classifier.predict(X_test)
```


```python
print(f'Accuracy for Our Bagging Classifier is \x1b[1;33m{accuracy_score(y_test, y_pred)}\x1b[0m')
```

    Accuracy for Our Bagging Classifier is [1;33m0.8524590163934426[0m


### Deleting Every Attribute
In this part we delete every attribiute and see what accuracy they got.


```python
max_accuracy = 0
max_col = ''
for col in X_train.columns:
    
    bagging_classifier = BaggingClassifier().fit(X_train.drop([col], axis=1), y_train)
    y_pred = bagging_classifier.predict(X_test.drop([col], axis=1))
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f'Accuracy for Bagging Classifier without \x1b[1;33m{col}\x1b[0m is \x1b[1;33m{accuracy}\x1b[0m')
    if accuracy > max_accuracy:
        max_accuracy = accuracy
        max_col = col

print(f'Best col to drop is \x1b[1;33m{max_col}\x1b[0m')
```

    Accuracy for Bagging Classifier without [1;33mage[0m is [1;33m0.8032786885245902[0m
    Accuracy for Bagging Classifier without [1;33msex[0m is [1;33m0.8524590163934426[0m
    Accuracy for Bagging Classifier without [1;33mcp[0m is [1;33m0.7377049180327869[0m
    Accuracy for Bagging Classifier without [1;33mtrestbps[0m is [1;33m0.8360655737704918[0m
    Accuracy for Bagging Classifier without [1;33mchol[0m is [1;33m0.8360655737704918[0m
    Accuracy for Bagging Classifier without [1;33mfbs[0m is [1;33m0.7868852459016393[0m
    Accuracy for Bagging Classifier without [1;33mrestecg[0m is [1;33m0.7540983606557377[0m
    Accuracy for Bagging Classifier without [1;33mthalach[0m is [1;33m0.7540983606557377[0m
    Accuracy for Bagging Classifier without [1;33mexang[0m is [1;33m0.8360655737704918[0m
    Accuracy for Bagging Classifier without [1;33moldpeak[0m is [1;33m0.8032786885245902[0m
    Accuracy for Bagging Classifier without [1;33mslope[0m is [1;33m0.7868852459016393[0m
    Accuracy for Bagging Classifier without [1;33mca[0m is [1;33m0.8852459016393442[0m
    Accuracy for Bagging Classifier without [1;33mthal[0m is [1;33m0.8688524590163934[0m
    Best col to drop is [1;33mca[0m


### Create Random Forest Classifier 

In this part we select 5 cols and fit our bagging classifer.


```python
class RandomForestClassifier(BaggingClassifier):
    def __init__(self, n_estimators=5, max_samples=150, n_cols=5):
        self.n_cols = n_cols
        super().__init__(n_estimators, max_samples)
    def fit(self, X, y):
        self.cols = X.columns[np.random.randint(0, len(X.columns), self.n_cols)]
        return super().fit(X[self.cols], y)
    def predict(self, X):
        return super().predict(X[self.cols])
        
```


```python
random_classifier = RandomForestClassifier().fit(X_train, y_train)
```


```python
y_pred = random_classifier.predict(X_test)
```


```python
print(f'Accuracy for Our Random Forest Classifier is \x1b[1;33m{accuracy_score(y_test, y_pred)}\x1b[0m')
```

    Accuracy for Our Random Forest Classifier is [1;33m0.8524590163934426[0m


## Questions 

### What is bootstrapping and what is it's compact on variance ?

Bootstrapping means selection of data to train on estimators with replacement. Bootstrapping introduces a bit more diversity in the subsets that each predictor is trained on, so bagging ends up with a slightly higher bias than pasting, but this also means that predictors end up being less correlated so the ensemble’s variance is reduced. Overall, bagging often results in better models, which explains why it is generally preferred. However, if you have spare time and CPU power you can use cross-validation to evaluate both bagging and pasting and select the one that works best.

### What is overfiting? Why decision tree overfit? Bagging trying to solve what problem?

Overfitting happens when a model learns the detail and noise in the training data to the extent that it negatively impacts the performance of the model on new data. This means that the noise or random fluctuations in the training data is picked up and learned as concepts by the model.

Decision tree overfit because in high depth we have very few data points corresponding to our rules. And our model will overfited on our data. 

Bagging solve the overfiting problem. Bagging seperate input in fewer input points and train a decision tree with low depth factor on it. Then it doesn't train on all data points and can not overfit. 

### What is random forest and bagging relation? Random forest trying to solve what problem?

Random forset consists of bagging with some reduction in features. On the other hand it make data smaller in features dimension and do bagging on sub models. 

Random forest also like bagging tries to stop overfiting. Because it got more data features and make our tree simpler. On the other hand, random forest reduce variance of data points. It also reduce execuation and train time.

### Concolusion from accuracies

* As we see because of ensemble type of bagging, it's got better accuracy from decision tree. Cause decision tree can be overfit. 
* Random forest also can get better accuracy if we have chance to select better features. Because it makes variance of data points fewer and tries to not overfit. 
