# NP_ML
## Introduction
Classical machine learning algorithms with pure python numpy.

And a tool to help you understand the root of all the classical algorithms instead of blindly using tools like sklearn or tensorflow.

## Directory<a name="directory"></a>
- [Introduction](#introduction)
- [Directory](#directory)
- [Algorithm list](#algorithm-list)
  * [Classical ML](#classical-ml)
    + Perceptron
    + K Nearest Neightbor (KNN)
    + Naive Bayes
    + Decision Tree
    + Random Forest
    + Logistic Regression
    + SVM
    + AdaBoost
    + HMM
    + KMeans
  * [NLP](#nlp)
    + LDA
  * [Time Series Analysis](#time-series-analysis)
    + AR
- [Usage](#usage)
  * Installation
  * Examples for *Statistical Learning Method*(《统计学习方法》)
- [Reference](#reference)
## Algorithm List<a name="algorithm-list"></a>
### Classical ML<a name="classical-ml"></a>
- Perceptron

For perceptron, the example used the [UCI/iris dataset](https://archive.ics.uci.edu/ml/datasets/iris). Since the basic perceptron is a binary classifier, the example used the data for versicolor and virginica. Also, since the iris dataset is not linear separable, the result may vary much.
<p align="center">
    <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/4/41/Iris_versicolor_3.jpg/1024px-Iris_versicolor_3.jpg" height="200">
    <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/9/9f/Iris_virginica.jpg/1024px-Iris_virginica.jpg" height="200">
</p>
<p align="center">
    Figure: versicolor and virginica. Hard to distinguish... Right?
</p>

```
$ cd examples
$ python perceptron_primary.py
```

<p align="center">
    <img src="https://raw.githubusercontent.com/zhuzilin/NP_ML/master/examples/figures/perceptron.png" width="640">
</p>
<p align="center">
    Perceptron result on the Iris dataset.
</p>

- K Nearest Neightbor (KNN)

For KNN, the example also used the UCI/iris dataset.

```
$ cd examples
$ python knn.py
```

<p align="center">
    <img src="https://raw.githubusercontent.com/zhuzilin/NP_ML/master/examples/figures/knn.png" width="640">
</p>
<p align="center">
    KNN result on the Iris dataset.
</p>

- Naive Bayes

For naive bayes, the example used the [UCI/SMS Spam Collection Dataset](https://www.kaggle.com/uciml/sms-spam-collection-dataset) to do spam filtering.

```
$ cd examples
$ python naive_bayes.py
```

For this example only, for tokenizing, nltk is used. And the result is listed below:

```
preprocessing data...
100%|############################################################################| 5572/5572 [00:00<00:00, 8656.12it/s]
finish preprocessing data.

100%|###########################################################################| 1115/1115 [00:00<00:00, 55528.96it/s]
accuracy:  0.9757847533632287
```

We got 97.6% accuracy! That's nice!

And we try two example, a typical ham and a typical spam. The result show as following.

```
example ham:
Po de :-):):-):-):-). No need job aha.
predict result:
ham

example spam:
u r a winner U ave been specially selected 2 receive 澹1000 cash or a 4* holiday (flights inc) speak to a live operator 2 claim 0871277810710p/min (18 )
predict result:
spam
```

- Decision Tree

For decision tree, the example used the UCI/tic-tac-toe dataset. The input is the status of 9 block and the result is whether x win.
<p align="center">
    <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/3/32/Tic_tac_toe.svg/2000px-Tic_tac_toe.svg.png" width="200">
</p>
<p align="center">
    tic tac toe.
</p>

```
$ cd examples
$ python decision_tree.py
```
Here, we use ID3 and CART to generate a one layer tree.

For the ID3, we have:
```
root
├── 4 == b : True
├── 4 == o : False
└── 4 == x : True
accuracy = 0.385
```
And for CART, we have: 
```
root
├── 4 == o : False
└── 4 != o : True
accuracy = 0.718
```
In both of them, feature_4 is the status of the center block. We could find out that **the center block matters!!!** And in ID3, the tree has to give a result for 'b', which causes the low accuracy.

- Random Forest
- Logistic Regression
- SVM
- AdaBoost
- HMM
- Kmeans

For kmeans, we use the [make_blob()](http://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_blobs.html#sklearn.datasets.make_blobs) function in sklearn to produce toy dataset.

```
$ cd examples
$ python kmeans.py
```

<p align="center">
    <img src="https://raw.githubusercontent.com/zhuzilin/NP_ML/master/examples/figures/kmeans.png" width="640">
</p>
<p align="center">
    Kmeans result on the blob dataset.
</p>

### NLP<a name="nlp"></a>
- LDA
### Time Series Analysis<a name="time-series-analysis"></a>
- AR

## Usage<a name="usage"></a>
- Installation

If you want to use the visual example, please install the package by:
```
  $ git clone https://github.com/zhuzilin/NP_ML
  $ cd NP_ML
  $ python setup.py install
```

- Examples for *Statistical Learning Method*(《统计学习方法》)

For example: 

```
  $ cd example/StatisticalLearningMethod
  $ python adaboost.py
```
## Reference<a name="reference"></a>
Classical ML algorithms was validated by naive examples in [*Statistical Learning Method*(《统计学习方法》)](https://www.amazon.com/%E7%BB%9F%E8%AE%A1%E5%AD%A6%E4%B9%A0%E6%96%B9%E6%B3%95%EF%BC%88%E7%BB%9F%E8%AE%A1%E5%AD%A6%E4%B9%A0%E6%96%B9%E6%B3%95-%E7%BB%9F%E8%AE%A1%E8%87%AA%E7%84%B6%E8%AF%AD%E8%A8%80%E5%A4%84%E7%90%86-%E7%AC%AC2%E7%89%88-%E5%85%B12%E6%9C%AC%E5%A5%97%E8%A3%85%EF%BC%89-Chinese-ebook/dp/B01M8KB8FF/ref=sr_1_1?ie=UTF8&qid=1521303280&sr=8-1&keywords=%E7%BB%9F%E8%AE%A1%E5%AD%A6%E4%B9%A0%E6%96%B9%E6%B3%95)

Time series models was validated by example in [Bus 41202](http://faculty.chicagobooth.edu/ruey.tsay/teaching/bs41202/sp2017/)
