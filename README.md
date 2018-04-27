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
    + KNN
    + Naive Bayes
    + Decision Tree
    + Random Forest
    + Logistic Regression
    + SVM
    + AdaBoost
    + HMM
  * [NLP](#nlp)
    + LDA
  * [Time Series Analysis](#time-series-analysis)
    + AR
- [Usage](#usage)
  * Installation
  * Direct Usage
- [Reference](#reference)
## Algorithm List<a name="algorithm-list"></a>
### Classical ML<a name="classical-ml"></a>
- Perceptron

For perceptron, the example used the UCI/iris dataset. Since the basic perceptron is a binary classifier, the example used the data for versicolor and virginica.
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
    Figure: Perceptron result on the Iris dataset.
</p>

- KNN
- Naive Bayes
- Decision Tree

For decision tree, the example used the UCI/tic=tac-toe dataset. The input is the status of 9 block and the result is whether x win.
<p align="center">
    <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/3/32/Tic_tac_toe.svg/2000px-Tic_tac_toe.svg.png" width="200">
</p>
<p align="center">
    Figure: tic tac toe.
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

- Direct Usage

If you just what to try each model, there is a naive example in each model file. So you can just run the model_name.py file in the np_ml/model_name/ file.

For example:

```
  $ git clone https://github.com/zhuzilin/NP_ML
  $ cd NP_ML/np_ml/adaboost
  $ python adaboost.py
```
## Reference<a name="reference"></a>
All classical ML algorithms was validated by example in [*Statistical Learning Method*(《统计学习方法》)](https://www.amazon.com/%E7%BB%9F%E8%AE%A1%E5%AD%A6%E4%B9%A0%E6%96%B9%E6%B3%95%EF%BC%88%E7%BB%9F%E8%AE%A1%E5%AD%A6%E4%B9%A0%E6%96%B9%E6%B3%95-%E7%BB%9F%E8%AE%A1%E8%87%AA%E7%84%B6%E8%AF%AD%E8%A8%80%E5%A4%84%E7%90%86-%E7%AC%AC2%E7%89%88-%E5%85%B12%E6%9C%AC%E5%A5%97%E8%A3%85%EF%BC%89-Chinese-ebook/dp/B01M8KB8FF/ref=sr_1_1?ie=UTF8&qid=1521303280&sr=8-1&keywords=%E7%BB%9F%E8%AE%A1%E5%AD%A6%E4%B9%A0%E6%96%B9%E6%B3%95)

Time series models was validated by example in [Bus 41202](http://faculty.chicagobooth.edu/ruey.tsay/teaching/bs41202/sp2017/)
