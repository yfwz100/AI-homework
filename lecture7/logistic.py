# -*- coding: utf-8 -*-

import numpy.random as rng
from numpy import zeros
import theano
import theano.tensor as T


class Classifier(object):
    """ The abstract class of classifier.
    """

    def __init__(self, **kwargs):
        """ Initialize the classifier.
        :param kwargs: the keyword arguments.
        """
        super(Classifier, self).__init__()
        self._predict = None
        self._default_props = None
        self.__props = kwargs

    def __getattr__(self, attr):
        return self.__props.get(attr, self._default_props.get(attr))

    def fit(self, X, Y):
        """ Fit the data.
        :param X: the training matrix.
        :param Y: the label matrix.
        :return: the classifier itself.
        """
        pass

    def predict(self, X):
        """ Predict the label of given instances.
        :param X: the testing matrix.
        :return: the vector of labels.
        """
        if self._predict:
            return self._predict(X)
        else:
            raise RuntimeError("No model is trained.")


class LogisticRegression(Classifier):
    """ Binary linear classifier
    """

    def __init__(self, **kwargs):
        super(LogisticRegression, self).__init__(**kwargs)
        self._default_props = dict(
            training_steps=10000,
            learning_rate=0.01
        )

    def fit(self, X, Y):
        # `N` is the number of instances; `feats` is the number of features.
        N, feats = X.shape

        # define the symbols used in the classifier.
        x = T.matrix('x')
        y = T.vector('y')
        w = theano.shared(rng.randn(feats), name='w')
        b = theano.shared(0., name='b')

        # Logistic Probability.
        p_1 = 1 / (1 + T.exp(-T.dot(x, w) - b))

        # the prediction result matrix.
        prediction = p_1 > 0.5

        # the prediction function.
        self._predict = theano.function(inputs=[x], outputs=prediction)

        # cross entropy cost function.
        xent = -y * T.log(p_1) - (1 - y) * T.log(1 - p_1)
        cost = xent.mean() + 0.01 * (w ** 2).sum()
        gw, gb = T.grad(cost, [w, b])

        # the training function.
        train = theano.function(
            inputs=[x, y],
            outputs=[prediction, xent],
            updates=((w, w - self.learning_rate * gw), (b, b - self.learning_rate * gb))
        )

        for i in range(self.training_steps):
            _, self._err = train(X, Y)


class SoftmaxRegression(Classifier):
    """ The extended linear classifier for multiple labels.
    """

    def __init__(self, **kwargs):
        super(SoftmaxRegression, self).__init__(**kwargs)
        self._default_props = dict(
            training_steps=10000,
            learning_rate=0.01
        )

    def fit(self, X, Y):
        feats = X.shape[1]
        L = max(Y) + 1

        # define the symbols.
        x = T.matrix('x')
        y = T.lvector('y')
        w = theano.shared(
            value=zeros((feats, L), dtype=theano.config.floatX),
            name='w',
            borrow=True
        )
        b = theano.shared(
            value=zeros((L,), dtype=theano.config.floatX),
            name='b',
            borrow=True
        )

        # Probability of y given x
        p_y_given_x = T.nnet.softmax(T.dot(x, w) + b)

        prediction = T.argmax(p_y_given_x, axis=1)

        cost = -T.mean(T.log(p_y_given_x)[T.arange(y.shape[0]), y])
        error = T.mean(T.neq(prediction, y))

        gw, gb = T.grad(cost, [w, b])

        self._predict = theano.function(inputs=[x], outputs=prediction)

        train = theano.function(
            inputs=[x, y],
            outputs=[error, cost],
            updates=[
                (w, w - self.learning_rate * gw),
                (b, b - self.learning_rate * gb)
            ]
        )

        for i in range(self.training_steps):
            self._err, self._cost = train(X, Y)


def test_logistic(X, Y):
    clf = LogisticRegression(training_steps=10)
    clf.fit(X, Y)
    print(clf.predict(X))
    print(Y)


def test_softmax(X, Y):
    clf = SoftmaxRegression()
    clf.fit(X, Y)
    print(clf.predict(X))
    print(Y)


if __name__ == '__main__':
    X = rng.randn(10, 2)
    Y = rng.randint(size=10, low=0, high=2)
    print('### Softmax ###')
    test_softmax(X, Y)
    print('### Logistic ###')
    test_logistic(X, Y)
