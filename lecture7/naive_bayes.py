# -*- coding: utf8 -*-

from __future__ import division
from fractions import Fraction as Frac

import logging

logging.basicConfig(level=logging.INFO)

__author__ = 'yfwz100'

__all__ = ['PlainNaiveBayes', 'LaplaceNaiveBayes']


class PlainNaiveBayes(object):
    def __init__(self):
        super(PlainNaiveBayes, self).__init__()
        self._label_cnt = dict()
        self._all_label_cnt = 0
        self._label_word_cnt = dict()
        self._word_cnt = dict()
        self._all_word_cnt = 0
        self._vocabulary = set()

    def train(self, instances):
        for label, instance in instances:
            if label not in self._word_cnt:
                self._word_cnt[label] = dict()
            wc = self._word_cnt[label]
            for word in instance:
                if word in wc:
                    wc[word] += 1
                else:
                    wc[word] = 1
                if label not in self._label_word_cnt:
                    self._label_word_cnt[label] = 0
                self._label_word_cnt[label] += 1
                self._vocabulary.add(word)

            if label not in self._label_cnt:
                self._label_cnt[label] = 0
            self._label_cnt[label] += 1

        self._all_label_cnt = sum(self._label_cnt.itervalues())
        self._all_word_cnt = sum(self._label_word_cnt.itervalues())

    def get_word_prob(self, label, word):
        word = word.capitalize()
        if word in self._word_cnt[label]:
            return Frac(self._word_cnt[label][word], self._label_word_cnt[label])
        else:
            return Frac(0)

    def get_label_prob(self, label):
        if label in self._label_cnt:
            return Frac(self._label_cnt[label], self._all_label_cnt)
        else:
            return Frac(0)

    def get_post_prob(self, instance, label=None):
        result = dict()
        for label in self._word_cnt.iterkeys():
            result[label] = self.get_label_prob(label)
            for word in instance:
                result[label] *= self.get_word_prob(label, word)
        regularization = sum(result.itervalues())
        if label:
            logging.debug('%f/%f' % (result[label], regularization))
            return Frac(result[label], regularization)
        else:
            return {k: Frac(v, regularization) for k, v in result.iteritems()}

    def __str__(self):
        return 'Plain Naive Bayes.'


class LaplaceNaiveBayes(PlainNaiveBayes):
    def __init__(self, laplace=1):
        super(LaplaceNaiveBayes, self).__init__()
        self._laplace = laplace

    def get_word_prob(self, label, word):
        word = word.capitalize()
        numerator = self._laplace
        denominator = self._laplace * len(self._vocabulary)
        if word in self._word_cnt[label]:
            numerator += self._word_cnt[label][word]
            denominator += self._label_word_cnt[label]
        return Frac(numerator, denominator)

    def get_label_prob(self, label):
        numerator = self._laplace
        denominator = self._laplace * len(self._label_cnt)
        if label in self._label_cnt:
            numerator += self._label_cnt[label]
            denominator += self._all_label_cnt
        logging.debug("Label Prob. = %f / %f" % (numerator, denominator))
        return Frac(numerator, denominator)

    def __str__(self):
        return "Laplace Naive Bayes (alpha=%d)" % self._laplace


def test(sentences):
    instances = map(lambda d: (d[0], [w.capitalize() for w in d[1].split(' ')]), sentences)

    # for clf in [PlainNaiveBayes(), LaplaceNaiveBayes(2)]
    for clf in [LaplaceNaiveBayes(2)]:
        clf.train(instances)

        print clf
        print 'P(Spam) = %s' % clf.get_label_prob('Spam')
        print 'P("secret"|Spam) = %s' % clf.get_word_prob('Spam', 'secret')
        print 'P("secret"|Ham) = %s' % clf.get_word_prob('Ham', 'secret')
        print 'P(Spam|"Sports") = %s' % clf.get_post_prob(['sports'], 'Spam')
        print 'P(Spam|"Secret is secret") = %s' % clf.get_post_prob("Secret is secret".split(' '), 'Spam')
        print 'P(Spam|"Today is a secret day") = %s' % clf.get_post_prob("Today is a secret day".split(' '), 'Spam')


if __name__ == '__main__':
    # test([
    #     ('Spam', 'Offer is secret'),
    #     ('Spam', 'Click secret link'),
    #     ('Spam', 'Secret sports link'),
    #     ('Ham', 'Play sports today'),
    #     ('Ham', 'Went play sports'),
    #     ('Ham', 'Secret sports event'),
    #     ('Ham', 'Sport is today'),
    #     ('Ham', 'Sport costs money')
    # ])
    test([
        ('Spam', 'Offer is secret'),
        ('Spam', 'Click secret link below'),
        ('Spam', 'Secret sports link here'),
        ('Ham', 'Play sports today'),
        ('Ham', 'Went play sports'),
        ('Ham', 'Secret sports event tomorrow'),
        ('Ham', 'Sports are hold today'),
        ('Ham', 'Sports cost much money')
    ])