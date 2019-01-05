# Writen By:
# Yuval Maymon - 315806299
# Nofar Menashe - 205486210

import sys
import collections
import numpy as np
import multiprocessing
import math
import pickle

import csv
import matplotlib.pyplot as plt
import pandas as pd

NUM_OF_CLUSTERS = 9
LAMBDA = 0.05
EPSILON = 0.001
K = 10
STOP_CRITERIA = 50


def save_to_file(dict, filename):
    with open(filename, 'wb') as handle:
        pickle.dump(dict, handle)


def read_from_file(filename):
    with open(filename, 'rb') as handle:
        obj = pickle.loads(handle.read())
    return obj


def get_words_in_file(filename):
    file = open(filename, "r")
    lines = file.readlines()
    words = []
    for line in lines:
        words.extend(line.split())
    return words


def save_results_graph(data, datatype):
    plt.figure()
    plt.plot(range(len(data)), data)
    plt.xlabel('epoch')
    plt.ylabel(datatype)
    plt.savefig(datatype + '.png')


def get_articles_from_file(filename):
    file = open(filename, "r")
    lines = file.readlines()

    articles = []
    topics = []
    articleContent = ""

    for line in lines:
        if line[:6] == '<TRAIN':
            if articleContent != "":
                articles.append(articleContent)
                articleContent = ""
            topics.append(line.split(">")[0].split()[2:])

        else:
            articleContent += line

    articles.append(articleContent)
    print(topics)

    return articles, topics


def get_all_words_in_articles(articles):
    words = []
    for article in articles:
        for word in article.split():
            words.append(word)

    return words


def remove_rare_words(wordsDictionary):
    return {w: wordsDictionary[w] for w
            in wordsDictionary
            if wordsDictionary[w] >= 4}


class Article:
    def __init__(self, articleIndex, articleString, tags, vocabulary):
        articleWords = articleString.split()
        filteredWords = [word for word in articleWords if word in vocabulary]

        self.articleCounter = collections.Counter(filteredWords)
        self.length = len(filteredWords)
        self.tags = tags

        self.wt = np.zeros(NUM_OF_CLUSTERS)
        self.wt[articleIndex % NUM_OF_CLUSTERS] = 1

class EMModel:
    def __init__(self, articles, articlesTags, vocabulary, topics):
        self.alpha = None
        self.P = None
        self.articles = []

        self.topics = topics
        self.vocabulary = collections.Counter(vocabulary)

        for t, article in enumerate(articles):
            self.articles.append(Article(t, article, articlesTags[t], vocabulary))

    def E_step(self):
        print("start E step")
        for article in self.articles:
            z = self.get_z(article)
            m = max(z)

            denominator = sum([math.exp(z[i] - m) for i in range(NUM_OF_CLUSTERS) if z[i] - m >= -K])
            for i in range(NUM_OF_CLUSTERS):
                if z[i] - m < -K:
                    article.wt[i] = 0
                else:
                    article.wt[i] = math.exp(z[i] - m) / denominator

        print("end E step")

    def get_z(self, article):
        z = np.zeros(NUM_OF_CLUSTERS)
        for i in range(NUM_OF_CLUSTERS):
            z[i] = np.log(self.alphas[i]) + sum([n * np.log(self.P[i][k]) for (k, n) in article.articleCounter.items()])
        return z

    def M_step(self):
        print("start M step")

        self.set_alpha()

        manager = multiprocessing.Manager()
        Ps = manager.dict()
        calc_Pi_processes = [multiprocessing.Process(target=self.calculate_pi, args=(i, Ps)) for i in
                             range(NUM_OF_CLUSTERS)]

        for Pi_process in calc_Pi_processes:
            Pi_process.start()

        for Pi_process in calc_Pi_processes:
            Pi_process.join()

        self.P = dict(Ps)
        print("end M step")

    def calculate_alpha_i(self, i):
        articleSumOfW = sum([atricle.wt[i] for atricle in self.articles])

        alpha_i = float(articleSumOfW) / len(self.articles)

        return EPSILON if alpha_i == 0 else alpha_i

    def set_alpha(self):
        alphas = [self.calculate_alpha_i(i) for i in range(NUM_OF_CLUSTERS)]
        sumOfAlphas = sum(alphas)
        self.alphas = [alpha / sumOfAlphas for alpha in alphas]

    def calculate_pi(self, i, P):
        P_is = {}
        for k in self.vocabulary:
            enumerator_sum = 0
            denumerator_sum = 0

            for article in self.articles:
                enumerator_sum += article.wt[i] * article.articleCounter[k]
                denumerator_sum += article.wt[i] * article.length

            P_is[k] = (float(enumerator_sum) + LAMBDA) / (denumerator_sum + (len(self.vocabulary) * LAMBDA))

        P[i] = P_is

    def calculate_likelihood(self):
        articlesLikelihood = []
        for article in self.articles:
            z_is = self.get_z(article)
            m_t = max(z_is)

            exponents = [math.exp(z_is[i] - m_t) for i in range(NUM_OF_CLUSTERS) if z_is[i] - m_t >= -1 * K]
            articlesLikelihood.append(m_t + math.log(sum(exponents)))

        return sum(articlesLikelihood)

    def calculate_mean_perplexity(self):
        sumOfPerplexities = 0

        for article in self.articles:
            prediction = np.argmax(article.wt)
            sumOfWordsProbabilities = 0
            for k, n in article.articleCounter.items():
                PofPerdictedCluster = (self.P[prediction][k] * article.length + LAMBDA) / \
                                      (article.length + len(self.vocabulary) * LAMBDA)
                sumOfWordsProbabilities += np.log(PofPerdictedCluster) * n

            sumOfPerplexities += np.exp(sumOfWordsProbabilities / -article.length)

        return sumOfPerplexities / len(self.articles)

    def create_confusion_matrix(self):
        matrix = []
        for i in range(NUM_OF_CLUSTERS):
            matrix.append({topic: 0 for topic in self.topics})
            for article in self.articles:
                if np.argmax(article.wt) == i:
                    print(article.tags)
                    for tag in article.tags:
                        matrix[i][tag] += 1

        lines = [("{0},{1}".format(str(i), (",".join([str(c) for c in row.values()]))))
                 for i, row in enumerate(matrix)]
        lines.insert(0, "," + ",".join(self.topics))

        fh = open("matrix.csv", "w")
        fh.write("\n".join(lines))
        fh.close()
        # matrix = [[i+1].extend(row) for i, row in enumerate(matrix)]
        # headers = ['index'].extend(self.topics)
        # print(headers)
        # with open('confusionMatrix.csv', 'w', newline='') as file:
        #     writer = csv.DictWriter(file, fieldnames=headers)
        #     writer.writeheader()
        #     writer.writerows(matrix)


if __name__ == "__main__":
    development_set_filename = sys.argv[1]
    topics_set_filename = sys.argv[2]

    developmentArticles, developmentArticlesTopics = get_articles_from_file(development_set_filename)
    developmentWords = get_all_words_in_articles(developmentArticles)

    wordsCounter = collections.Counter(developmentWords)
    wordsCounter = remove_rare_words(wordsCounter)

    vocabulary = wordsCounter.keys()

    topics = get_words_in_file(topics_set_filename)

    EM = EMModel(developmentArticles, developmentArticlesTopics, vocabulary, topics)

    logLikelihoodArray = []
    meanPerplexityArray = []

    EM.M_step()
    EM.create_confusion_matrix()

    while True:  # change to threshold
        print("epoch #" + str((len(logLikelihoodArray) + 1)))

        if len(logLikelihoodArray) > 3 and logLikelihoodArray[-1] - logLikelihoodArray[-3] < STOP_CRITERIA:
            print("Stop algorithm !")
            break

        EM.E_step()
        EM.M_step()

        logLikelihood = EM.calculate_likelihood()
        perplexity = EM.calculate_mean_perplexity()
        print(logLikelihood)

        logLikelihoodArray.append(logLikelihood)
        meanPerplexityArray.append(perplexity)

        EM.create_confusion_matrix()

    save_results_graph(logLikelihoodArray, "likelihood")
    save_results_graph(meanPerplexityArray, "preplexity")
