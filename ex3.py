import sys
import collections
import numpy as np
import math
import pickle


def save_to_file(dict, filename):
    with open(filename, 'wb') as handle:
        pickle.dump(dict, handle)


def read_from_file(filename):
    with open(filename, 'rb') as handle:
        obj = pickle.loads(handle.read())
    return obj


def get_articles_from_file(filename):  # from ex2
    file = open(filename, "r")
    lines = file.readlines()
    articles = []
    articleContent = ""
    for line in lines:
        if line[:6] != '<TRAIN' and line[:5] != '<TEST':
            articleContent += line
        elif articleContent != "":
            articles.append(articleContent)
            articleContent = ""

    articles.append(articleContent)
    return articles


def get_all_words_in_articles(articles):
    words = []
    for article in articles:
        for word in article.split():
            words.append(word)

    return words


def get_w_of_article(article):
    wordsDict = collections.counter(article.split())
    return remove_rare_words(wordsDict)


def get_rare_words(wordsDictionary):
    rareWords = []

    for word in wordsDictionary:
        if wordsDictionary[word] <= 3:
            rareWords.append(word)

    return rareWords


def remove_rare_words(wordsDictionary):
    return {w: wordsDictionary[w] for w
            in wordsDictionary
            if wordsDictionary[w] >= 4}


def calculate_alpha_i(w, i, numOfArticles):
    sumOfWs = 0

    for t in range(numOfArticles):
        sumOfWs += w[t][i]

    alpha = float(sumOfWs) / numOfArticles

    if alpha == 0:
        return EPSILON
    return alpha


def calculate_n_s(articles):
    articlesCounters = []

    for index, article in enumerate(articles):
        article_words = article.split()
        articlesCounters[index] = collections.Counter(article_words)

    return articlesCounters


def calculate_p_i_k(w, i, k, articles_lengths, articlesCounters, vocabularySize):
    enumerator_sum = 0
    denumerator_sum = 0

    for t in range(len(articles_lengths)):
        enumerator_sum += w[t][i] * articlesCounters[t][k]
        denumerator_sum += w[t][i] * articles_lengths[t]

    return (float(enumerator_sum) + LAMBDA) / (denumerator_sum + (vocabularySize * LAMBDA))


def calculate_p_is(w, i, articlesLengths, articlesCounters, vocabulary):
    P_is = {}
    for k in vocabulary:
        P_is[k] = calculate_p_i_k(w, i, k, articlesLengths, articlesCounters, len(vocabulary))

    return P_is


def initialize_EM_parameters(numOfArticles, articlesLengths, articlesCounters, vocabulary):
    w = []
    P = []
    alphas = np.zeros(NUM_OF_CLUSTERS)

    for t in range(numOfArticles):
        i = t % NUM_OF_CLUSTERS
        w.append(np.zeros(NUM_OF_CLUSTERS))
        w[t][i] = 1

    for i in range(NUM_OF_CLUSTERS):
        alphas[i] = calculate_alpha_i(w, i, numOfArticles)
        P.append(calculate_p_is(w, i, articlesLengths, articlesCounters, vocabulary))

    return w, alphas, P


def remove_rare_words_from_articles_counters(rare_words, articlesCounters):
    articlesCountersWithoutRares = []

    for articleCounter in articlesCounters:
        for word in rare_words:
            del articleCounter[word]

        articlesCountersWithoutRares.append(articleCounter)

    return articlesCountersWithoutRares


def get_articles_lengths_from_counters(articlesCounters):
    articlesLengths = []
    for articleCounter in articlesCounters:
        articlesLengths.append(sum(articleCounter.values()))

    return articlesLengths


def E_step(alphas, P, numOfArticles, vocabulary, articlesCounters):
    updated_Ws = []
    for t in range(numOfArticles):
       updated_Ws.append(calculate_w_t(t, alphas, P, articlesCounters, vocabulary))
    return updated_Ws


def calculate_w_t(t, alphas, P, articlesCounters, vocabulary):
    wt = np.zeros(NUM_OF_CLUSTERS)
    z_is = []

    for i in range(NUM_OF_CLUSTERS):
        z_is.append(calculate_z_i(i, t, alphas, P, vocabulary, articlesCounters))
    m = max(z_is)

    exponents = np.zeros(NUM_OF_CLUSTERS)
    for i in range(NUM_OF_CLUSTERS):
        if z_is[i] - m >= -1 * K:
            exponents[i] = math.exp(z_is[i] - m)

    denumerator = sum(exponents)

    for i in range(NUM_OF_CLUSTERS):
        if z_is[i] - m < -1 * K:
            wt[i] = 0
        else:
            wt[i] = exponents[i] / denumerator

    return wt


def calculate_z_i(i, t, alphas, P, vocabulary, articlesCounters):
    sum = 0
    for k in vocabulary:
        sum += articlesCounters[t][k] * np.log(P[i][k])

    return np.log(alphas[i]) + sum


def M_step(w, vocabulary, numOfArticles, articlesLengths, articlesCounters):
    alphas = np.zeros(NUM_OF_CLUSTERS)
    P = []

    for i in range(NUM_OF_CLUSTERS):
        alphas[i] = calculate_alpha_i(w, i, numOfArticles)
        P.append(calculate_p_is(w, i, articlesLengths, articlesCounters, vocabulary))

    sumOfAlphas = sum(alphas)
    normalizedAlphas = [alpha_i / sumOfAlphas for alpha_i in alphas]

    return normalizedAlphas, P


def calculate_likelihood(alphas, P, vocabulary, articlesCounters, numOfArticles):
    articlesLikelihood = np.zeros(numOfArticles)

    for t in range(numOfArticles):
        z_is = []
        for i in range(NUM_OF_CLUSTERS):
            z_is.append(calculate_z_i(i, t, alphas, P, vocabulary, articlesCounters))

        m_t = max(z_is)

        exponents = np.zeros(NUM_OF_CLUSTERS)
        for i in range(NUM_OF_CLUSTERS):
            if z_is[i] - m_t >= -1 * K:
                exponents[i] = math.exp(z_is[i] - m_t)

        articlesLikelihood[t] = m_t + math.log(sum(exponents))

        return sum(articlesLikelihood)


if __name__ == "__main__":
    NUM_OF_CLUSTERS = 9
    LAMBDA = 0.05
    EPSILON = 0.001
    K = 10
    STOP_CRITERIA = 10

    # development_set_filename = sys.argv[1]
    #
    # developmentArticles = get_articles_from_file(development_set_filename)
    # numOfArticles = len(developmentArticles)
    #
    # developmentWords = get_all_words_in_articles(developmentArticles)
    #
    # wordsCounter = collections.Counter(developmentWords)
    # rareWords = get_rare_words(wordsCounter)
    #
    # wordsCounter = remove_rare_words(wordsCounter)
    #
    # articlesCounters = [collections.Counter(get_all_words_in_articles([article])) for article in developmentArticles]
    # articlesCounters = remove_rare_words_from_articles_counters(rareWords, articlesCounters)
    #
    # articlesLengths = get_articles_lengths_from_counters(articlesCounters)
    #
    # vocabulary = wordsCounter.keys()
    #
    # w, alphas, P = initialize_EM_parameters(numOfArticles, articlesLengths, articlesCounters, vocabulary)
    #
    # save_to_file(w, "w.txt")
    # save_to_file(alphas, "alphas.txt")
    # save_to_file(P, "P.txt")
    # # save_to_file(vocabulary, "vocabulary.txt")
    # save_to_file(articlesLengths, "articlesLengths.txt")
    # save_to_file(articlesCounters, "articlesCounters.txt")
    # save_to_file(wordsCounter, "wordsCounter.txt")
    # save_to_file(rareWords, "rareWords.txt")
    # save_to_file(developmentWords, "developmentWords.txt")
    # save_to_file(developmentArticles, "developmentArticles.txt")
    # save_to_file(numOfArticles, "numOfArticles.txt")


    w = read_from_file("w.txt")
    alphas = read_from_file("alphas.txt")
    P = read_from_file("P.txt")
    articlesLengths = read_from_file("articlesLengths.txt")
    articlesCounters = read_from_file("articlesCounters.txt")
    wordsCounter = read_from_file("wordsCounter.txt")
    rareWords = read_from_file("rareWords.txt")
    developmentWords = read_from_file("developmentWords.txt")
    developmentArticles = read_from_file("developmentArticles.txt")
    numOfArticles = read_from_file("numOfArticles.txt")
    vocabulary = wordsCounter.keys()

    logLikelihoodArray = []

    while True:  # change to threshold
        print("epoch #" + str((len(logLikelihoodArray) + 1)))

        if len(logLikelihoodArray) > 3 and logLikelihoodArray[-1] - logLikelihoodArray[-3] < STOP_CRITERIA:
            print("Stop algorithm !")
            break

        w = E_step(alphas, P, numOfArticles, vocabulary, articlesCounters)
        print("done e step")

        alpha, P = M_step(w, vocabulary, numOfArticles, articlesLengths, articlesCounters)
        print("done m step")

        logLikelihood = calculate_likelihood(alphas, P, vocabulary, articlesCounters, numOfArticles)
        print(logLikelihood)

        logLikelihoodArray.append(logLikelihood)

