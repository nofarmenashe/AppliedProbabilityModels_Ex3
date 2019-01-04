import sys
import collections
import numpy as np
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

    return float(sumOfWs) / numOfArticles


# remove rares from articles
# calculate articles_lengths
# S-L-E-E-P !-!-!

def calculate_n_s(articles):
    articlesCounters = []

    for index, article in enumerate(articles):
        article_words = article.split()
        articlesCounters[index] = collections.Counter(article_words)

    return articlesCounters


def calculate_p_i_k(w, i, k, articles_lengths, articlesCounters):
    enumerator_sum = 0
    denumerator_sum = 0

    for t in range(len(articles_lengths)):
        enumerator_sum += w[t][i] * articlesCounters[t][k]
        denumerator_sum += w[t][i] * articles_lengths[t]

    return float(enumerator_sum) / denumerator_sum


def calculate_p_is(w, i, articlesLengths, articlesCounters, vocabulary):
    P_is = {}
    for k in vocabulary:
        P_is[k] = calculate_p_i_k(w, i, k, articlesLengths, articlesCounters)

    return P_is


def initialize_EM_parameters(numOfArticles, articlesLengths, articlesCounters, vocabulary):
    w = []
    P = []
    n = []
    alphas = np.zeros(NUM_OF_CLUSTERS)

    for t in range(numOfArticles):
        i = t % NUM_OF_CLUSTERS
        w.append(np.zeros(NUM_OF_CLUSTERS))
        w[t][i] = 1

    for i in range(NUM_OF_CLUSTERS):
        alphas[i] = calculate_alpha_i(w, i, numOfArticles)
        P.append(calculate_p_is(w, i, articlesLengths, articlesCounters, vocabulary))

    return w, alphas, P


# def M_Step(cluster):
# not implemented


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


def E_step(alpha, P, numOfArticles, vocabulary, articlesCounters):
    updated_Ws = np.zeros(numOfArticles, NUM_OF_CLUSTERS)
    for t in range(numOfArticles):
        for i in range(NUM_OF_CLUSTERS):
            updated_Ws[t][i] = alpha[i] * np.product([
                np.power(P[i][k], articlesCounters[t][k])
                for k in vocabulary])
        sumOfClustersProbabilities = np.sum(w[t])
        updated_Ws[t] = [w_t_i / sumOfClustersProbabilities for w_t_i in w[t]]
    return updated_Ws


if __name__ == "__main__":
    # development_set_filename = sys.argv[1]
    #
    # NUM_OF_CLUSTERS = 9
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
    # vocabulary = read_from_file("vocabulary.txt")

    articlesLengths = read_from_file("articlesLengths.txt")
    articlesCounters = read_from_file("articlesCounters.txt")
    wordsCounter = read_from_file("wordsCounter.txt")
    rareWords = read_from_file("rareWords.txt")
    developmentWords = read_from_file("developmentWords.txt")
    developmentArticles = read_from_file("developmentArticles.txt")
    numOfArticles = read_from_file("numOfArticles.txt")
    vocabulary = wordsCounter.keys()

    print(articlesLengths)





    w, alpha, P = initialize_EM_parameters(numOfArticles, articlesLengths, articlesCounters, vocabulary)

    while (True):  # change to threshhold
        w = E_step(alpha, P, numOfArticles, vocabulary, articlesCounters)
        # alpha, P = M_step(w, articlesCounters)
