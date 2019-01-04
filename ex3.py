import sys
import collections
import numpy as np


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


def calculate_p_i_k(w, i, k, articles_lengths, numOfArticles, n_s):
    enumerator_sum = 0
    denumerator_sum = 0

    for t in range(numOfArticles):
        enumerator_sum += w[t][i] * n_s[t][k]
        denumerator_sum += w[t][i] * articles_lengths[t]

    return float(enumerator_sum)/ denumerator_sum


def calculate_p_i(w, i, vocabulary, articles_lengths, numOfArticles, n_s):
    P_i = []
    for k in vocabulary:
        P_i[k] = calculate_p_i_k(w, i, k, articles_lengths, numOfArticles, n_s)

    return P_i


def initialize_EM_parameters(articles, vocabulary):
    w = []
    n = []
    alphas = np.zeros(NUM_OF_CLUSTERS)

    for t, article in enumerate(articles):
       i = t % NUM_OF_CLUSTERS
       w[t] = np.zeros(NUM_OF_CLUSTERS)
       w[t][i] = 1

    for i in range(NUM_OF_CLUSTERS):
        alphas[i] = calculate_alpha_i(w, i, len(articles))
        P[i] = calculate_p_i(w, i, vocabulary, articles_lengths, numOfArticles, n_s)

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

if __name__ == "__main__":
    development_set_filename = sys.argv[1]

    NUM_OF_CLUSTERS = 9

    developmentArticles = get_articles_from_file(development_set_filename)
    development_set_words = get_all_words_in_articles(developmentArticles)

    wordsCounter = collections.Counter(development_set_words)
    wordsCounter = remove_rare_words(wordsCounter)

    rareWords = get_rare_words(wordsCounter)

    articlesCounters = [collections.Counter(get_all_words_in_articles([article])) for article in developmentArticles]
    articlesCounters = remove_rare_words_from_articles_counters(rareWords, articlesCounters)

    vocabulary = wordsCounter.keys()

    w, alpha, P = initialize_EM_parameters(developmentArticles, vocabulary)