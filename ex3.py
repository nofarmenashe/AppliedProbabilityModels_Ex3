import sys


def get_articles_from_file(filename): # from ex2
    file = open(filename, "r")
    lines = file.readlines()
    articles = []
    articleContent = ""
    for line in lines:
        if line[:6] != '<TRAIN' and line[:5] != '<TEST':
            articleContent += line
        elif articleContent != "":
            articles.append(articleContent)
            article = ""

    articles.append(article)
    return articles


def get_all_words_in_articles(articles): # from ex2
    words = []
    for article in articles:
        for word in article.split():
            words.append(word)

    return words


def split_articles_to_clusters(articles):
    clustered_articles_array = []

    for i in range(NUM_OF_CLUSTERS):  # init clusters array
        clustered_articles_array.append([])

    for i, article in enumerate(articles):  # split articles to clusters
        clustered_articles_array[i % NUM_OF_CLUSTERS].append(article)

    return clustered_articles_array


def M_Step():
    


if __name__ == "__main__":
    development_set_filename = sys.argv[1]

    NUM_OF_CLUSTERS = 9

    developmentArticles = get_articles_from_file(development_set_filename)
    development_set_words = get_all_words_in_articles(developmentArticles)

    initClusteredArticles = split_articles_to_clusters(developmentArticles)
    alpha,
