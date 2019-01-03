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


def initialize_EM(articles):
    clusteredArticlesArray = []

    for i in range(NUM_OF_CLUSTERS): # init clusters array
        clusteredArticlesArray.append([])

    for i, article in enumerate(articles): # split articles to clusters
        clusteredArticlesArray[i % NUM_OF_CLUSTERS].append(article)

    return clusteredArticlesArray


if __name__ == "__main__":
    development_set_filename = sys.argv[1]

    NUM_OF_CLUSTERS = 9

    developmentArticles = get_articles_from_file(development_set_filename)
    development_set_words = get_all_words_in_articles(developmentArticles)

    initClusteredArticles = initialize_EM(developmentArticles)
    print(len(developmentArticles))
    print(len(initClusteredArticles))
    print(len(initClusteredArticles[0]))
    print(len(initClusteredArticles[1]))
    print(len(initClusteredArticles[2]))
    print(len(initClusteredArticles[3]))
    print(len(initClusteredArticles[4]))
    print(len(initClusteredArticles[5]))
    print(len(initClusteredArticles[6]))
    print(len(initClusteredArticles[7]))
    print(len(initClusteredArticles[8]))
