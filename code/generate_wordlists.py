from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords

######################################################################################################


# cleans title of punctuation and sends to lower case (and optionally, stop words)
def titles_to_wordlist(title, remove_stopwords=False):
    # Function to convert a document to a sequence of words,
    # optionally removing stop words.  Returns a list of words.
    # No need to remove HTML
    #
    # 1. Remove non-letters (and delete hyphens for now... this will have to be revised later)
    title_text = re.sub("[^a-zA-Z0-9]", " ", title)
    #
    # 2. Convert words to lower case and split them
    words = title_text.lower().split()
    #
    # 3. Optionally remove stop words (false by default)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if w not in stops]
    #
    # 4. Return a list of words
    return words

######################################################################################################


# cleans content of html (and optionally, stop words)
def content_to_wordlist(content, remove_stopwords=False):
    # Function to convert a document to a sequence of words,
    # optionally removing stop words.  Returns a list of words.
    #
    # 1. Remove HTML
    content_text = BeautifulSoup(content, 'lxml').get_text()
    #
    # 2. Remove non-letters (and delete hyphens for now... this will have to be revised later)
    content_text = re.sub("[^a-zA-Z0-9]", " ", content_text)
    #
    # 3. Convert words to lower case and split them
    words = content_text.lower().split()
    #
    # 4. Optionally remove stop words (false by default)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if w not in stops]
    #
    # 5. Return a list of words
    return words

######################################################################################################


# splits tags into list of words
def tags_to_wordlist(tags):
    # Tags are already in lower case and grouped
    # by using hyphens to incidate multiple words
    #
    # 1. Split the tags
    words = tags.split()
    #
    # 2. Return a list of words
    return words


######################################################################################################
