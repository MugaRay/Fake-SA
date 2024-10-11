import nltk
from nltk.corpus import stopwords
from nltk.stem import  WordNetLemmatizer
import re

"""
Used fro pre-processing and cleaning datasets
"""

# Download NLTK resources if not already present
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('punkt_tab')

lemma = WordNetLemmatizer()

def normalize(document: str) -> str:
    return document.lower()

def remove_emoji(string: str) -> str:
    emoji_pattern = re.compile(
        r"["
        r"\U0001F600-\U0001F64F"  # emoticons
        r"\U0001F300-\U0001F5FF"  # symbols & pictographs
        r"\U0001F680-\U0001F6FF"  # transport & map symbols
        r"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        r"\U00002702-\U000027B0"
        r"\U000024C2-\U0001F251"
        r"]+", 
        flags=re.UNICODE
    )
    return emoji_pattern.sub(r' ', string)

def remove_unwanted(document: str) -> str:
    # Remove user mentions, URLs starting with http, https, or www, hashtags, and numbers
    document = re.sub(r"(@[A-Za-z0-9_]+)|((http|https|www)\S+)|(#\S+)|(\d+)", "", document)

    # Remove links that start with something other than http or www
    document = re.sub(r"\b(?!http|www)\w+\.com\S*\b", "", document)

    # Remove emoji's
    document = remove_emoji(document)

    # Remove unwanted punctuation
    document = re.sub(r'[^\w\s]', '', document)

    # Remove double spaces
    document = re.sub(r'\s+', ' ', document).strip()

    return document

def remove_stopwords(tokens: list, tweet=False) -> list: # type: ignore
    if tweet:
        unwantedWords = {"post", 'replies', 'media', 'likes', 
                         "views", "lte", 'view', "li", "vo", "4g", "cell c", 
                         "real411", "follow", ':', "tweet", "tweets", 
                         "repost", "reposts", "rt", "reply", 
                         "replies", "quote", "bookmarks"}
        stopwords_set = set(stopwords.words("english")).union(unwantedWords)
    else:
         stopwords_set = set(stopwords.words("english"))
    return [token for token in tokens if token not in stopwords_set]


def lemmatize(tokens: list, lemmatizer: WordNetLemmatizer) -> list:
    return [lemmatizer.lemmatize(token, pos='v') for token in tokens]



def remove_single_letters(text):
    words = nltk.word_tokenize(text)
    # Filter out single character tokens
    filtered_words = [word for word in words if len(word) > 1]
    return ' '.join(filtered_words)


def clean_text(document: str, lemmatizer=lemma, tweet=False) -> str: # type: ignore
    document = normalize(document)
    document = remove_unwanted(document)
    document = remove_single_letters(document)
    tokens = document.split()
    tokens = remove_stopwords(tokens, tweet)
    tokens = lemmatize(tokens, lemmatizer)
    
    return " ".join(tokens)


if __name__ == "__main__":

    """This is a test below """
    example = """20:06
    +
    Vo)) LTE all z
    Post
    Lisa the First
    Follow
    @Nelly_thefirst
    Pauli Van Wyk lives everyday under
    the weight of her own lies and
    propaganda about EFF and VBS. It is
    her wettest dream to see the black
    people unrepresented. She is a relic of
    apartheid Media. She has now become
    the victim of her own propaganda and
    far from reality.
    7:46 . 12 Jul 24 . 742 Views
    13 Reposts 32 Likes
    LI
    @lekoloaneManam2
    @lekolo...
    .
    12h
    Replying to @Nelly_thefirst
    She is a devil worshipper
    2
    22
    3
    101
    Lisa the First @Nelly_thefirst . 11h
    She is obsessed
    1
    LV
    ild 28
    Show replies
    Show probable spam
    Post your reply
    <"""

    print(clean_text(example))