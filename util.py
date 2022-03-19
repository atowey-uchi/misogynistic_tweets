import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import nltk
import string
from gensim import corpora, models
import gensim
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from gensim.utils import effective_n_jobs
import nltk
from nltk.stem.snowball import SnowballStemmer
from wordcloud import WordCloud, STOPWORDS

tweet_regex = "[^a-z\s]"
tweet_stop = ['like', 'co', 'https', 'amp', 'would', 'know', 'get', 'us', 'one',
              'go', 'ncpol', 'ginny', 'iam', 'ncga', 'ncpol','roaring', 'fape',
              'ftzoqz','gsmathrive', 'pge','wtfthere', 'camden', 'thv', 'yxgjw',
              'followup', 'bydoes', 'atgy', 'fewest', 'xbthe', 'dcvpobp', 'emw',
              'jpc', 'ovqe', 'kjsvzrtyou', 'heaxfynndon', 'jjyx',
              'kjqjpcatjtcongratulations', 'qyr', 'quhtge', 'iaw', 'hdg',
              'ankeny', 'lwiacckis', 'nrcc', 'nturtnh', 'ycbvysgchn', 'pixdsgw',
              'xgxensj', 'vjust', 'ijthanks', 'horsesinourhandsughhh', 'mnqi',
              'panies', 'rd', 'lnk', 'wsmsldrmsfi', 'dvejdlnh', 'bwell',
              'fzn', 'etjust', 'gutlessdo', 'curran', 'xuf', 'yearsthese',
              'horsesinourhandsughhh', 'oyndqu', 'nlast', 'ngv', 'lbeiwsktufgo',
              'jmmrfprsnlsuper', 'djcaicp', 'res', 'h', 'dhey', 'hiutlbncbkhey',
              'eynmqnwyp', 'vonmg', 'okfwqsixdb', 'zgft', 'os', 'qcuedasd',
              'itcay', 'pdzjgsusan', 'need', 'make', 'back', 'think', 'bwnvkfr',
              'rqq', 'wnmdgqkwy', 'zrrvpyfvz', 'bmvkfkktwzyour',
              'zdsyvhbvvktourists', 'tredyffrin', 'raniere', 'yhqxbsfmyt',
              'xbc', 'mitchand', 'pqkiwu', 'zhu', 'concurrences', 'qapbmcsbs',
              'fti', 'pzjfvpvmhu', 'vhpvyyy', 'hey', 'ob', 'mds', 'baugh',
              'priscilla', 'germaine', 'ross', 'ny', 'congressional',
              'election', 'maine', 'senate', 'state', 'voting', 'right', 'good',
              'country', 'votesafe', 'grassroots', 'latogether', 'equalpay',
              'organizers', 'gofundme', 'sela', 'southeastla', 'erick',
              'lakeshow', 'latinasareessentialit', 'latinasareessentialstill',
              'kjyh', 'lavotesvoters', 'mqgulbobby', 'people', 'want', 'left',
              'cobb', 'time', 'state', 'america', 'american', 'president',
              'state', 'biden', 'stefanik', 'shiptrump',
              'missourideservesbetter', 'plank', 'people', 'trump', 'vote',
              'court', 'want', 'time', 'take', 'election', 'let', 'going', 'u',
              'president', 'country', 'nothing', 'retire', 'graham', 'please',
              'az', 'pa', 'armenian', 'mi', 'senator', 'va', 'nv', 'fl',
              'swing', 'republican', 'senator', 'really', 'say', 'see', 'saw',
              'party', 'never', 'keep']
stopwords = nltk.corpus.stopwords.words('english') + tweet_stop


def main_cleaner(name, congress=False):
    """
    This function cleans data from our biographies of women in senate and congress
    Relevant csv: congress_women.csv, senate_women.csv
    """
    df = pd.read_csv("Data/" + name + ".csv")
    df[["state_name", "state"]] = df.state.str.split(" - ",
                                                     expand=True)
    df = df.rename(columns={"District": "district"})
    if congress == True:
        df["district"] = df["district"].apply(
            district_at_large)
    return df

def cleaner(name, feature):
    """
    This function cleans data from the GovTrack USA Website with \
    ideology and leadership scores
    Relevant csv: senate_ideology.csv, senate_leadership.csv, \
    congress_ideology.csv, congress_leadership.csv
    """
    df = pd.read_csv("Data/" + name + ".csv")
    df["name"] = df["name"].str.replace(r"^b'", "", regex=True)
    df["name"] = df["name"].str.replace(r"'", "", regex=True)
    df["name"] = df["name"].str.replace(r"\\xc3\\xa1", "a", regex=True)
    df["name"] = df["name"].str.replace(r"\\xc3\\xb3", "o", regex=True)
    df = df.rename(columns={"name": "Last Name"})
    df = df[["Last Name", feature, "state", "district"]]
    return df

def data_merge(main_df, ideology_df, leadership_df, criteria):
    """
    Once we have cleaned all our data, this function merges our dataframes
    """
    df_final = pd.merge(main_df, ideology_df, on=criteria, how="left")
    df_final = pd.merge(df_final, leadership_df, on=criteria, how="left")
    return df_final[['ID', 'Year', 'First Name', 'Middle Name', 'Last Name',
                     'party', 'Level', 'Position', 'state', 'race_ethnicity',
                     'state_name', 'ideology','leadership', "district"]]

def district_at_large(dist):
    """
    Some politicians have their district listed as "AL" (at-large)
    Cleans this value and replaces it with a zero
    """
    dist = str(dist)
    if dist == "AL":
        return 0
    else:
        dist = dist.replace("0", "")
        return int(dist)

def black_color_func(word,
                     font_size,
                     position,orientation,
                     random_state=None,
                     **kwargs):
    '''
    Changes our word cloud to a black and white wordcloud
    '''
    return("hsl(0,100%, 1%)")

# wordcloud function
def generate_wordcloud(dataframe, custom_stopwords=stopwords, comment_words=''):
    '''
    Plot a black & white wordcloud

    Inputs:
        dataframe: a dataframe with tweets
        custom_stopwords: additional stopwords to include
        comment_words: words to be a part of the word cloud
    Outputs:
        a wordcloud plot
    '''
    for val in dataframe["tweet"]:

        # typecaste each val to string
        val = str(val)

        # split the value
        tokens = val.split()

        # Converts each token into lowercase
        for i in range(len(tokens)):
            tokens[i] = tokens[i].lower()

        comment_words += " ".join(tokens)+" "

    wordcloud = WordCloud(width = 3000, height = 2000,
                max_words = 500,
                background_color ='white',
                stopwords = stopwords+custom_stopwords,
                min_font_size = 10).generate(comment_words)
    wordcloud.recolor(color_func = black_color_func)
    # plot the WordCloud image
    plt.figure(figsize = (15, 10))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
#     plt.tight_layout(pad = 0)

    plt.show()

def tokenize_tweets(text_series):
    '''
    Cleans + tokenizes Pandas series of strings from the Salem Witch dataset.

    Returns pandas series of lists of tokens
    '''
    clean = text_series.str.lower() \
                       .str.replace(tweet_regex,
                                    " ",
                                    regex=True)

    stop = nltk.corpus.stopwords.words('english') + tweet_stop

    tokenize = lambda text: [i for i in nltk.word_tokenize(text) \
                             if i not in stop]
    tokens = clean.apply(tokenize)
    return tokens

def prepare_data(tokens):
    '''
    Prepares Pandas series of lists of tokens for use within a \
    Gensim topic model

    Returns an id2word dictionary + bag of words corpus
    '''

    dictionary = corpora.Dictionary([i for i in tokens])

    bow_corpus = [dictionary.doc2bow(text) for text in tokens]

    return dictionary, bow_corpus

def fill_topic_weights(df_row, bow_corpus, ldamodel):
    '''
    Fill DataFrame rows with topic weights for topics in tweets.

    Modifies DataFrame rows *in place*.
    '''
    try:
        for i in ldamodel[bow_corpus[df_row.name]]:
            df_row[str(i[0])] = i[1]
    except:
        return df_row
    return df_row

def binary_encoder(val):
    """
    Encode some of our categorial variables in binary format for \
    regression modelling
    """
    if val == "Democrat" or val == "White" or val == "U.S. Representative":
        return 0
    else:
        return 1
