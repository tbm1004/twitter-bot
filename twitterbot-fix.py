import os
import pickle
import random
import re
import twitter
import nltk
from collections import defaultdict
from bayesText import *
from visual import *

def connect():
    api = twitter.Api(
        consumer_key='',
        consumer_secret='',
        access_token_key='',
        access_token_secret='',
        tweet_mode='extended')
    if api.VerifyCredentials() is None:
        raise RuntimeError('Invalid twitter credentials')
    return api


def pickle_cached(fname_tmpl):
    def f(g):
        def wrapped(*args, **kwargs):
            root = os.path.dirname(__file__)
            fname = os.path.join(root, fname_tmpl.format(*args, **kwargs))
            if not os.path.exists(fname):
                res = g(*args, **kwargs)
                with open(fname, 'wb') as fh:
                    pickle.dump(res, fh)
            with open(fname, 'rb') as fh:
                return pickle.load(fh)
        return wrapped
    return f

@pickle_cached('timeline-{1}.cached.pickle')
def timeline(api, screen_name):
    print("Loading and Caching New Tweets")
    res = []
    while True:
        print(".", end="")
        more = api.GetUserTimeline(
            screen_name=screen_name,
            count=280,
            max_id=res[-1].id - 1 if res else None)
        if not more:
            break
        res.extend(more)
    print("\n")
    return res


def sanitize(txt):
    txt = re.sub('&.*;', ' ', txt)
    txt = txt.lower()
    chars = '.,!?()"…'
    trans = str.maketrans(chars, ' '*len(chars))
    if "rt @" in txt:
        txt = ""
        return
    #print(txt.translate(trans))
    return txt.translate(trans)

    

def wordmap(tweets, user):
    newtweets = []
    for stuff in tweets:
        newstuff = stuff.full_text
        fixed = re.sub('https(.*)', '', str(newstuff))
        newtweets.append(str(fixed))
    res = defaultdict(list)
    for tweet in newtweets:
        words = sanitize(tweet)
        if(words):
            words = words.split()   
            for w0, w1 in zip(words, words[1:]):
                if w0.startswith('@') or w1.startswith('@'):
                    continue
                res[w0].append(w1)
    
#DONT DELETE
#Next step is to check if file exists instead of commenting
#Make more generic
# =============================================================================
#     counttrain = 0
#     counttest = 0
#     x = 0
#     y = 0
#     train = int(len(tweets) * .80)
#     test = len(tweets) - train
#     print("newtweets count: " + str(len(newtweets)))
#     print("train: " + str(train) + "\n" + "test: " + str(test))
#     while counttrain < train:
#         save_path = 'C:\\Users\\Taylor\\.spyder-py3\\programs\\project\\projTrain\\' + str(user)
#         fname = str(user) + str(x) + ".txt"
#         completename = os.path.join(save_path, fname)
#         f = open(completename,"w+",encoding="utf-8")
#         f.write(newtweets[counttrain])
#         f.close()
#         x+=1
#         counttrain += 1
#     print(counttrain)
#     while counttest < test:
#         save_path = 'C:\\Users\\Taylor\\.spyder-py3\\programs\\project\\projTest\\' + str(user)
#         fname = str(user) + str(y) + ".txt"
#         completename = os.path.join(save_path, fname)
#         f = open(completename,"w+",encoding="utf-8")
#         f.write(newtweets[counttrain])
#         f.close()
#         y+=1
#         counttest += 1
#         counttrain +=1
# =============================================================================
        
    return res


MAX_TWEET = 280


def should_end(last_word, tweet_length):
    MIN_TWEET = int(random.randrange(20,225))
    if tweet_length < MIN_TWEET:
        return False
    if tweet_length < MAX_TWEET:
        _, tag = nltk.pos_tag([last_word])[0]
        return tag.startswith('NN') or tag.startswith('VV')
    return True


def generate(wm):
    def get(word):
        if word in wm:
            source = wm[word]
        else:
            source = list(wm)
        return random.choice(source)
    res = [get(None)]
    while not should_end(res[-1], sum(map(len, res))):
        res.append(get(res[-1]))
    return res

def predict():
    baseDirectory = "../project/" #top level folder name
    trainingDir = baseDirectory + "projTrain/" #training folder name
    testDir = baseDirectory + "projTest/" #test folder name
    bT = BayesText(trainingDir, baseDirectory + "stopwords.txt")
    print("Running Test ...")
    bT.test(testDir)
    print("\nClassify post@classify" + ":", bT.classify(baseDirectory + "classify.txt"))

def run(inp):
    api = connect()
    usernames = []
    gentweet = {}
    usernames.append(inp)
    randomtweet = random.randint(0,len(usernames) - 1)
    for user in usernames:
        tweets = timeline(api, user)
        wm = wordmap(tweets, user)
        words = generate(wm)
        tweet = ' '.join(words).capitalize() + '.'
        gentweet[user] = tweet
    f = open("classify.txt","w+",encoding="utf-8")
    
    print("Tweet generated from: " + str(list(gentweet.keys())[randomtweet]))
    print("New Tweet: " + str(list(gentweet.values())[randomtweet]))
    
    f.write((list(gentweet.values())[randomtweet]))
    f.close()
    predict()
    cl = visual.words(inp)
    visual.cloud(cl)
    
if __name__ == '__main__':
    inp = input("Enter a user to generate a Tweet and Word Cloud: " )
    run(inp)
