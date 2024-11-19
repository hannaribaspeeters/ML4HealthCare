---
title: 'Report: AI for Global Health using Natural Language Processing'

---

# Report: AI for Global Health using Natural Language Processing

## Part 1: Data pre-processing

Q1: **[What are the methods used ?]** 
Our preprocessing pipeline has three steps. In the first step, pretokenization cleaning, we remove any nan values and duplicates from the dataset. This results in a final dataset of 568773 samples. Further we apply some cleaning to the raw texts. This includes removal of urls, old retweet labels as well as contractions. Contractions are expanded using the [contractions python library](https://github.com/kootenpv/contractions). 

**[Code Snipped]**  Pretokenization Processing
```
def old_style_retweet_removal(text):
  text = re.sub('RT @[\w_]+:','', text)
  text = re.sub(r'^RT[\s]+', '', text)
  return text

def url_removal(text):
  return re.sub(r'http\S+','', text) # r=raw \S=string

def clean_hashtags(text):
  return re.sub(r'#', '', text)

def contraction_expansion(text):
  def fix(word):
    try:
      return contractions.fix(word)
    except Exception as e:
      return word # this is required as it crashes with some non-english chars e.g. İbrahimÇelikkol 
  return ' '.join([fix(word) for word in text.split() if word])

def stopwords_removal(text):
  return " ".join([word for word in text.split() if word not in stop_words])

def pretokenization_cleaning(text):
  text = old_style_retweet_removal(text)
  text = url_removal(text)
  text = clean_hashtags(text)
  text = single_numberic_removal(text)
  text = contraction_expansion(text)
  return text

```
In a second step we tokenize the texts. This means splitting the text into a list of individual words or tokens. The *NLTK* library provides a tweet specific [TweetTokenizer](https://www.nltk.org/api/nltk.tokenize.casual.html#:~:text=Twitter-aware%20tokenizer) that handles the removal of twitter handles and casing. Further it can replace repeated character sequences of length 3 or greater with sequences of length 3 (e.g. "amaaaaaazing!!!!!" is parsed to \["amaaazing", "!", "!", "!"]). The TweetTokenizer will also detect emojis and allocate them a single token.

In the last step we preprocess the individual tokens. This includes the removal of stopwords, invisible unicode characters, punctuations and numbers. Finally we lemmatize the words. In lemmatization we reduce the words to the grammatical base form. For example "has" is lemmatized to "have" or "cars" to "car". For this we first compute the positional tags for each token. A positional tag is the grammatical entity of a token, like for example "car" is a *NOUN* or "have" is a *VERB*. This will improve the capabilities of the lemmatizer. We use the [WordNetLemmatizer](https://www.nltk.org/api/nltk.stem.wordnet.html?highlight=wordnetlemmatizer) model as well as the [default positional tagger](https://www.nltk.org/api/nltk.tag.pos_tag.html) that nltk providess.

**[Code Snipped]**  Posttokenization Processing
```
def stopwords_removal(tokens):
  return [token for token in tokens if token not in stop_words]

def punctuation_removal(tokens):
    return [token for token in tokens if token not in  ['...', '!', '(', ')', '+', '-', '[', ']', '{', '}', '|', ';', ':', "'", '"', '\\', ',', '<', '>', '.', '/', '?', '@', '#', '$', '£', '%', '^', '&', '*', '_', '~', '“', '”', '…', '‘', '’']]

def remove_invisible_unicode_specials(tokens):
    return [token for token in tokens if token not in [
        u'\u2060', # Word Joiner
        u'\u2061', # FUNCTION APPLICATION
        u'\u2062', # INVISIBLE TIMES
        u'\u2063', # INVISIBLE SEPARATOR
        u'\u2064', # INVISIBLE PLUS
        u'\u2066', # LEFT - TO - RIGHT ISOLATE
        u'\u2067', # RIGHT - TO - LEFT ISOLATE
        u'\u2068', # FIRST STRONG ISOLATE
        u'\u2069', # POP DIRECTIONAL ISOLATE
        u'\u206A', # INHIBIT SYMMETRIC SWAPPING
        u'\u206B', # ACTIVATE SYMMETRIC SWAPPING
        u'\u206C', # INHIBIT ARABIC FORM SHAPING
        u'\u206D', # ACTIVATE ARABIC FORM SHAPING
        u'\u206E', # NATIONAL DIGIT SHAPES
        u'\u206F', # NOMINAL DIGIT SHAPES
        u'\u200B', # Zero-Width Space
        u'\u200C', # Zero Width Non-Joiner
        u'\u200D', # Zero Width Joiner
        u'\u200E', # Left-To-Right Mark
        u'\u200F', # Right-To-Left Mark
        u'\u061C', # Arabic Letter Mark
        u'\uFEFF', # Byte Order Mark
        u'\u180E', # Mongolian Vowel Separator
        u'\u00AD'  # soft-hyphen
        ]]

def numbers_removal(tokens):
    return [token for token in tokens if not token.isdigit()]

def to_wordnet_pos(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:          
        return None

def lemmatize(tokens):
    tokens_tags = [(token, to_wordnet_pos(tag)) for token, tag in nltk.pos_tag(tokens)]
    
    lemmatized_tokens = []
    for word, tag in tokens_tags:
        if tag is None:
            lemmatized_tokens.append(word)
        else:        
            lemmatized_tokens.append(lemmatizer.lemmatize(word, tag))
    return lemmatized_tokens

lemmatizer = WordNetLemmatizer()
```
**[Quantitative analysis & Question answer]** Compared to other types of free-text, tweets have distinct features such as hashtags, mentions, URLs, and emojis. With this in mind, we aim to address these elements in order to establish a consistent format for tweets. From the table below, we can see that a large number of tweets contain links and mentions. As they have a high granularity they may negatively influece the performance of standard NLP models, which is why we remove all of them. Because tweets are very similar to spoken language we expect a lot of contractions (e.g. 'isn't'), to combat this we expand them.
| Substring | Count  |
| -------- | -------- | 
| RT @     | 489     | 
| http     | 270216  | 
| @        | 235447  | 

Further we decided to also remove any punctuations, numbers and stopwords (e.g. "i", "me", "but", ...) as they have no benefit for traditional NLP methods. Upon analysing the unigrams of the preprocessed corpus we notices the appearence of special invisibel unicode characters. We deemed them non interesting and also removed them. The list of unicode characters that are removed is taken from [here](https://stackoverflow.com/questions/17978720/invisible-characters-ascii). Note that for transformer based learning methods we recommend using the raw texts, as they are much more capable and cleaning the text might remove information they could benefit from. Lastly we apply lemmatizing to normalise the words. We chose lemmatization over stemming because lemmatization takes in the context and POS tags of the words and usually gives better results. 


Our preprocessing pipeline is adapted from these two blog posts: [[1]](https://blog.devgenius.io/preprocessing-twitter-dataset-using-nltk-approach-1beb9a338cc1) [[2]](https://www.kaggle.com/code/redwankarimsony/nlp-101-tweet-sentiment-analysis-preprocessing).

Q2: **[What are the methods used ?]** Our data analysis involved exploring the distribution of uni- and bi-grams within our corpus of tweets, which is a standard practice in computational linguistics. An n-gram, also known as a Q-gram, refers to a consecutive sequence of n elements in a given text or speech sample. Our exploration produced visual representations of the distribution of both uni-grams and bi-grams across our datasets.

In total there are 299'430/3'684'672 unigrams/bigrams in the preprocessed corpus and 1'075'593/5'048'398 in the raw text corpus. Already here we see that the preprocessing removed a lot of noise. 

**[Quantitative analysis & Question answer]**
![unigrams](https://i.imgur.com/QFQEEzl.png)
We start by looking at the unigrams, both before and after preprocessing. It is very clear that the semantic meaning of the top words after preprocessing is much higher. Without the preprocessing the most common words are punctuations and non informative stopwords like 'the' and 'to'. When looking at the preprocessed corpus we see that all words have a clear semantic meaning. It reflects the semantics of the corpus much better as one can clearly see what the texts are about from these top 20 words ('covid', 'coronavirus', 'pandemic', 'china').

![bigrams](https://i.imgur.com/zSeXQHB.png)

The situation looks very similar when looking at bigrams. The only real difference here is that based on bigrams we can can detect the hashtags due to their two tokens. As the dataset was created by searching for hashtags it is to expect that they would appear among the top bigrams. As hashtags are removed in preprocessing these bigrams do not appear anymore in the preprocessed dataset. Other than the hashtags the bigrams before preprocessing contain much less semantic meaningful bigrams. We see now very meaning full bigrams like 'wear mask' or 'coronavirus outbreak'.

Let us now focus on just the preprocessed tokens. We now filter the dataset for just very positive (>3) and very negative (<-3) tweets.
![neg-pos](https://i.imgur.com/mNhVJnq.png)
We can clearly see the difference between the very positive and the very negative tweets. Especially for positive tweets the word 'love' and related concepts like the heart emoji play a very important role. The most common bigram for negative tweets seems to be 'let us', which might be reflecting the resistance towards regulations and restrictions. Also 'human right' appears which was often used as argument against the covid restrictions. Interestingly the bigram of the two laughing smiley seems to be indicative of negative sentiments. We hypothesise that the smileys are used in a sarcastic, trolling manner.

Let us now look at the distribution of the sentiment labels.
![](https://i.imgur.com/zKPPTZ7.png)

On the left you see the distribution of the sentiments, in orange the positive sentiments and in blue the negative sentiments. On the right the combinations. Overall 32.2% of the tweets have a negative score less than -1 and 43.3% a positive score higher than 1. We therefore have more tweets that are classified as being positive. Further we can see that most tweets are neither positive nor negative (1 -1). Only a small amount of tweets is classified with an extrem of 5/-5. Due to this imbalance it is important to think about how to train any model as there are far more neutral than extrem tweets.


Also interesting to see is that there are tweets that go in both directions as for example 3 and -4. We conclude that we should learn for each tweet a score for positive and negative sentiment seperately.

Q3: **[Metric]**

When thinking about a metric to measure our problem we have two choices. Formulate it as a classification problem, where each value 1 to 5 (or negative) is a class. Alternatively we can formulate it as a regression problem, where we want to get as close to the true value as possible, the closer the better. We argue that classification is the


weaker approach as e.g. for a true value of 5 the prediction of 4 is much better than the prediction of 1. With a classification framing this could not be represented. We therefore choose to look at is in a regression setting and use a distance based metric. We choose mean squared error (MSE) because it is higher the farther away the prediction is from the ground truth (quadratic).

As we have seen the dataset is very imbalanced. When prediction tweets we want to make sure that we do not miss any strong sentiments (e.g. values higher than 3). To combat this we introduce two alternative variants of the MSE. Both metrics follow the approach of seperately computing the MSE for every ground truth sentiment value, so called bins (we therefore have 5 bins). The first measurement is called **mean bin MSE (MBMSE)**, which completely ignores any imbalance in the dataset and takes the mean over all bins. The MBMSE ignores any counts in the input. This is not optimal as e.g. if we have 10000 correct predictions for sentiment values 1 to 4, but 1 very wrong prediction of sentiment value 5 the MBMSE will still be very high although the prediction is quite good. To combat this we introduce **log bin MSE (LBMSE)**, a compromise between MSE and MBMSE. Instead of averaging the results without taking into account the counts in each bin, we weight the MSE for each bin by the logarithm of the bin count. With this we make sure that a bin with a large count has a larger influence on the score but the influence stagnates the larger the count gets. Because the logarithm behaves almost linearly very close to zero and almost constant for large input values we further add the parameter alpha that scales the counts. With this we can guide how close LBMSE should be to MSE (small alpha) or to MBMSE (large alpha).

Let us analyse the behaviour of these metrics. The plot below shows what happens to the errors on our dataset if we predict a constant value (the x-axis is the value we predict for all positive sentiments). As our dataset is very imbalanced the MSE and the absolute error predict a very low error if we just predict 1 (because most samples are 1). The MBMSE is balanced and predicts the same error for both prediction 1 and prediction 5. LBMSE with α=1 gives a slightly higher error for just predicting 5 compared to just predicting 1. By reducing the α to 0.005 we see that it gets closer to what the MSE predicts. We will report MSE, MBMSE and LBMSE (α=0.005). The value of α was chosen because it lies approximately between MSE and MBMSE.

![](https://i.imgur.com/z70jlAt.png)


**[Code Snipped]**  Proposed metrics
```
def mse(y_true, y_pred):
    return np.square(y_true - y_pred)

def binify_results(y_true, mse):
    bin = []
    for i in np.unique(y_true):
        bin_elements = (y_true == i)
        bin.append((np.mean(mse[bin_elements]), bin_elements.sum()))

    return np.array(bin)

def mbmse(y_true, y_pred):
    mse_vals = mse(y_true, y_pred)
    bin = binify_results(y_true, mse_vals)
    return np.mean(bin[:,0])

def lbmse(y_true, y_pred, alpha=1):
    mse_vals = mse(y_true, y_pred)
    bin = binify_results(y_true, mse_vals)
    weights = np.log(bin[:,1]*alpha)+1
    return np.sum(bin[:,0] * weights) / np.sum(weights)
```


Q3: **[Dataset Splitting]**

To make sure that the split reflects the imbalance of the dataset, we split each combination of sentiments (e.g. 2 -2) seperately and combine them afterwards. We use a 20% test, 10% val and 70% train split.

As you can see in the figure below all splits have the approximatelly same sentiment distribution. A similar situation can be seen when looking at the tweet counts over time for all three splits. The distribution of tweets seems to be approximatelly the same. Therefore we do not suspect a big problem with label shifts over time and this being missrepresented by the splits.
![](https://i.imgur.com/mhpSBMr.png)

![](https://i.imgur.com/CDqsNx1.png)

**[Code Snipped]** Computing the split
```
# split the dataset into test, val, train seperately for each sentiment comb
unique_sentiment_combinations = np.unique(df.Sentiment)

test_ids, val_ids, train_ids = [], [], []

for sentiment in unique_sentiment_combinations:
    # get all ids for the current sentiment
    sentiment_ids = df[df.Sentiment == sentiment].index.values

    # shuffle the ids
    np.random.shuffle(sentiment_ids)

    # get the number of ids
    n_ids = len(sentiment_ids)

    # get the number of ids for each set
    n_test = int(n_ids * 0.2)
    n_val = int(n_ids * 0.1)
    n_train = n_ids - n_test - n_val

    # get the ids for each set
    test_ids.extend(sentiment_ids[:n_test])
    val_ids.extend(sentiment_ids[n_test:n_test+n_val])
    train_ids.extend(sentiment_ids[n_test+n_val:])
```


## Part 2: NLP learning based approaches for sentiment analysis


Q1: **[What are the methods used ?]**
VaDER sentiment analysis works by assigning sentiment scores to individual words based on their emotional valence and intensity. The valence of a word refers to whether it is positive, negative, or neutral, while the intensity refers to how strongly the emotion is expressed.

VaDER contains a dictionary of words that have been pre-assigned sentiment scores based on their valence and intensity. The process of assigning sentiment scores involved multiple annotators who independently rated the sentiment of words. For example, the word "happy" would have a positive sentiment score with a high intensity, while the word "sad" would have a negative sentiment score with a high intensity. For a text, the scores for each word are then aggregated to produce an overall sentiment score.

Q2 **[Provide a code snippet detailing how to use it for our task]**
**[Code snippet]**
```
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
analyzer.polarity_scores(test_data['preprocessed'][0])
```
Since VaDER is a lexicon-based approach to sentiment analysis, it relies on pre-assigned sentiment scores for individual words rather than machine learning algorithms trained on large datasets. Hence, there are certain pre-processing steps that might be unnecessary when using VaDER for sentiment analysis.

For example, VaDER may not require extensive data cleaning steps such as removing stop words or stemming because it assigns sentiment scores to individual words based on their valence and intensity, regardless of their frequency or context. Similarly, VaDER may not require extensive feature engineering or text normalization because it focuses on individual words rather than the overall structure or context of the text.

However, there are still certain pre-processing steps that may be necessary when using VaDER. For example, VaDER may require some basic text cleaning to remove any special characters or formatting that could affect the sentiment analysis (# or @).

Q3 **[Apply this method to our TweetsCOV19 dataset]**
We can then apply Vader to each of our tweets, taking care to rescale the outputs to the correct size.
**[Code snippet]**
```
test_data['Vader'] = test_data['preprocessed'].progress_apply(lambda tweet: [analyzer.polarity_scores(tweet)['pos']*4+1,analyzer.polarity_scores(tweet)['neg']*-4-1])
```
And we get the following results :
|VADER|POSITIVE|NEGATIVE|
|----|----|----|
|**MSE**|1.01|1.8|
|**LBMSE**|0.69|1.44|
|**MBMSE**|1.93|1.75|

It can be seen that compared to the results of Q8 the results are not bad. The results are better than the worse results of Q8 while being an very fast and simple method.
However, the results are far behind the results of the transformers.


### Word Embeddings

Q1: **[Bag of Words (BoW)]**
Bag of Words is an embedding approach which consider a text document as a "bag" of its individual words, disregarding any information about the order in which the words appear or any grammatical structure. The resulting matrix is a high-dimensional representation of the corpus, where each row corresponds to a document (here tweet) and each column corresponds to a unique word in the vocabulary. The value in each cell of the matrix represents the frequency of the corresponding word in the corresponding document (here tweet).

**[Code snippet]**
```
#We create a dummy function to use the already preprocessed tweets
def dummy(doc):
    return doc
    
vectorizer = CountVectorizer(tokenizer=dummy,preprocessor=dummy)
bow = vectorizer.fit_transform(train_set['preprocessed'])
train_set['bow'] = train_set['preprocessed'].progress_apply(lambda x: vectorizer.transform(x))
```

Q2: **[TF-IDF ]**
TF-IDF (Term Frequency-Inverse Document Frequency) is a method for text representation that assigns weights to the words in a document based on their importance in the document and across a corpus of documents. The TF-IDF score is calculated by multiplying the term frequency (TF) of a word by the inverse document frequency (IDF) of the word.

The TF-IDF score of a word in a document is given by:

$$ TF-IDF(w, d) = \text{tf}(w, d) * log(\frac{N}{\text{df}(w)}) $$
where:

-$w$ is a word in the document

-$d$ is the document

-tf$(w, d)$ is the term frequency of $w$ in $d$

-$N$ is the total number of documents in the corpus

-df$(w)$ is the number of documents in the corpus that contain $w$

**[Code snippet]**
```
tf_idf_transformer = TfidfTransformer(smooth_idf=False)
tf_idf = tf_idf_transformer.fit_transform(bow)
train_set['tfIdf'] = train_set['preprocessed'].progress_apply(lambda x: tf_idf_transformer.transform(x))
```

Q3: **[Word2Vec]**
The idea behind Word2Vec is to train a neural network on a large corpus of text data and learn vector representations of words that capture the semantic and syntactic relationships between them. The network is typically trained using either the continuous bag-of-words (CBOW) or skip-gram algorithm.

In CBOW, the network is trained to predict a target word based on the context words that surround it. The input to the network is a set of context words, and the output is the target word. In skip-gram, the network is trained to predict the context words given a target word.

We did not see any qualitative improvement using CBOW or skip-gram and therefore chose to use CBOW for its simplicity and its best computational performance.

**[Code snippet]**
The following code is to create our own Word2Vec dictionary on our dataset:
```
word_2_vec_Cbow = word2vec.Word2Vec(sentences=display_tweet['preprocessed'])
word_2_vec_Skip_Gram = word2vec.Word2Vec(sentences=display_tweet['preprocessed'],sg=1)
train_set['word2vec'] = train_set['preprocessed'].progress_apply(lambda tweet: [word_2_vec_Cbow.wv[word] for word in tweet])
```
And this code is to use a pretrained model:
```
pretrained_word2vec = api.load("word2vec-google-news-300")
```

Q4: **[GloVe]**
The GloVe model is based on the idea that the meaning of a word can be inferred from its co-occurrence statistics with other words in a corpus. The model learns word embeddings by factorizing a co-occurrence matrix, which contains the frequencies of word co-occurrences in a given corpus. The matrix is constructed by sliding a window over the text and counting the number of times each pair of words appears in the same context.

The GloVe model then factorizes the co-occurrence matrix using matrix factorization techniques such as Singular Value Decomposition or Alternating Least Squares. This factorization produces word embeddings that capture the relationships between words in the corpus.

**[Code snippet]**
```
pretrained_glove = api.load("glove-twitter-25")
train_set['glove'] = train_set['preprocessed'].progress_apply(lambda tweet: [pretrained_glove[word] for word in tweet])
```
Q5: **[FASTEXT]**
FASTEXT is an extension of the word2vec algorithm, which learns vector representations of words based on their co-occurrence in a large corpus of text. FastText takes this a step further by not only learning word embeddings, but also learning embeddings for subword units, such as character n-grams.

The algorithm works by first tokenizing the input text into words and subwords, and then training a neural network to predict the target label (such as a category or a sentiment) based on the input text. During training, the network learns both the word and subword embeddings, and uses them to make predictions.

One of the key advantages of FastText is that it is able to handle out-of-vocabulary (OOV) words, which are words that are not present in the training data. This is because the subword embeddings capture the meaning of word parts, which can be used to make predictions even for previously unseen words.

```
def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in tqdm(fin):
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = list(map(float, tokens[1:]))
    return data
pretrained_fastext = load_vectors("/kaggle/input/ml4hc-nlp/wiki-news-300d-1M.vec")
```
Q6: **[Visualization of embeddings]**
For the visualisation of embedding we chose to compare UMAP representation according to positive and negative sentiment of all the dataset.

In order to better interpret the results we decided to represent each tweet by a single vector using question 7 with an aggregation by maximum.


**[BOW]**
![](https://hackmd.io/_uploads/Hyp6CBxSh.png)
**[TF_IDF]**
![](https://hackmd.io/_uploads/B16gJIeHh.png)
**[WORD2VEC]**
![](https://hackmd.io/_uploads/BkdQkLgSh.png)
**[GLOVE]**
![](https://hackmd.io/_uploads/SyjV1UxB3.png)
**[FASTEXT]**
![](https://hackmd.io/_uploads/SyZP18gH3.png)
One can see a clear difference between the first two methods (BOW and TF_IDF) and the rest of the other embeddings. Their UMAP representation does not seem to create clusters and it seems very difficult to interpret the results. These two embeddings therefore seem less promising for the rest of the project.
For the other three methods, different clusters can be observed, suggesting a relevant semantic representation.
However, it is difficult to identify clusters that are related to positive and negative sentiments as these clusters seem to host both positive and negative sentiment tweets.
These three methods seem to be more promising because although they do not have a clear positive and negative cluster, they seem to have a better semantic understanding of the tweets.
Among these three methods the most promising ones seem to be WORD2VEC and FASTEXT







Q7: **[Tweet embeddings]**
The three approaches we used are to take the average, sum and maximum (by coordinate) of the words contained in a tweet to represent the tweet as a single vector.
We manage missing words by ignoring them in the design of the vector for the given tweet.
```
def tweet_embedding_max(dic,tweet):
    max = []
    for tw in tweet:
        if tw in dic:
            max.append(dic[tw])
    if len(max) > 0:
        result = [max(i) for i in zip(*max)]
    else:
        result = None
    return result
    
def tweet_embedding_mean(dic,tweet):
    mean = []
    for tw in tweet:
        if tw in dic:
            mean.append(dic[tw])
    if len(mean) > 0:
        result = np.mean(mean,axis=0)
    else:
        result = None
    return result

def tweet_embedding_sum(dic,tweet):
    sum = []
    for tw in tweet:
        if tw in dic:
            sum.append(dic[tw])
    if len(sum) > 0:
        result = np.sum(sum,axis=0)
    else:
        result = None
    return result
```
Q8: **[Classifier]**
As explained in the section on the metrics used, we consider that the problem should use regression models rather than classification models.
We then chose to use a multy layer percepton (MLP), a linear regression and a Radom forest regressor. And we use the metrics described above.
- MLP (Multilayer Perceptron) is a type of neural network that consists of multiple layers of nodes, here we used a hidden layer with 20 nodes and the output layer has 2 nodes for the positive sentiment of the tweet and the negative sentiment.
- Linear regression is a type of regression analysis where the goal is to model the relationship between a dependent variable and one or more independent variables by fitting a linear equation to the observed data. The equation takes the form of $y = b_0 + b_1x_1 + b_2x_2 + ... + b_nx_n$, where y is the dependent variable, $x_1, x_2, ..., x_n$ are the independent variables, and $b_0, b_1, b_2, ..., bn$ are the coefficients to be estimated.
- Random forest regressor is an ensemble learning method that combines multiple decision trees to improve the accuracy of the prediction. Each decision tree is constructed by randomly selecting a subset of the training data and a subset of the input features at each split. The final prediction is the average of the predictions of all the trees in the forest.

We have set the hyper parameters of each model with the available resources.
To analyse the performance of each model for each embedding and aggregation method we had to restrict the size of the dataset to fit our computational restrictions.
We did this by keeping the dataset as balanced as possible.
```
# Load the data
test_data = pd.read_pickle("/kaggle/input/ml4hc-nlp/val_embeddings_sum.pkl").dropna()
embeddings_df = pd.read_pickle("/kaggle/input/ml4hc-nlp/embeddings_sum.pkl")
groups = embeddings_df.groupby(["sentiment_pos","sentiment_neg"])
groups = groups.apply(lambda x: x if len(x) < 1000 else x.sample(1000))
groups = groups.reset_index(drop=True)
groups_clean = groups.dropna()
```
This give the following results for the lbmse metric:
| SUM POSITIVE | BOW | TF-IDF | WORD2VEC | FASTEXT | GLOVE
| -------- | -------- | -------- |-------- | -------- |-------- |
| **regression**     | 0.75   | 0.93  | 0.48   | 0.46  | 0.58|
| **random forest**     | 0.68 | 0.73  | 0.58   | 0.59  |0.67 |
| **MLP**     | 0.66 | 0.64  | 0.41   | 0.44  | 0.55|

| SUM NEGATIVE | BOW | TF-IDF | WORD2VEC | FASTEXT | GLOVE
| -------- | -------- | -------- |-------- | -------- |-------- |
| **regression**     | 1.65   | 1.8  | 1.26   | 1.02  | 1.27|
| **random forest**     | 1.64 | 1.65  | 1.41   | 1.3  |1.38 |
| **MLP**     | 1.47| 1.53  | 1.2   | 0.91  | 1.24|

| MEAN POSITIVE | BOW | TF-IDF | WORD2VEC | FASTEXT | GLOVE
| -------- | -------- | -------- |-------- | -------- |-------- |
| **regression**     | 0.91   | 0.95  | 0.56   | 0.59  | 0.68|
| **random forest**     | 0.73 | 0.76  | 0.64   | 0.68  |0.73 |
| **MLP**     | 0.70 | 0.66  | 0.48   | 0.53  | 0.64|

| MEAN NEGATIVE | BOW | TF-IDF | WORD2VEC | FASTEXT | GLOVE
| -------- | -------- | -------- |-------- | -------- |-------- |
| **regression**     | 1.71   | 1.75  | 1.32   | 1.12  | 1.38|
| **random forest**     | 1.64 | 1.62  | 1.45   | 1.37  |1.44 |
| **MLP**     | 1.51 | 1.49  | 1.21   | 0.96  | 1.31|

| MAX POSITIVE | BOW | TF-IDF | WORD2VEC | FASTEXT | GLOVE
| -------- | -------- | -------- |-------- | -------- |-------- |
| **regression**     | 0.73   | 0.75  | 0.58   | 0.48  | 0.59|
| **random forest**     | 0.79 | 0.82  | 0.7   | 0.54  |0.62 |
| **MLP**     | 0.7 | 0.6  | 0.55   | 0.45  | 0.58|

| MAX NEGATIVE | BOW | TF-IDF | WORD2VEC | FASTEXT | GLOVE
| -------- | -------- | -------- |-------- | -------- |-------- |
| **regression**     | 1.56  | 1.57  | 1.49  | 1.1  | 1.31|
| **random forest**     | 1.73 | 1.72  | 1.61   | 1.16  |1.32 |
| **MLP**     | 1.53 | 1.48  | 1.52   | 1.07  | 1.3|

**[Code snippet for the regression and random forest]**
```
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score

# Define the models
models = {
    'logreg': LinearRegression(n_jobs=-1),
    'rf': RandomForestRegressor(n_jobs=-1,verbose=2,n_estimators=10)
}

# Define the embeddings to use
embeddings = ['bow','tfIdf','word2vecCbow', 'fastext', 'glove']

# Loop over the embeddings and train a separate model for each
results = {}
for emb in embeddings:
    print("Using embedding ",emb)
    if emb == 'bow' or emb == 'tfIdf':
        X_train = vstack(train_data[emb])
        X_test = vstack(test_data[emb])
    else:
        X_train = train_data[emb].to_list()
        X_test = test_data[emb].to_list()
        
    # Get the training and testing data for this embedding
    
    y_train = train_data[['sentiment_pos','sentiment_neg']].to_numpy()
    y_test = test_data[['sentiment_pos','sentiment_neg']].to_numpy()
    
    # Train and evaluate the models
    for name, model in models.items():
        print("training whith ",name)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Store the results
        if emb not in results:
            results[emb] = {}
        results[emb][name] = y_pred
```
**[Code snippet for the MLP]**
```
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

class MLPRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLPRegressor, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

hidden_dim = 20# number of hidden units
output_dim = 2 # 2 output values for negative and positive sentiment

embeddings = ['bow','tfIdf','word2vecCbow', 'fastext', 'glove']
for emb in embeddings:
    print("Training MLP for ",emb)
    y_train = train_data[['sentiment_pos','sentiment_neg']].to_numpy()
    y_test = test_data[['sentiment_pos','sentiment_neg']].to_numpy()
    
    if emb == 'bow' or emb == 'tfIdf':
        input_size = train_data[emb][0].shape[1]
        X_train = vstack(train_data[emb]).tocoo()
        X_test = vstack(test_data[emb]).tocoo()
        # convert X_train to a sparse tensor
        X_train = torch.sparse.FloatTensor(torch.LongTensor(np.array([X_train.row, X_train.col])),
                                            torch.FloatTensor(X_train.data),
                                            torch.Size(X_train.shape))
        X_test = torch.sparse.FloatTensor(torch.LongTensor(np.array([X_test.row, X_test.col])),
                                            torch.FloatTensor(X_test.data),
                                            torch.Size(X_test.shape))
    else:
        input_size = len(train_data[emb][0])
        X_train = train_data[emb].to_list()
        X_train = torch.FloatTensor(X_train)
        X_test = test_data[emb].to_list()
        X_test = torch.FloatTensor(X_test)
    # create the MLP regressor
    model = MLPRegressor(input_size, hidden_dim, output_dim)
    # create a TensorDataset from X_train and y_train
    train_dataset = TensorDataset(X_train, torch.from_numpy(y_train))
    test_dataset = TensorDataset(X_test,torch.from_numpy(y_test))
    # create a DataLoader from the train_dataset
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset,batch_size=32,shuffle=False)

    # move the model to the GPU if available
    if torch.cuda.is_available():
        model.to('cuda')

    # define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())

    # train the model
    for epoch in range(10):
        print(epoch)
        # iterate over the data and labels in batches
        for data, target in tqdm(train_loader):
            # move the data and labels to the GPU if available
            if torch.cuda.is_available():
                data = data.to('cuda')
                target = target.to('cuda')

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward pass
            output = model(data)

            # calculate the loss
            loss = criterion(output, target.float())

            # backward pass and optimization
            loss.backward()
            optimizer.step()
    print('eval')
    output = []
    model.eval()
    for data, target in tqdm(test_loader):
        if torch.cuda.is_available():
            data = data.to('cuda')
        output.append(model(data).cpu().detach().numpy())
    y_pred = output
    y_pred = np.concatenate(y_pred,axis=0)
    print("lbmse : ",lbmse(y_test[:,0],y_pred[:,0]),lbmse(y_test[:,1],y_pred[:,1]))
```


Q9: **[Performance comparison ]**
Of the possible regressions, MLP almost always gives better results than the other two methods. However, linear regression is sometimes very close to the results of the MLP, while being simpler and faster than the MLP. The random forest regressor performs less well and is quite slow, especially with high dimensional embeddings like BOW and TF_IDF.
Among the different possible embeddings, our hypotheses in question 6 seem to be confirmed:
WORD2VEC and FASTEXT and GLOVE perform better than BOW and TF_IDF.
While WORD2VEC and GLOVE seem to be quite close, FASTEXT seems to be a bit better than the other two on average.
Furthermore BOW and TF-IDF are by far the slowest methods due to their very large dimensions.
For the different ways of creating embeddings the three methods are quite close.These three methodes are of the same order of magnitude in computing speed as well. However the sum seems to be the best of the three options, it reaches in particular the best score in positive (0.41 with WORD2VEC) and in negative (0.91 with FASTEXT).
If we had a method to deploy we would choose an MLP with FASTEXT and sum aggregation.

A possible improvement would be to test more architecture for the MLP, to train the model for longer and on more data if resources are no longer an important limit.
Furthermore it would be interesting to consider using ensemble methods to combine predictions from multiple models (e.g. using WORD2VEC and FASTEXT with linear regressions and MLP).
It would also be interesting to try to fine tune the pretrained models of WORD2VEC and FASTEXT on our tweet corpus to see if it is possible to improve their representation.


### Transformers

Q1: **[Transfomer-based language models]**

Transformer-based large language models are a type of neural network architecture that has gained significant attention in natural language processing due to their state-of-the-art performance on a variety of language tasks. They use a self-supervised learning approach to learn general-purpose representations of natural language text.

The key idea behind transformer-based models is to process the input text in parallel, rather than sequentially as in traditional recurrent neural networks (RNNs). The transformer architecture was introduced in the seminal ["Attention is all you need" paper](https://arxiv.org/abs/1706.03762) in 2017 and includes the multi-head attention  mechanism that allows the model to attend to different parts of the input sequence simultaneously, enabling it to capture long-range dependencies and contextual relationships between words. Further it introduces positional encodings. Since the transformer doesn't process a text sequencially but looks at the full text at once, the model can't distinguish the position of tokens. To circumvent that the authors proposed positional encodings, where they add a positional identifier to the embedding of each word.

Transformers are often classified as being encoder-only, decoder-only, or encoder-decoder models, depending on whether they contain an encoder, a decoder, or both. Encoder-only models process the whole input sequence in parallel and generate a fixed-length vector representation of the input sequence, which can be used for downstream tasks such as text classification or sequence labeling. Decoder-only models, on the other hand, take a fixed-length vector representation of the input sequence as input and generate a sequence of outputs, such as in the case of language generation. 

Encoder-decoder models contain both an encoder and a decoder component, and are often used in sequence-to-sequence tasks such as machine translation or text summarization. In these models, the encoder processes the input sequence and generates a fixed-length vector representation, which is then passed to the decoder component to generate the output sequence.

The training process for transformer-based models involves pre-training on a large corpus of text using an unsupervised learning objective such as masked language modeling, next sentence prediction or simple autoregressive text prediction. This pre-training step allows the model to learn general-purpose language representations, which can then be fine-tuned on downstream tasks such as text classification or question-answering.

Some of the architecture choices and training details that are crucial to the performance of transformer-based models include:

**Model size**: Larger models typically have more parameters and can capture more complex patterns in the data, but require more training time and computational resources.
**Pre-training data**: Pre-training on a large, diverse corpus of text can help the model learn more robust language representations. Especially in the recent months [it has been shown](https://arxiv.org/pdf/2302.13971v1.pdf) that the data quality also plays an important role.
**Pre-training objectives**: Different pre-training objectives can lead to different strengths and weaknesses in the model's language representations. For example, masked language modeling may be better for capturing syntax and semantics, while next sentence prediction may be better for capturing discourse-level relationships.
**Training parameters**: As with all machine learning methods the training parameters have a huge influence on downstream performance as well as convergence speed. The most important parameters that one needs to choose are learning rate and batchsize. [It has been shown](https://arxiv.org/pdf/2005.14165.pdf) that for large language models (LLMs) large batch sizes and smaller learning rates seem to be working best.
**Fine-tuning procedure**: The fine-tuning process requires careful design to enable the model to adapt to the target task while preserving the general-purpose language representations acquired during pre-training. However, if the same model is intended to solve multiple downstream tasks using various fine-tuning objectives, the process requires particular attention to prevent the phenomenon known as "catastrophic forgetting," which involves the model's loss of previously acquired knowledge during fine-tuning.

### Comparison of BERT, RoBERTa and GPT-3
**BERT (Bidirectional Encoder Representations from Transformers)** is a popular transformer-based model that was introduced by Google in 2018. It is an encoder-only architecture. Importantly it is *bidirectional*, meaning that when creating an embedding for an input token it does look at the full context, both right and left of the token.  It is pre-trained on a large corpus of text using a masked language modeling (MLM) objective as well as next sentence prediction (NSP). In MLM training a random token of the input sequence is masked and the models goal is to predict the masked token. The NSP objective refers to the task of predicting whether a given sentence is the next sentence in a sequence of sentences. BERT uses WordPiece embeddings with a vocabulary size of approximately 30k tokens. Further, the model uses Positional Encodings and the Gelu activation function. BERT refers to a sequence of models of which the largest has 340M parameters. It was trained for 1M steps with a batchsize of 256 on roughly 13GB of compressed text data.

**RoBERTa (Robustly Optimized BERT pre-training approach)** is a variant of BERT that was introduced by Facebook AI in 2019. It uses a similar architecture and pre-training objective as BERT, but with some modifications to the training procedure that lead to improved performance on a number of benchmarks. They trained the model longer on a larger dataset with a larger batchsize. The proposed training dataset is around 160GB of compressed text data and the increased batchsize is 8096. Further the model is trained for approximatelly 5 times the training amount of BERT. Other than BERT, RoBERTa is only trained on MLM without NSP. Another improvement they proposed is dynamic masking. In BERT the mask for each training sample is generated ahead of time, which results in BERT seeing the same training mask approximately 4 times over the course of its training. For the training of RoBERTa, the mask is genereated for each step individually which results in RoBERTa not seeing any mask twice. This is crutial for the longer training of RoBERTa. Lastly they increased the vocabulary size of the tokenizer from 30k to 50k.

**GPT (Generative Pre-trained Transformer)** is a family of transformer-based models that were introduced by OpenAI. Unlike BERT, GPT is a decoder-only model and is pre-trained on a large corpus of text using a autoregressive left-to-right language modeling objective. Simplified, GPT is trained by predicting the next word in an input sequence, which allows it to generate coherent and fluent text. The third version of GPT (GPT-3) was released in 2020 and is, compared to the other two models, **not open-sourced**    . The largest version of GPT-3 has around 172B parameters and is therefore much larger than BERT. It has been trained on 570GB of text, which corresponds to around 400M tokens. The batchsize is increased to 3.2M and its context window is set to 2048, which means that when generating a token, GPT-3 looks at the last 2048 tokens. The context window for BERT and RoBERTa is only 512. To ensure high quality, the authors sampled the high quality parts of the dataset more frequently (mostly books and wikipedia) while the rest (internet) is sampled at most once. GPT-3 has been used for a variety of tasks such as text completion, text generation, and question-answering. Its capabilities for zero-shot task solving were ground braking and were considered state-of-the-art until its successor, GPT-4, was released in March 2023.

Q2: **[Scalability]**

Embedding-based approaches and transformer-based language models differ significantly in their scalability because  embedding-based typically have significantly fewer parameters than transformer-based models.

For example, the original word2vec model has a total of only two layers: an input layer and a projection layer. The projection layer consists of the learned embeddings for each word in the vocabulary, and the number of parameters in this layer is equal to the product of the vocabulary size and the embedding size. For a vocabulary of 100,000 words and an embedding size of 300 dimensions, the projection layer of the word2vec model would have only 30 million parameters. In contrast, a small transformer-based model like BERT has approximately 110 million parameters, while larger models like GPT-3 have billions of parameters.

The smaller number of parameters in embedding-based approaches can make them more computationally efficient and require less data to train. However, they are typically limited in their ability to handle complex tasks such as generating sequences of text or answering questions based on context. Additionally, embedding-based approaches may struggle with rare or out-of-vocabulary words, which can be mitigated by using subword tokenization but may require more computation resources.

In contrast, transformer-based models can handle more complex tasks and have shown impressive performance on a wide range of natural language processing tasks. However, their large number of parameters can make them computationally expensive to train and deploy. Additionally, larger datasets are typically required to train effective transformer-based models, especially for more complex tasks.

Overall, the choice between an embedding-based approach and a transformer-based approach depends on the specific task and available computational resources. For smaller-scale applications or tasks that primarily require word-level information, embedding-based approaches can be a good choice. For larger-scale applications or tasks that require more context and higher-level language understanding, transformer-based models may be more appropriate despite their larger computational requirements.
    
Q3: **[Code]**

The objective is to perform multi-class regression using a pre-trained RoBERTa language model called "cardiffnlp/twitter-roberta-base-sentiment-latest". To accomplish this we have to customize the architecture such that we have:

- Two nodes in the output layer that correspond to the two labels (positive, negative)
- Sigmoid activation for each node in the output layer. This is because we normalized the label and we use a scale from 0 to 1 instead of 1 to 5 or -1 to -5. 
- Binary cross-entropy loss function. 

This problem setting can be configured in the arguments of 'AutoModelForSequenceClassification.from_pretrained()' method. 

The argument problem_type="multi_label_classification" is convenient for our setting as it sets the loss function to be BCEWithLogitsLoss() which is the binary cross entropy loss with a sigmoid activation function built in. 

```
# Import necessary packages
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer, 
    TrainingArguments, 
    DataCollatorWithPadding
)

# Specify the pre-trained model to be used
MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"

# Define mapping between class labels and IDs
label2id = {"positive": 0, "negative": 1}
id2label = {0: "positive", 1: "negative"}

# Instantiate tokenizer and data collator
tokenizer = AutoTokenizer.from_pretrained(MODEL)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Tokenize dataset using preprocess function
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length = 512)

tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Load pre-trained model and set up model architecture for fine-tuning
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL,
    num_labels=2, # two nodes in the output
    id2label=id2label,
    label2id=label2id,
    problem_type="multi_label_classification",
    ignore_mismatched_sizes=True
)

# Freeze/unfreeze layers of the model
# Freeze embedding layer
for param in model.roberta.embeddings.parameters():
    param.requires_grad = False

# Unfreeze classifier layer
for param in model.classifier.parameters():
    param.requires_grad = True

# Freeze all but last i layers of the encoder
for param in model.roberta.encoder.layer[:-i].parameters():
    param.requires_grad = False

# Specify training hyperparameters
training_args = TrainingArguments(
    output_dir=f"my_model_{version_name}",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=False
)

# Initialize trainer object
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["val"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()
```
Q4: **[Performance analysis]**
In this section, we present the performance evaluation of our models for predicting the negative and positive sentiments separately. As discussed in Part 1 Q3, we use the LBMSE metric (α=0.005) due to the significant class imbalance in our dataset.  

The two tables below summarize the evaluation metrics of the models on the validation set for positive and negative sentiments, respectively. The first column reports the results of the pretrained model without further finetuning, and the second column presents the results of the model after unfreezing the classifier and the last three layers of the encoder. 


| *Positive* | No-Finetuning | Finetuning |
| -------- | -------- | -------- |
| **MSE**     | 0.16997   | 0.00580  |
| **MBMSE**     | 0.13483 | 0.04867  |
| **LBMSE**     | 0.11239 | 0.01168  |



| *Negative* | No-Finetuning | Finetuning |
| -------- | -------- | -------- |
| **MSE**     | 0.17501   | 0.00958  |
| **MBMSE**     | 0.12825 | 0.02859  |
| **LBMSE**     | 0.11626 | 0.01978  |


First, we observe that the errors in predicting negative and positive sentiments are similar. 

We observe that fine-tuning the last three layers of the encoder significantly improves the model's performance. The LBMSE is reduced by a factor of ~10 in both cases, demonstrating that fine-tuning allows the model to adjust its weights and learn more appropriate representations for our task.

In order to further improve the performance of our sentiment analysis model, several alternative approaches could be investigated if computational resources were not a bottleneck. For instance:

1. **Ensemble Learning**: This approach involves training multiple models, each of which makes its own prediction. The final prediction is then obtained by taking the average or majority vote of the individual predictions. This technique is known to reduce the generalization error and improve the robustness of the model. However, training multiple models can be computationally expensive and may require significant resources.

2. **Hyperparameter Search**: The performance of deep learning models heavily depends on the choice of hyperparameters such as the learning rate, batch size, number of epochs, etc. Therefore, it is crucial to find the best set of hyperparameters that optimize our model's performance. This can be achieved through an exhaustive search over the space of hyperparameters, although this can be time-consuming and requires a significant amount of computational resources.

3. **Stochastic Weight Averaging (SWA)**: This technique involves computing a running average of the weights of the model during training, which is then used as the final weights of the model instead of the last set of weights obtained after training. The intuition behind SWA is that it smooths out the trajectory of the weights during training, which can reduce overfitting and improve generalization performance. This technique has shown promising results in a variety of tasks, including image classification, language modeling, and object detection. 


Q5: **[Transfer learning details]**

We further investigate the impact of fine-tuning on the performance of the pre-trained language model for our task. Specifically, we generate a set of models, each having a different number of frozen layers, and compare their performance on the evaluation set. The baseline model, which has all the layers frozen, serves as a reference point for comparison. In contrast, the other models have all the layers frozen except for the classifier head and the last k layers of the encoder.

The two figures below show the results when unfreezing 0, 1, 2 or 3 layers of the encoder. 

![](https://hackmd.io/_uploads/r1kWe_0Vh.png)

![](https://hackmd.io/_uploads/HJReed0Vh.png)



Results indicate that increasing the number of unfrozen layers leads to a decrease in the LBMSE, suggesting that adapting more weights to the specific task improves performance. In other words, it suggests that a greater degree of task-specific adaptation can be achieved by unfreezing more layers of the pre-trained model. 



Q6: **[Embeding analysis]**

We use the pretrained model without the classification head to obtain the tweet embeddings in a 768-dimension space. The representation of each tweet's content as an embedding can be obtained by examining the embedding of the [CLS] token, which encapsulates the information from the entire text sequence.

To visualize these embeddings, we applied UMAP to reduce their dimensionality to 2D. We then plotted the embeddings of highly positive (ground truth sentiment score of 4 or 5) and highly negative (ground truth sentiment score of -4 or -5) tweets.
The first of the plots below corresponds to the embeddings obtained without fine tuning while the second plot corresponds to the embeddings obtained after fine tuning the last three layers of the encoder. 

![](https://hackmd.io/_uploads/SJ5eXNC43.png)

![](https://hackmd.io/_uploads/SyXTMER43.png)


We observe that, the embeddings generated by the pre-trained model are somewhat effective at separating the two types of tweets based on sentiment score. 
Besides, we observe that fine-tuning the last 3 layers of the encoder further improves the separation between positive and negative tweets. The fine-tuned model's embeddings exhibit a tighter clustering of positive and negative tweets, with less overlap between the two clusters.
This observation corroborates the findings of Q4, indicating that fine-tuning has a positive impact on the model's ability to extract useful information from the tweets.

The clustering obtained here is better than those obtained in Part 2 Q6. This was expected because pretrained models like RoBERTa are trained on massive amounts of text data, and can capture more complex semantic relationships between words and phrases. In contrast, word embedding approaches like GloVe or TF-IDF only capture co-occurrence patterns and may not be as effective at capturing more nuanced semantic relationships.  
Moreover, the pretrained model generates embeddings with much higher dimensionality than word embedding approaches. This higher dimensionality allows for more effective separation of clusters in visualization.

## Part 3: Downstream Global Health Analysis

Q1: **[Research Question]**


The COVID-19 pandemic had a significant impact on public health and the global economy. Vaccines were crucial in fighting the virus, and it's important to understand public views to promote vaccine acceptance and reduce hesitancy. Analyzing sentiment in tweets can provide insights into public opinions on COVID-19 vaccination over time.

By analyzing tweet sentiment, we can see how attitudes are changing and identify any concerns or misconceptions that might be stopping people from getting vaccinated. This information can help public health messages and interventions to increase vaccine acceptance. It may also be useful for the public relations campaigns of individual vaccine companies.

Following up on the approach of [Public Perception of COVID-19 Vaccine by Tweet Sentiment Analysis](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9594036&tag=1) by Yang and Sornlertlamvanich we intend to instead leverage pretrained large language models for analysing the public opinion of COVID-19 vaccinations over the course of the pandemic. Since our data is limited to the start of the pandemic we focus on exploring the effect of the first wave on the publics opinion towards vaccinations. 

Q2: **[Methodology]**

To analyse the sentiment of vaccine related tweets we choose to leverage the RoBERTa model that was finetuned on the training set of the TweetsCOV19 dataset. It is performs best on the validation set and allows us to leverage both the finetuned embeddings as well as the predicted sentiments on the vaccine related tweets.

We intend to look at a subset (test set) of the TweetsCOV19 dataset with a size of 113'745 tweets. Because our dataset is limited we initially look at the problem on a global scope and do not discriminate between different vaccinations. 

## Extracting vaccine related tweets
We have developed a novel method to extract vaccine-related tweets from our dataset. Due to the small size of the dataset, this step is crucial. We leverage the semantic understanding of our language model, which has been fine-tuned on COVID-specific tweets, to identify words that refer to vaccinations. Initially, we define a small set of vaccine-related candidate words and embed them using only the RoBERTa layer of our model without the sentiment classifier. Next, we preprocess the tweets and generate the complete vocabulary used in our dataset, which contains 105,894 unique words. We then use the fine-tuned RoBERTa model to embed all words in the vocabulary. In cases where a word has multiple tokens, we compute the embedding by averaging over all tokens, while ignoring the CLS and SEP tokens.

We have obtained a complete representation of our vocabulary in the semantic space of our model. By examining neighboring words in this space, we can identify new words that relate to vaccines. To achieve this, we examine the top 100 neighboring words for each of our candidate words and manually remove any tokens that do not appear to be vaccination-related.

As a set of initial candidate words we use 
`{"vaccine", "vaccines", "vaccination", "moderna", "pfizer", "astrazeneca", "mrna"}`.
By using the proposed method we extend this initial list of 7 candidate words to following 52 words:
```
{'anti-vaccination',
 'anti-vaccine',
 'antivaccine',
 'astrazeneca',
 "astrazeneca's",
 'avoided.vaccination',
 'bcgvaccine',
 'covidvaccine',
 'fluvaccine',
 'getvaccinated',
 'moderna',
 'mrna',
 'novaccine',
 'novaccines',
 'novax',
 'pfizer',
 'pre-vaccine',
 'prevaccine',
 'pro-vaccine',
 'rollouts',
 'shiv-vaccine',
 'unvaccinated',
 'vacc',
 'vaccin',
 'vaccinate',
 'vaccinated',
 'vaccinating',
 'vaccination',
 'vaccinator',
 'vaccine',
 'vaccine-associated',
 'vaccine-induced',
 'vaccine-injured',
 'vaccine-preventable',
 'vaccine.this',
 'vaccineagenda',
 'vaccinee',
 'vaccinefreedom',
 'vaccines',
 'vaccines-by',
 'vaccinesafety',
 'vaccinesceptics',
 'vaccinesforall',
 'vaccinesinjure',
 'vaccineskill',
 'vaccinesprofits',
 'vaccineswork',
 'vaccineunicorn',
 'vacine',
 'vacs',
 'vax',
 'vaxx'}
```
With the initial 7 candidate words we can extract 1241 tweets that talk about vaccines. With the extended list we can increase this number to 1446 which is a 17% larger dataset. We therefore deem this method successful.

The neighborhoods of the words 'vaccine', 'vaccines', and 'vaccination' yielded a lot of new relevant words, while the other candidate words were less successful. This may be due to their specificity and infrequency in the training data, which prevented the model from deriving a meaningful position for them in the semantic space. It is also important to note that the training data for the model only goes up until May 2020, and since the first COVID-19 vaccines were not [authorized until December 2020](https://en.wikipedia.org/wiki/List_of_COVID-19_vaccine_authorizations), it is likely that the names of the vaccines were not widely known before May 2020, rendering their learned embeddings less useful.

Top 10 neighbourhood of `vaccine` (together with cosine distance to embedding of `vaccine`):
```
{'vaccine': 0.0,
 'vaccination': 0.02323389,
 'vaccines': 0.035549045,
 'vaccinate': 0.037489653,
 'prevaccine': 0.03864801,
 'vaccinator': 0.044668496,
 'vaccinee': 0.04500544,
 'pre-vaccine': 0.045475602,
 'vaccinating': 0.048684,
 'vaccine.this': 0.05058527}
```
Top 10 neighbourhood of `moderna`
```
{'moderna': 0.0,
 'hoffa': 0.044623017,
 'nationa': 0.045935333,
 'felda': 0.04615116,
 'ricta': 0.047292054,
 'choola': 0.04785067,
 'aleigha': 0.048243523,
 'aveda': 0.04916358,
 'bolta': 0.049726963,
 'cipla': 0.050793827}
```

**[Code Snipped]** Vocabulary embeddings
```
# compute embeddings in batches
vocab = ... # list of all words in the dataset
vocab_embeddings = []
for i in tqdm(range(0, len(vocab), batch_size)):
    batch = vocab[i:i+batch_size]
    batch_tokens = tokenizer(batch, padding=True, return_tensors="pt")
    batch_tokens_cuda = batch_tokens.to("cuda")
    batch_tokens = {k: v.cpu().numpy() for k, v in batch_tokens.items()}
    batch_embeddings = model.roberta(**batch_tokens_cuda)[0].detach().cpu().numpy()
    for j, word in enumerate(batch):
        # average over all tokens except CLS and SEP
        vocab_embeddings.append(batch_embeddings[j][batch_tokens["attention_mask"][j] == 1][1:-1].mean(axis=0))
```

**[Code Snipped]** Find alternative candidate words
```
# find nearest neighbors
from sklearn.neighbors import NearestNeighbors
nn = NearestNeighbors(n_neighbors=100, metric="cosine")
nn.fit(vocab_embeddings)
# find nearest neighbors for each vaccine word
vaccine_words = ["vaccine", "vaccines", "vaccination", "moderna", "pfizer", "astrazeneca", "mrna"]
vacc_nn = {}
for word in vaccine_words:
    nn_indices = nn.kneighbors(vocab_embeddings[vocab.index(word)].reshape(1, -1))[1]
    nn_similarities = nn.kneighbors(vocab_embeddings[vocab.index(word)].reshape(1, -1))[0]
    vacc_nn[word] = {vocab[i]: nn_similarities[0][j] for j, i in enumerate(nn_indices[0])}
```

## Analysing Tweet Sentiments

We use the pretrained RoBERTa model finetuned on the last 3 encoder blocks to predict the sentiment of selected vaccine related tweets.

**[Code Snipped]** Predict Sentiment
```
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

df_vacc = pd.read_pickle("df_vacc.pkl")
texts = df_vacc["TweetText"].tolist()

# load model
model = AutoModelForSequenceClassification.from_pretrained("/home/julian/repositories/ml4h/project2/my_model_v_3/checkpoint-24886").to("cuda")
model.eval()
tokenizer = AutoTokenizer.from_pretrained("/home/julian/repositories/ml4h/project2/my_model_v_3/checkpoint-24886")

# predict
import torch
from tqdm import tqdm

batch_size = 1024
sentiments = []

with torch.no_grad():
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i+batch_size]
        batch_tokens = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to("cuda")
        batch_sentiments = model(**batch_tokens)[0].detach().cpu().numpy()
        sentiments.append(batch_sentiments)
        
# rescale sentiments
sentiments = F.sigmoid(torch.tensor(sentiments))
# rescale
sentiments = (sentiments) * 4 + 1
```
We can then plot the sentiment over time with the following code snipped.

**[Code Snipped]** Predict Sentiment
```
df_vacc.Timestamp = pd.to_datetime(df_vacc.Timestamp)
df_vacc["months"] = df_vacc.Timestamp.apply(lambda x: f"{x.year}-{x.month}")
df_vacc["Positive Sentiment"] = sentiments.numpy()[:, 0]
df_vacc["Negative Sentiment"] = sentiments.numpy()[:, 1]
df_vacc.groupby("months")[["Positive Sentiment", "Negative Sentiment"]].mean().sort_index().plot()
```


Q3: **[Results & Analysis]**

We analyse the sentiment of the population through the lens of 1446 tweets between October 2019 and Mai 2020. As you can see in the plot below between October 2019 and February we have around 100 vaccine related tweets per month. Covid started to spread around the beginning of 2020, which is clearly reflected in the number of tweets we see about vaccines. 


![NumberTweetsVaccine](https://hackmd.io/_uploads/Bklz_ZT42.png)

In the Figure below you can see the location of the tweets. The size of the blue circle around a red point is directly proportional to the number of tweets from that country. We see that by far most tweets come from the US. Followed up by England, India and Canada. 

![TweetsLocation](https://hackmd.io/_uploads/SJVxaMpNh.png)

We are analyzing the predicted sentiments over time and have presented our findings in the following figure. The average sentiment for each month is displayed. Our analysis indicates that following the onset of COVID-19, the negative sentiment decreased significantly after February 2020 in comparison to the time prior. Although positive sentiment remains relatively consistent throughout, it takes a dip at the beginning of 2020 before rising again. Of particular interest, we observe a peak in negative tweets during January, before the actual outbreak of Covid.

![](https://hackmd.io/_uploads/SkNeyma42.png)

We can observe the same phenomenon when looking at highly negative or highly positive tweets. The plot below shows the percentage of tweets per month that are classified as having a sentiment higher than 3.

![](https://hackmd.io/_uploads/rJUVBQp43.png)

We manually sampled 10 of the negative tweets in January. About half of them refer to vaccination in general and have no covid specific references, e.g. `'"Public health advocates sound alarm over #antivaxx documentary" https://t.co/ajnWy0TnYK via @jordanomstead @CBCEdmonton\n\nMe: "This is not about an open debate... This is really about the spreading of harmful information that can really erode public confidence in vaccines."`. The other half can be interpreted as a panic reaction to first leaked information about the virus, e.g. `'Tw// corona virus\n\nThis is so fckn scary bro people are just dropping like flies and there’s still no vaccine or cure  https://t.co/IB6qCsLA1B'`. Another explanation to the spike of negativity around January 2020 could be the general public hysteria in the US around the the [events of January 6](https://en.wikipedia.org/wiki/January_6_United_States_Capitol_attack). This may have had an influence on the general sentiment on twitter. 

Starting from March 2020, we noticed a decline in negative tweets and an increase in positive ones. We interpret this trend as a sign that the general public began to comprehend the severe consequences of the virus on society and healthcare, and the vaccine as a potential solution to this issue.

Based on the analysis of tweet sentiment related to COVID-19 vaccinations, we can make the following takeaways:

* Following the outbreak of the virus outside of China, there was a surge in positive sentiment towards COVID-19 vaccinations on Twitter, as observed by our analysis. We believe that this trend indicates the public's determination to stay strong during a crisis and view the 'vaccination' as a viable solution to the problem.
* Prior to the start of the pandemic (Dez. 2019 and Jan. 2020), we observed an elevated level of negative sentiment. Our hypothesis is that this can be attributed to the limited information that was accessible during December 2019 and January 2020, leading to a sense of panic among the public.
* The use of language models, such as RoBERTa, can provide valuable insights into public opinion on COVID-19 vaccinations by analyzing tweet sentiment.
* The proposed method for extracting vaccine-related tweets based on candidate words and semantic embeddings can be successful in increasing the dataset size and identifying new vaccination-related terms.
* The success of this method depends on the initial selection of candidate words, the quality of the semantic embeddings used as well as the size of the twitter dataset investigated.

Q4: **[Comparison to literature]**

We compare our approach to Yang and Sornlertlamvanich's [Public Perception of COVID-19 Vaccine by Tweet Sentiment Analysis](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9594036&tag=1). They focused on sentiment towards vaccines during the first vaccinations rollout from December 2020 to June 2021, while we analyzed the public's opinion towards vaccinations during the early months of the COVID pandemic. They also examined sentiment towards specific vaccines in the US, UK, and Japan, while we looked at the general sentiment towards vaccinat
Although we have no data between June to November 2020, Yang and Sornlertlamvanich's results for the US show a continued positive trend in sentiment towards all vaccines, consistent with our findings.

Our methodology differs from Yang and Sornlertlamvanich's approach in several ways. First, we use a score from 1 to 5 for both negative and positive sentiment instead of treating sentiment as one-dimensional, from negative to positive sentiment. This allows for a more nuanced analysis since a tweet can be both positive and negative. Second, we use a RoBERTa model to predict sentiments, while they use [TextBlob](https://textblob.readthedocs.io/en/dev/quickstart.html#sentiment-analysis) based on a Naive Bayes classifier. We believe that our approach is more capable due to the model's nuanced language understanding, although further verification is needed. Finally, we leverage the semantic space learned by the LLM to detect alternative words referencing tweets, which could be used to extend Yang and Sornlertlamvanich's dataset with more data.

Q5: **[Discussion]**

As stated earlier, our approach differs from that of Yang and Sornlertlamvanich in that we have employed contemporary tools and made use of models that have been pretrained on the task of comprehending human language in general. These models acquire a semantic space that can be utilized for both candidate words retrieval and sentiment analysis. But running and training large language models requires significant resources, and they are difficult to comprehend due to their black-box nature. Therefore, we need to rely on quantitative performance analysis to validate our method (Part 2).

Our approach leverages the sentiment model's semantic space to greatly expand the initial set of candidate words, enabling us to extract more tweets related to the analyzed issue. This technique reduces bias and avoids selecting tweets that unconsciously skew sentiment analysis. For instance, choosing only tweets containing the word "vaccine" may lead to less negative sentiment (mean negativity is -1.56) compared to using "vax" (mean negativity of -2.42). However, extending the set of candidate words also introduces noise to the dataset.

The dataset of only 1446 tweets is limited, and our interpretations should be taken with caution due to the small sample size and early timeframe of the data (only the beginning of the pandemic). We recommend repeating this experiment on a larger dataset covering the entire pandemic, as different regions of the world handle the pandemic differently. This would allow for a better global analysis and enable tracing public opinion back to individual countries and companies. Like Yang and Sornlertlamvanich, we could focus on specific vaccines and individual world regions.



Q6: **[Summary & Conclusion]**


Understanding public attitudes towards COVID-19 vaccination is crucial to promote acceptance and reduce hesitancy. Our study utilized the pretrained RoBERTa-based language model to analyze public opinion on COVID-19 vaccination during the pandemic. We employed a novel method that expanded the candidate word list, extracting 17% more vaccine-related tweets using semantic language understanding. We applied a finetuned RoBERTa model to determine the sentiment of vaccine-related tweets.

Our findings revealed an initial negative response to vaccination during the early days of the pandemic, which we attribute to the first panic wave caused by scarce information about the virus. However, with the arrival of the first waves in Western countries, a general positive sentiment towards vaccination emerged. Our methodology can be extended to study opinions on individual vaccines or specific regions. Our work highlights the potential of natural language processing techniques in analyzing public opinions on significant public health issues, such as vaccine acceptance during a pandemic.
