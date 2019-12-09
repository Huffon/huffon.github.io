---
layout: post
title: "그림으로 이해하는 BERT"
subtitle: '시각 자료를 통해 이해하는 BERT 사용기'
author: "devfon"
header-style: text
lang: kr
tags:
  - AI
  - NLP
  - Deep Learning
---

<div class="img-div-any-width" markdown="0">
  <image src="/img/in-post/distilBERT/bert-distilbert-sentence-classification.png"/>
  <br />
</div>

지난 몇 년간 기계학습을 활용한 자연어 처리 기술에는 많은 발전이 있어왔습니다. 그리고 이제 발전된 자연어 처리 기술들은 연구실을 넘어 우리가 실제 사용하는 제품에까지 적용되고 있습니다. 이 중 가장 좋은 사례는 최근 [Google이 검색 엔진에 자연어 처리 모델인 BERT를 적용](https://www.blog.google/products/search/search-language-understanding-bert/)하기로 결정했다는 소식입니다. Google은 해당 결정(혹은 검색에 있어서 적용될 수 있을만큼 발전한 자연어 이해 기술)이 지난 5년 중 가장 큰 한 걸음이자, 

represents "the biggest leap forward in the past five years, and one of the biggest leaps forward in the history of Search".

이번 포스트는 문장 분류 작업을 수행하기 위해 BERT를 사용하는 방법을 다루는 간단한 튜토리얼입니다. BERT를 처음 사용하시는 분들도 이해할 수 있을 정도로 기본적이지만 이와 동시에, BERT를 이해함에 있어 중요한 개념들을 보기에 충분히 기술적인 예제입니다. 

포스트와 더불어 
Alongside this post, I've prepared a notebook. You can see it here [the notebook](https://github.com/jalammar/jalammar.github.io/blob/master/notebooks/bert/A_Visual_Notebook_to_Using_BERT_for_the_First_Time.ipynb) or [run it on colab](https://colab.research.google.com/github/jalammar/jalammar.github.io/blob/master/notebooks/bert/A_Visual_Notebook_to_Using_BERT_for_the_First_Time.ipynb).
<!--more-->


## Dataset: SST2
우리가 이번 예제에서 사용할 데이터셋은 [SST2](https://nlp.stanford.edu/sentiment/index.html)입니다. SST2는 [NSMC](https://github.com/e9t/nsmc)와 유사한 데이터셋으로 영화 리뷰 문장이 긍정일 경우 라벨 값으로 1을, 부정일 경우 라벨 값으로 0을 가집니다.

<table class="features-table">
  <tr>
    <th class="mdc-text-light-green-600">
    sentence
    </th>
    <th class="mdc-text-purple-600">
    label
    </th>
  </tr>
  <tr>
    <td class="mdc-bg-light-green-50" style="text-align:left">
      a stirring , funny and finally transporting re imagining of beauty and the beast and 1930s horror films
    </td>
    <td class="mdc-bg-purple-50">
      1
    </td>
  </tr>
  <tr>
    <td class="mdc-bg-light-green-50" style="text-align:left">
      apparently reassembled from the cutting room floor of any given daytime soap
    </td>
    <td class="mdc-bg-purple-50">
      0
    </td>
  </tr>
  <tr>
    <td class="mdc-bg-light-green-50" style="text-align:left">
      they presume their audience won't sit still for a sociology lesson
    </td>
    <td class="mdc-bg-purple-50">
      0
    </td>
  </tr>
  <tr>
    <td class="mdc-bg-light-green-50" style="text-align:left">
      this is a visually stunning rumination on love , memory , history and the war between art and commerce
    </td>
    <td class="mdc-bg-purple-50">
      1
    </td>
  </tr>
  <tr>
    <td class="mdc-bg-light-green-50" style="text-align:left">
      jonathan parker 's bartleby should have been the be all end all of the modern office anomie films
    </td>
    <td class="mdc-bg-purple-50">
      1
    </td>
  </tr>
</table>


## 모델: 문장의 감정 분석
우리의 목표는 위에서 살펴본 SST2 데이터셋 내 문장을 입력 값으로 받아 긍정의 '1' 혹은 부정의 '0'을 예측해내는 모델을 만드는 것입니다. 우리는 모델이 수행하는 작업을 다음 그림과 같이 생각할 수 있습니다.
Our goal is to create a model that takes a sentence (just like the ones in our dataset) and produces either 1 (indicating the sentence carries a positive sentiment) or a 0 (indicating the sentence carries a negative sentiment). We can think of it as looking like this:


<div class="img-div-any-width" markdown="0">
  <image src="/img/in-post/distilBERT/sentiment-classifier-1.png"/>
  <br />
</div>


Under the hood, the model is actually made up of two model.
 *  [DistilBERT](https://medium.com/huggingface/distilbert-8cf3380435b5)  processes the sentence and passes along some information it extracted from it on to the next model. DistilBERT is a smaller version of BERT developed and open sourced by the team at [HuggingFace](https://huggingface.co/). It's a lighter and faster version of BERT that roughly matches its performance.
 * The next model, a basic Logistic Regression model from scikit learn will take in the result of DistilBERT's processing, and classify the  sentence as either positive or negative (1 or 0, respectively).

The data we pass between the two models is a vector of size 768. We can think of this of vector as an embedding for the sentence that we can use for classification.


<div class="img-div-any-width" markdown="0">
  <image src="/img/in-post/distilBERT/distilbert-bert-sentiment-classifier.png"/>
  <br />
</div>

If you've read my previous post, [Illustrated BERT](/illustrated-bert/), this vector is the result of the first position (which receives the [CLS] token as input).

## 모델 훈련 과정
While we'll be using two models, we will only train the logistic regression model. For DistillBERT, we'll use a model that's already pre-trained and has a grasp on the English language. This model, however is neither trained not fine-tuned to do sentence classification. We get some sentence classification capability, however, from the general objectives BERT is trained on. This is especially the case with BERT's output for the first position (associated with the [CLS] token). I believe that's due to BERT's second training object -- Next sentence classification. That objective seemingly trains the model to encapsulate a sentence-wide sense to the output at the first position. The [transformers](https://github.com/huggingface/transformers) library provides us with an implementation of DistilBERT as well as pretrained versions of the model.


<div class="img-div-any-width" markdown="0">
  <image src="/img/in-post/distilBERT/model-training.png"/>
  <br />
</div>

## Tutorial Overview
So here's the game plan with this tutorial. We will first use the trained distilBERT to generate sentence embeddings for 2,000 sentences.


<div class="img-div-any-width" markdown="0">
  <image src="/img/in-post/distilBERT/bert-distilbert-tutorial-sentence-embedding.png"/>
  <br />
</div>

We will not touch distilBERT after this step. It's all Scikit Learn from here. We do the usual train/test split on this dataset:

<div class="img-div-any-width" markdown="0">
  <image src="/img/in-post/distilBERT/bert-distilbert-train-test-split-sentence-embedding.png"/>
  <br />
  Train/test split for the output of distilBert (model #1) creates the dataset we'll train and evaluate logistic regression on (model #2). Note that in reality, sklearn's train/test split shuffles the examples before making the split, it doesn't just take the first 75% of examples as they appear in the dataset.
</div>

Then we train the logistic regression model on the training set:

<div class="img-div-any-width" markdown="0">
  <image src="/img/in-post/distilBERT/bert-training-logistic-regression.png"/>
  <br />
</div>


## How a single prediction is calculated
Before we dig into the code and explain how to train the model, let's look at how a trained model calculates its prediction.

Let's try to classify the sentence “a visually stunning rumination on love”. The first step is to use the BERT tokenizer to first split the word into tokens. Then, we add the special tokens needed for sentence classifications (these are [CLS] at the first position, and [SEP] at the end of the sentence).

<div class="img-div-any-width" markdown="0">
  <image src="/img/in-post/distilBERT/bert-distilbert-tokenization-1.png"/>
  <br />
</div>

The third step the tokenizer does is to replace each token with its id from the embedding table which is a component we get with the trained model. Read [The Illustrated Word2vec](/illustrated-word2vec/) for a background on word embeddings.

<div class="img-div-any-width" markdown="0">
  <image src="/img/in-post/distilBERT/bert-distilbert-tokenization-2-token-ids.png"/>
  <br />
</div>
Note that the tokenizer does all these steps in a single line of code:
```python
tokenizer.encode("a visually stunning rumination on love", add_special_tokens=True)
```

Our input sentence is now the proper shape to be passed to DistilBERT.

If you've read [Illustrated BERT](/illustrated-bert/), this step can also be visualized in this manner:

<div class="img-div-any-width" markdown="0">
  <image src="/img/in-post/distilBERT/bert-distilbert-input-tokenization.png"/>
  <br />
</div>


## Flowing Through DistilBERT

Passing the input vector through DistilBERT works [just like BERT](/illustrated-bert/). The output would be a vector for each input token. each vector is made up of 768 numbers (floats).

<div class="img-div-any-width" markdown="0">
  <image src="/img/in-post/distilBERT/bert-model-input-output-1.png"/>
  <br />
</div>

Because this is a sentence classification task, we ignore all except the first vector (the one associated with the [CLS] token). The one vector we pass as the input to the logistic regression model.

<div class="img-div-any-width" markdown="0">
  <image src="/img/in-post/distilBERT/bert-model-calssification-output-vector-cls.png"/>
  <br />
</div>

From here, it's the logistic regression model's job to classify this vector based on what it learned from its training phase. We can think of a prediction calculation as looking like this:

<div class="img-div-any-width" markdown="0">
  <image src="/img/in-post/distilBERT/bert-distilbert-sentence-classification-example.png"/>
  <br />
</div>

The training is what we'll discuss in the next section, along with the code of the entire process.


## The Code
In this section we'll highlight the code to train this sentence classification model. A notebook containing all this code is available on [colab](https://colab.research.google.com/github/jalammar/jalammar.github.io/blob/master/notebooks/bert/A_Visual_Notebook_to_Using_BERT_for_the_First_Time.ipynb) and [github](https://github.com/jalammar/jalammar.github.io/blob/master/notebooks/bert/A_Visual_Notebook_to_Using_BERT_for_the_First_Time.ipynb).

Let's start by importing the tools of the trade

```python
import numpy as np
import pandas as pd
import torch
import transformers as ppb # pytorch transformers
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
```


The dataset is [available](https://github.com/clairett/pytorch-sentiment-classification/) as a file on github, so we just import it directly into a pandas dataframe

```python
df = pd.read_csv('https://github.com/clairett/pytorch-sentiment-classification/raw/master/data/SST2/train.tsv', delimiter='\t', header=None)
```

We can use df.head() to look at the first five rows of the dataframe to see how the data looks.

```python
df.head()
```
Which outputs:

<div class="img-div-any-width" markdown="0">
  <image src="/img/in-post/distilBERT/sst2-df-head.png"/>
  <br />
</div>

### Importing pre-trained DistilBERT model and tokenizer

```python
model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')

## Want BERT instead of distilBERT? Uncomment the following line:
#model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased')

# Load pretrained model/tokenizer
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model = model_class.from_pretrained(pretrained_weights)
```

We can now tokenize the dataset. Note that we're going to do things a little differently here from the example above. The example above tokenized and processed only one sentence. Here, we'll tokenize and process all sentences together as a batch (the notebook processes a smaller group of examples just for resource considerations, let's say 2000 examples).

### Tokenization
```python
tokenized = df[0].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))
```

This turns every sentence into the list of ids.

<div class="img-div-any-width" markdown="0">
  <image src="/img/in-post/distilBERT/sst2-text-to-tokenized-ids-bert-example.png"/>
  <br />
</div>

The dataset is currently a list (or pandas Series/DataFrame) of lists. Before DistilBERT can process this as input, we'll need to make all the vectors the same size by padding shorter sentences with the token id 0. You can refer to the notebook for the padding step, it's basic python string and array manipulation.

After the padding, we have a matrix/tensor that is ready to be passed to BERT:

<div class="img-div-any-width" markdown="0">
  <image src="/img/in-post/distilBERT/bert-input-tensor.png"/>
  <br />
</div>



### Processing with DistilBERT

We now create an input tensor out of the padded token matrix, and send that to DistilBERT

```python
input_ids = torch.tensor(np.array(padded))

with torch.no_grad():
    last_hidden_states = model(input_ids)
```

After running this step, `last_hidden_states` holds the outputs of DistilBERT. It is a tuple with the shape (number of examples, max number of tokens in the sequence, number of hidden units in the DistilBERT model). In our case, this will be 2000 (since we only limited ourselves to 2000 examples), 66 (which is the number of tokens in the longest sequence from the 2000 examples), 768 (the number of hidden units in the DistilBERT model).

<div class="img-div-any-width" markdown="0">
  <image src="/img/in-post/distilBERT/bert-distilbert-output-tensor-predictions.png"/>
  <br />
</div>

### BERT 출력 텐서 Unpacking
이제 3차원의 출력 값을 Unpack 해봅시다.
We can first start by examining its dimensions:

<div class="img-div-any-width" markdown="0">
  <image src="/img/in-post/distilBERT/bert-output-tensor.png"/>
  <br />
</div>

### Recapping a sentence's journey
Each row is associated with a sentence from our dataset. To recap the processing path of the first sentence, we can think of it as looking like this:

<div class="img-div-any-width" markdown="0">
  <image src="/img/in-post/distilBERT/bert-input-to-output-tensor-recap.png"/>
  <br />
</div>


### Slicing the important part
For sentence classification, we're only only interested in BERT's output for the [CLS] token, so we select that slice of the cube and discard everything else.

<div class="img-div-any-width" markdown="0">
  <image src="/img/in-post/distilBERT/bert-output-tensor-selection.png"/>
  <br />
</div>

This is how we slice that 3d tensor to get the 2d tensor we're interested in:

```python
 # Slice the output for the first position for all the sequences, take all hidden unit outputs
features = last_hidden_states[0][:,0,:].numpy()
```

And now `features` is a 2d numpy array containing the sentence embeddings of all the sentences in our dataset.

<div class="img-div-any-width" markdown="0">
  <image src="/img/in-post/distilBERT/bert-output-cls-senteence-embeddings.png"/>
  <br />
  The tensor we sliced from BERT's output
</div>


## 로지스틱 회귀를 위한 데이터셋
Now that we have the output of BERT, we have assembled the dataset we need to train our logistic regression model. The 768 columns are the features, and the labels we just get from our initial dataset.


<div class="img-div-any-width" markdown="0">
  <image src="/img/in-post/distilBERT/logistic-regression-dataset-features-labels.png"/>
  <br />
  The labeled dataset we use to train the Logistic Regression. The features are the output vectors of BERT for the [CLS] token (position #0) that we sliced in the previous figure. Each row corresponds to a sentence in our dataset, each column corresponds to the output of a hidden unit from the feed-forward neural network at the top transformer block of the Bert/DistilBERT model.
</div>

After doing the traditional train/test split of machine learning, we can declare our Logistic Regression model and train it against the dataset.


```python
labels = df[1]
train_features, test_features, train_labels, test_labels = train_test_split(features, labels)

```
Which splits the dataset into training/testing sets:


<div class="img-div-any-width" markdown="0">
  <image src="/img/in-post/distilBERT/bert-distilbert-train-test-split-sentence-embedding.png"/>
  <br />
</div>

Next, we train the Logistic Regression model on the training set.

```python
lr_clf = LogisticRegression()
lr_clf.fit(train_features, train_labels)
```

Now that the model is trained, we can score it against the test set:

```python
lr_clf.score(test_features, test_labels)
```

Which shows the model achieves around 81% accuracy.

## 벤치마크 지표
For reference, the highest accuracy score for this dataset is currently **96.8**. DistilBERT can be trained to improve its score on this task -- a process called fine-tuning which updates BERT's weights to make it achieve a better performance in the sentence classification (which we can call the *downstream task*). The fine-tuned DistilBERT turns out to achieve an accuracy score of **90.7**. The full size BERT model achieves **94.9**.

## 노트북 자료
Dive right into [the notebook](https://github.com/jalammar/jalammar.github.io/blob/master/notebooks/bert/A_Visual_Notebook_to_Using_BERT_for_the_First_Time.ipynb) or [run it on colab](https://colab.research.google.com/github/jalammar/jalammar.github.io/blob/master/notebooks/bert/A_Visual_Notebook_to_Using_BERT_for_the_First_Time.ipynb).


And that's it! That's a good first contact with BERT. The next step would be to head over to the documentation and try your hand at [fine-tuning](https://huggingface.co/transformers/examples.html#glue). You can also go back and switch from distilBERT to BERT and see how that works.

Thanks to [Clément Delangue](https://twitter.com/ClementDelangue), [Victor Sanh
](https://twitter.com/SanhEstPasMoi), and the Huggingface team for providing feedback to earlier versions of this tutorial.
