'''
Copyright 2015 Serendio Inc.
Author - Satish Palaniappan

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and limitations under the License.
'''
__author__ = "Satish Palaniappan"


import pickle
import config
import sys

from Sentiment.SocialFilter.Twokenize.twokenize import *
import re

class extractor(object):

	def __init__(self):
		self.path = config.basePath +config.general
		self.SentiModel = self.load_obj("_model")
		self.ch2 = self.load_obj("_feature_selector")
		self.vectorizer = self.load_obj("_vectorizer")

	def load_obj(self,name):
		    with open( self.path + name + '.pkl', 'rb') as f:
		        return pickle.load(f)

	def simpleProcess(self,text):
		text = text.lower().strip()
		line = re.sub(Url_RE,"",text)
		line = re.sub(r"[@#]","",line)
		line =u" ".join(tokenize(line))
		return line

	def getSentimentScore(self,message):
		vec = self.vectorizer.transform([self.simpleProcess(message)])
		Tvec = self.ch2.transform(vec)
		predScore = self.SentiModel.predict(Tvec)
		return float(predScore)

#### TEST
S = extractor()
# message = "I am very happy"
# # print(S.getSentimentScore(message))
# from scoreScaler import scorer
# print(scorer().scaleScore(S.getSentimentScore(message)))

import pandas as pd
import numpy as np
election_data = pd.read_csv('/home/azhar/python/NHCandidates 1.csv', encoding='utf-8')
comment_tag = election_data[['Comment','tags']]
comment_tag_clean = comment_tag.dropna()
comments = comment_tag_clean['Comment']
tags = comment_tag_clean['tags']

S = extractor()

comments = np.array(comments)
tags = np.array(tags)
total_correct = 0
for i in range(0,comments.shape[0]):
    # print i
    predictions = []
    true_values = []
    current_pred = ""
    from scoreScaler import scorer
    score_sentiment = scorer().scaleScore(S.getSentimentScore(comments[i]))
    if score_sentiment <= -2:
        predictions.append("untrustworthy")
        current_pred = "untrustworthy"
    elif score_sentiment >= 2:
        predictions.append("trustworthy")
        current_pred = "trustworthy"
    else:
        predictions.append("mixed")
        current_pred = "mixed"
    true_values.append(tags[i])
    print current_pred, tags[i],score_sentiment
    if current_pred == tags[i]:
        total_correct = total_correct + 1

print comment_tag_clean.shape[0], total_correct







