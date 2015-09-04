import re
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
import pickle
from general import extractor
from scoreScaler import scorer
class SynapsifyEnrich:
    keyword_dict = {}
    candidate_list = []
    keyword_file = 'keyword.txt'
    candidate_file = 'candidate.txt'    
    
    def __init__(self):
        for topics in ['Civil Rights','Economy','Education','Energy','Entitlements','Foreign Policy','Government and Political reform','Immigration','Jobs','National Security']:
            with open(topics+'.txt') as f:
                content = f.readlines()
                temp = []
                for keywords in content:
                    keywords = self.preProcessKeywords(keywords)
                    temp.append(keywords)
                self.keyword_dict[topics] = temp
                   
        with open(self.candidate_file) as f:
            content = f.readlines()
            for x in content:
                x = self.preProcessKeywords(x)
                self.candidate_list.append(x)
                
    def preProcessKeywords(self,text):       
        text = text.strip().lower()
        text = ''.join(i for i in text if ord(i)<128)
        text = re.sub('s/\s/(?:\s|-){0,2}/',' ',text)
        return text
    
    def getCandidateMatch(self,text):
        candidate_match = []
        for x in self.candidate_list:
            if text.find(x) != -1:
                candidate_match.append(x)
        return candidate_match
    
    def getTopics(self,text):
        topics_match = []
        for topics in self.keyword_dict:            
            regex_keys = r'\b|\b'.join(self.keyword_dict[topics])
            keywords_regex = re.compile(r'(\b{0}\b)'.format(regex_keys),re.UNICODE | re.IGNORECASE)
            # print keywords_regex.pattern
            if re.search(keywords_regex,text) != None: 
                topics_match.append(topics)
        return topics_match

    def document_to_wordlist_syn( self,doc,feature_names):

        review_text = re.sub("[^a-zA-Z]"," ", doc)

        words = review_text.lower().split()

        lmtzr = WordNetLemmatizer()
        word_lmtzr = []
        for w in words:
            word_lmtzr.append(lmtzr.lemmatize(w))


        ###Remove words of length 2
        words_len2 = []
        for w in word_lmtzr:
            if 3<= len(w) <= 25:
                words_len2.append(w)


        stops = set(stopwords.words("english"))
        new_words = ["hillary","clinton","hillaryclinton","hilary","trump","donald","donaldtrump","rt","bush",
                     "abc","abcs","yrs","yet","yea","yeah","ya","might","yes","reagan","used","usual"
                     "use","george","exactly","thiessen","seem","cruz","powell","msnbc","still",
                     "although","know","anyway","want","actually","must","ronald","ted","cruz","whole",
                     "whatever","something","mccain"]
        stops.update(new_words)
        neg_words = ["dont","not","wasnt","havent","doesnt","havent","wouldnt","wont","hasnt","isnt","werent","hadnt","didnt",
                     "couldnt","shouldnt","non"]
        stops_new = stops.difference(neg_words)
        words = [w for w in words_len2 if not w in stops_new]

        for w in words:

            if w in feature_names:
                continue
            else:
                syns = wn.synsets(w)
                # print "here"
                list_of_syn = set()
                for s in syns:
                    list_of_syn.update(s.lemma_names())
                for synonym in list_of_syn:
                    if synonym in feature_names:
                        words.append(synonym)

        return(words)

    def clean_test_data(self,X,feature_names):
        clean = []
        for i in xrange( 0, len(X)):
            clean.append(" ".join(self.document_to_wordlist_syn(X[i],feature_names)))
        return clean

    def getTrustwothiness(self,comment):
        ###Load pickl files
        with open('trust_model.pickle', 'rb') as handle:
            model = pickle.load(handle)
        with open('feature_names.pickle','rb') as handle:
            feature_names = pickle.load(handle)
        with open('vectorizer.pickle','rb') as handle:
            vectorizer = pickle.load(handle)
        comment = [comment]
        comment_clean = self.clean_test_data(comment,feature_names)
        # print feature_names
            # print comment_clean
        comment_features = vectorizer.transform(comment_clean)
        comment_features = comment_features.toarray()
        prediction = model.predict(comment_features)
        # print prediction
        if prediction == 1:
            return "Trustworthy"
        else:
            return "Untrustworthy"

    def getSentiment(self,text):

        S = extractor()
        score_sentiment = scorer().scaleScore(S.getSentimentScore(text))

        if score_sentiment <= -2:
            return "Negative"
        elif score_sentiment >= 2:
            return "Positive"
        else:
            return "Neutral"

  
senrich = SynapsifyEnrich()
print "Keyword Dictionary: ",senrich.keyword_dict
print "Candidate: ",senrich.getCandidateMatch('hillary clinton is a strong advocate for the minimum wage program')
print "Topics: ",senrich.getTopics('hillary clinton is a strong advocate for the minimum wage program 911')
print "Sentiment: ",senrich.getSentiment('hillary clinton is a strong advocate for the minimum wage program 911')
print "Trustworthiness: ",senrich.getTrustwothiness('hillary clinton is a blatant liar')
