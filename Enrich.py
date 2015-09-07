import re
import nltk
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
        self.wordnet_lemmatizer = WordNetLemmatizer()
        self.stop_list = set(stopwords.words('english'))
        self.stop_list1 = set(stopwords.words('english'))
        self.senti_object = extractor()
        with open('trust_model.pickle', 'rb') as handle:
            self.model = pickle.load(handle)
        with open('feature_names.pickle','rb') as handle:
            self.feature_names = pickle.load(handle)
        with open('vectorizer.pickle','rb') as handle:
            self.vectorizer = pickle.load(handle)
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
                
    def getNonAlphaNumericCount(self,text):
        count = 0
        for char in text:
            if not (char.isalpha() or char.isdigit()):
                count += 1
        return count
       
    def checkIgnoreWords(self,text):
        if(len(text) > 400 ):
            return True        
        if(text in self.stop_list):
            return True
        if((self.getNonAlphaNumericCount(text)/len(text) ) > 0.5):
            return True
        return False
    
    def extract_entity_names(self,t):
        entity_names = []
        if hasattr(t, 'label') and t.label:
            if t.label() == 'NE':
                entity_names.append(' '.join([child[0] for child in t]))
            else:
                for child in t:
                    entity_names.extend(self.extract_entity_names(child))
        return entity_names
            
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

    def document_to_wordlist_syn( self,doc,feature_names,fastmode):

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


        stops = self.stop_list1
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
        
        if fastmode == False:
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

    def clean_test_data(self,X,feature_names,fastmode):
        clean = []
        for i in xrange( 0, len(X)):
            clean.append(" ".join(self.document_to_wordlist_syn(X[i],feature_names,fastmode)))
        return clean

    def getTrustworthiness(self,comment,fastmode = True):
        ###Load pickl files
        
        comment = [comment]
        comment_clean = self.clean_test_data(comment,self.feature_names,fastmode)
        # print feature_names
            # print comment_clean
        comment_features = self.vectorizer.transform(comment_clean)
        comment_features = comment_features.toarray()
        prediction = self.model.predict(comment_features)
        # print prediction
        if prediction == 1:
            return "Trustworthy"
        else:
            return "Untrustworthy"

    def getSentiment(self,text):        
        score_sentiment = scorer().scaleScore(self.senti_object.getSentimentScore(text))
        if score_sentiment <= -2:
            return "Negative"
        elif score_sentiment >= 2:
            return "Positive"
        else:
            return "Neutral"
    
    def getLemmaList(self,text):
        text_tokens = nltk.word_tokenize(text)
        lemma_list = []
        for x in text_tokens:
            x = x.lower()
            if(self.checkIgnoreWords(x) == False):
                x = re.sub('s/[^a-z0-9]//ig','',x)
                x = self.wordnet_lemmatizer.lemmatize(x)
                if (len(x) > 3):
                    lemma_list.append(x)
        return lemma_list
    
    def getNamedEntities(self,text):
        sentences = nltk.sent_tokenize(text)
        tokenized_sentences = [nltk.word_tokenize(sentence) for sentence in sentences]
        tagged_sentences = [nltk.pos_tag(sentence) for sentence in tokenized_sentences]
        chunked_sentences = nltk.ne_chunk_sents(tagged_sentences, binary=True)
        entity_names = []
        for tree in chunked_sentences:
            entity_names.extend(self.extract_entity_names(tree))
        return entity_names  
            
senrich = SynapsifyEnrich()
print "Keyword Dictionary: ",senrich.keyword_dict
print "Candidate: ",senrich.getCandidateMatch('hillary clinton is a strong advocate for the minimum wage program')
print "Topics: ",senrich.getTopics('hillary clinton is a strong advocate for the minimum wage program 911')
print "Sentiment: ",senrich.getSentiment('hillary clinton is a strong advocate for the minimum wage program 911')
print "Trustworthiness: ",senrich.getTrustworthiness('hillary clinton is a blatant liar')
print "Lemma List: ",senrich.getLemmaList('hillary clinton is a blatant @.Int237838273 churches $$$$.2 liar')
print "Named entity List: ",senrich.getNamedEntities('Hillary Clinton is one of the presidential candidates for USA')
