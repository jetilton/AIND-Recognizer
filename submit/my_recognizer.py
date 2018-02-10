import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    words = test_set.wordlist

#    for word, model in models.items():
#        model = models[word]
#        dic = {}
#        for index,w in enumerate(words):
#            x,l = test_set.get_item_Xlengths(index)
#            try:score={w:model.score(x, l)}
#            except: score={w:float('inf')}
#            #print(score)
#            dic.update(score)
#        probabilities.append(dic)
#        guesses.append(max(dic.keys(), key = lambda x:dic[x]))
#    
#    return probabilities, guesses
        
    for index, word in enumerate(words):
        dic = {}
        x,l = test_set.get_item_Xlengths(index)
        for w, model in models.items():
            try:score = model.score(x,l)
            except: score = float('-inf')
            dic.update({w:score})
        probabilities.append(dic)
        guesses.append(max(dic.keys(), key = lambda x:dic[x]))
    
    return probabilities, guesses
    
        
#    for word in words:
#        
#        try:model = models[word]
#        except:probabilities.append({w:float('-inf') for w in words})
#        dic = {}
#        for index,w in enumerate(words):
#            x,l = test_set.get_item_Xlengths(index)
#            try:score={w:model.score(x, l)}
#            except: score={w:float('inf')}
#            #print(score)
#            dic.update(score)
#        probabilities.append(dic)
#        guesses.append(max(dic.keys(), key = lambda x:dic[x]))
#    
#    return probabilities, guesses
#    
