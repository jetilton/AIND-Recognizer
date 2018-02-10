import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        
        def bic(num_states):
            hmm_model = self.base_model(num_states)
            logL = hmm_model.score(self.X, self.lengths)
            N, f = self.X.shape
            m = num_states
            p = m**2 + 2*m*f-1
            logN = np.log(N)
            bic = -2 * logL + p * logN
            return bic
        score = float('inf')
        model = ''
        for state in range(self.min_n_components, self.max_n_components+1):
            try:
                if bic(state) < score: 
                    score = bic(state)
                    model = self.base_model(state)
            except: continue
                
        return model 
        #return self.base_model((min(range(self.min_n_components, self.max_n_components+1), key=bic)))


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores
        m = len(self.hwords)
        words = [x for x in self.words.keys() if x != self.this_word]
        dic = float('-inf')
        n = 0
        for state in range(self.min_n_components, self.max_n_components+1):
            try:
                model = self.base_model(state)
                logL = model.score(self.X, self.lengths)
                xl_list = [(self.hwords[word]) for word in words]
                logs = sum([model.score(*x) for x in xl_list])
                d = logL - 1 / (m-1) * logs
                if d > dic:
                    n = state
            except:pass
        return self.base_model(n)
        
        
        
        
        
        
#        def get_avg_log(num_states):
#            words = [x for x in self.words.keys() if x != self.this_word]
#            logL_list = []
#            for word in words:
#                try:
#                    x,l =  self.hwords[word]
#                    model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
#                                        random_state=self.random_state, verbose=False).fit(x, l)
#                    logL_list.append(model.score(x,l))
#                except: pass
#                    
#            return  sum(logL_list) / len(logL_list)
#        
#        score = float('-inf')
#        model = ''
#        for state in range(self.min_n_components, self.max_n_components+1):
#            try:
#                hmm_model = self.base_model(state)
#                logL = hmm_model.score(self.X, self.lengths)
#                avg_log = get_avg_log(state)
#                dic_score = logL - avg_log
#                if dic_score>score:
#                    score = dic_score
#                    model = hmm_model
#            except: pass
#        return model
        
class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection using CV
        def cv(num_states):
            folds = []
            splits = KFold(n_splits = min(3, len(self.sequences))).split(self.sequences)
            for train,test in splits:
                x_train, l_train = combine_sequences(train, self.sequences)
                x_test, l_test = combine_sequences(test, self.sequences)
                model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                            random_state=self.random_state, verbose=False).fit(x_train, l_train)
                folds.append(model.score(x_test, l_test))
            return sum(folds)/len(folds)
        
        score = float('-inf')
        model = ''
        for state in range(self.min_n_components, self.max_n_components+1):
            try:
                cv_score = cv(state)
                if cv_score > score:
                    score = cv_score
                    model = self.base_model(state)
            except: continue
        return model
            
            
            
            
            