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
        self.all_word_Xlengths = all_word_Xlengths
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
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        def BIC_calculate(logL, num_param, size_data):
            bicValue = -2 * logL + num_param * np.log(size_data)
            return bicValue
        bestBic = float("inf")
        bestModel = 0
        for num_s in range(self.min_n_components, self.max_n_components - 1):
            currentModel = self.base_model(num_s)
            # calculate the maxmium likelihood estimate which implies the fitting
            # level between the object model and data.
            logL = currentModel.score(self.X, self.lengths)
            # get the size of the observation data set and number of features
            size_data, num_feature = self.X.shape
            # calculate the number of free parameters in this model
            num_param = num_s ** 2 + 2 * num_s * num_feature - 1
            # calculate the bic
            bicValue = BIC_calculate(logL, num_param, size_data)
            if bicValue < bestBic:
                bestBic = bicValue
                bestModel = currentModel
            else:
                continue
        return bestModel




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
        
        def antiLogL_calculate(model):
            M = len(self.all_word_Xlengths)
            antiLogL_sum = 0
            for key in self.all_word_Xlengths:
                if key == self.this_word:
                    continue
                else:
                    X, lengths = self.all_word_Xlengths[key]
                    antiLogL_sum += model.score(X, lengths)
            return antiLogL_sum/(M-1)
            
        def DIC_calculate(logL, antiLogL_average):
            return logL - antiLogL_average
        
        bestDic = float("-inf")
        bestModel = 0
        for num_s in range(self.min_n_components, self.max_n_components - 1):
            currentModel = self.base_model(num_s)
            logL = currentModel.score(self.X, self.lengths)
            antiLogL_average = antiLogL_calculate(currentModel)
            dicValue = DIC_calculate(logL, antiLogL_average)
            if dicValue > bestDic:
                bestDic = dicValue
                bestModel = currentModel
            else:
                continue
        return bestModel
            


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        new_sequences = KFold(self.sequences)
        bestCv = float("-inf")
        bestModel = 0
        X_base = self.X
        lengths_base = self.lengths
        for num_s in range(self.min_n_components, self.max_n_components - 1):
            logL_sum = 0
            for cv_train_idx, cv_test_idx in new_sequences:
                self.X, self.lengths = combine_sequences(cv_train_idx, new_sequences)
                X_test, lengths_test = combine_sequences(cv_test_idx, new_sequences)
                currentModel = self.base_model(num_s)
                logL_sum += currentModel.score(X_test, lengths_test)
            self.X = X_base
            self.lengths = lengths_base
            if logL_sum > bestCv:
                bestCv = logL_sum
                bestModel = self.base_model(num_s)
            else:
                continue
        return bestModel                
            