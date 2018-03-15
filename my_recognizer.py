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
    # TODO implement the recognizer
    # return probabilities, guesses

    all_Xlengths = test_set.get_all_Xlengths()
    for key in all_Xlengths:
        this_X, this_lengths = all_Xlengths[key]
        logL_distribution = {}
        for key in models:
            try:
                logL_distribution[key] = models[key].score(this_X, this_lengths)
            except:
                logL_distribution[key] = float("-inf")
                continue
        probabilities.append(logL_distribution)
        this_guess = sorted(logL_distribution,key=lambda x:logL_distribution[x])[-1]
        guesses.append(this_guess)
    return probabilities, guesses
