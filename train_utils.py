
from fastai.text import *
from utils import get_text_representation, _model_meta

def train_lm():
    learner = language_model_learner(data_lm, AWD_LSTM)
    learner.fit_one_cycle(1, 1e-2)
    learner.save('best_lm')
    learner.save_encoder('best_enc')
    # learner.show_results()
    return learner

def train_clf():
    text_classifier = get_text_classifier(AWD_LSTM, len(data_lm.vocab.itos), data_clas.c, bptt=70, max_len=70*20,
                            config=None, drop_mult=.3, lin_ftrs=None, ps=None)
    learner = RNNLearner(data_clas, text_classifier, split_func=_model_meta[AWD_LSTM]['split_clas'])
    learner.load_encoder('best_enc')
    # learner.fit_one_cycle(1, slice(1e-3, 1e-2))
    # learner.save('mini_train_clas')
    # learner.load('mini_train_clas')
    # learner.show_results()

    # learner.unfreeze()

    # learn.freeze_to(-2)
    # learn.fit_one_cycle(1, slice(5e-3/2., 5e-3))

    # learn.unfreeze()
    # learn.fit_one_cycle(1, slice(2e-3/100, 2e-3))
    return learner