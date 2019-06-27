
from fastai.text import *
from utils import get_text_representation, _model_meta
from train_utils import train_lm, train_clf

# # # Data section
data_lm = (TextList.from_csv('resources', 'fa.wiki.small.csv')
                   .split_by_rand_pct()
                   .label_for_lm()
                   .databunch())
                   
data_clas = (TextList.from_csv('resources', 'fa.digikala.csv', cols='text', vocab=data_lm.vocab)
             .split_subsets(train_size=0.7, valid_size=0.3)
             .label_from_df(cols='data')
             .databunch(bs=16))

# data_lm.save()
# data_lm = pickle.load(open('resources/data_save.pkl', 'rb'))
# data_lm.show_batch()
# data_clas.show_batch()
print('LM vocab size', len(data_lm.vocab.itos),
', CLS vocab size', len(data_clas.vocab.itos),
', CLS n_class', data_clas.c)

# # # Train LM
lm_learner = train_lm()

# # # Fine-tuning language model section
# lm_model = get_language_model(AWD_LSTM, len(
#     data_lm.vocab.itos), config=None, drop_mult=1.)
# meta = _model_meta[AWD_LSTM]
# lm_learn = LanguageLearner(data_clas, lm_model, split_func=meta['split_lm'])
# fnames = [learn.path/learn.model_dir/f'{fn}.{ext}' for fn,ext in zip(pretrained_fnames, ['pth', 'pkl'])]
# learn.load_pretrained(*fnames)
# lm_learn.freeze()


# # # Classification section
cls_learner = train_clf()
print(cls_learner.summary())
print(get_text_representation(cls_learner, "سلام چطوری؟"))

