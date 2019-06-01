
from fastai.text import *
from fastai import *

_model_meta = {AWD_LSTM: {'hid_name': 'emb_sz', 'url': URLs.WT103_1,
                          'config_lm': awd_lstm_lm_config, 'split_lm': awd_lstm_lm_split,
                          'config_clas': awd_lstm_clas_config, 'split_clas': awd_lstm_clas_split},
               Transformer: {'hid_name': 'd_model', 'url': URLs.OPENAI_TRANSFORMER,
                             'config_lm': tfmer_lm_config, 'split_lm': tfmer_lm_split,
                             'config_clas': tfmer_clas_config, 'split_clas': tfmer_clas_split},
               TransformerXL: {'hid_name': 'd_model',
                               'config_lm': tfmerXL_lm_config, 'split_lm': tfmerXL_lm_split,
                               'config_clas': tfmerXL_clas_config, 'split_clas': tfmerXL_clas_split}}

data_lm = (TextList.from_csv('resources', 'partial-train.csv')
                   .split_by_rand_pct()
                   .label_for_lm()
                   .databunch())
# data_lm.save()
# data_lm.show_batch()

# TODO: Here i can fine-tune lm with smaller data

# learn = language_model_learner(data_lm, AWD_LSTM)
# learn.fit_one_cycle(1, 1e-2)
# learn.save('best_lm')
# learn.save_encoder('best_enc')

# learn.show_results()

data_clas = (TextList.from_csv('resources', 'fa.train.csv', cols='text', vocab=data_lm.vocab)
             .split_subsets(train_size=0.7, valid_size=0.3)
             .label_from_df(cols='data')
             .databunch(bs=16))
# data_clas.show_batch()


# # TODO: Load lm and fine-tune with classification data
# model = get_language_model(arch, len(data.vocab.itos), config=config, drop_mult=drop_mult)
# meta = _model_meta[arch]
# learn = LanguageLearner(data, model, split_func=meta['split_lm'], **learn_kwargs)



model = get_text_classifier(AWD_LSTM, len(data_lm.vocab.itos), data_clas.c, bptt=70, max_len=70*20,
                            config=None, drop_mult=1., lin_ftrs=None, ps=None)
meta = _model_meta[AWD_LSTM]
learn = RNNLearner(data_clas, model, split_func=meta['split_clas'])

learn.load_encoder('best_enc')
learn.fit_one_cycle(2, slice(1e-3, 1e-2))
learn.save('mini_train_clas')
learn.show_results()
