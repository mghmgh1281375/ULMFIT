from fastai.text import *


data_lm = (TextList.from_csv('resources', 'partial-train.csv')
                   .split_by_rand_pct()
                   .label_for_lm()
                   .databunch())
# data_lm = TextList.from_folder('resources')

# data_lm.save()
# data_lm.show_batch()

learn = language_model_learner(data_lm, AWD_LSTM)
learn.fit_one_cycle(1, 1e-2)
learn.save('best_lm')
learn.save_encoder('best_enc')

# https://s3.amazonaws.com/fast-ai-modelzoo
