import torch.onnx as onnx
import torch
from fastai.text import *


_model_meta = {AWD_LSTM: {'hid_name': 'emb_sz', 'url': URLs.WT103_1,
                          'config_lm': awd_lstm_lm_config, 'split_lm': awd_lstm_lm_split,
                          'config_clas': awd_lstm_clas_config, 'split_clas': awd_lstm_clas_split},
               Transformer: {'hid_name': 'd_model', 'url': URLs.OPENAI_TRANSFORMER,
                             'config_lm': tfmer_lm_config, 'split_lm': tfmer_lm_split,
                             'config_clas': tfmer_clas_config, 'split_clas': tfmer_clas_split},
               TransformerXL: {'hid_name': 'd_model',
                               'config_lm': tfmerXL_lm_config, 'split_lm': tfmerXL_lm_split,
                               'config_clas': tfmerXL_clas_config, 'split_clas': tfmerXL_clas_split}}


torch.device = 'cpu'


lm_vocab_size = 2658
cls_c = 9
dummy_input = torch.randint(0, 1, (70*20, lm_vocab_size))
model = get_text_classifier(AWD_LSTM, lm_vocab_size, cls_c, bptt=70, max_len=70*20,
                            config=None, drop_mult=0.3, lin_ftrs=None, ps=None)

data_lm = (TextList.from_csv('resources', 'fa.wiki.small.csv').split_by_rand_pct().label_for_lm().databunch())
data_clas = (TextList.from_csv('resources', 'fa.digikala.csv', cols='text', vocab=data_lm.vocab).split_subsets(train_size=0.7, valid_size=0.3).label_from_df(cols='data').databunch(bs=16))

meta = _model_meta[AWD_LSTM]
cls_learn = RNNLearner(data_clas, model, split_func=meta['split_clas'])
cls_learn.load_encoder('best_enc')
# cls_learn.fit_one_cycle(1, slice(1e-3, 1e-2))
# cls_learn.save('mini_train_clas')
# cls_learn.show_results()

# print(model.state_dict())
print(model.summary())
# torch.save(model, 'resources/models/torch_model_cls')
# state_dict = torch.load('resources/models/mini_train_clas.pth', map_location='cpu')#lambda storage, loc: storage)
# model.load_state_dict(state_dict['model'])














# input_names = [ "actual_input_1" ] + [ "learned_%d" % i for i in range(16) ]
# output_names = [ "output1" ]

# onnx.export(model, dummy_input, "resources/mini_train_clas.onnx", verbose=True)#, input_names=input_names, output_names=output_names)

