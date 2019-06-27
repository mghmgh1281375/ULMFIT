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

def get_text_representation(learner, text):
    representations = []
    hook = lambda module, input, output: representations.append(output)
    learner.layer_groups[-1][0].layers[2].register_forward_hook(hook)

    pred = learner.predict(text)
    return {'pred': pred, 'repr': representations}