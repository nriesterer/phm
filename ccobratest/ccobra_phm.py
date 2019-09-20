import ccobra
import numpy as np
import pandas as pd

import sys
import os

sys.path.append(os.path.abspath(os.path.split(__file__)[0] + os.sep + '../phm'))
import phm

def max_premise(task):
    ordering = ['O', 'E', 'I', 'A']
    if ordering.index(task[0]) < ordering.index(task[1]):
        return task[1]
    else:
        return task[0]

class PHMModel(ccobra.CCobraModel):
    def __init__(self, name='PHM', khemlani_phrase=True, o_heur_enabled=True, max_heur_enabled=True, direction_bias_enabled=True, grid_fit=False):
        super(PHMModel, self).__init__(name, ['syllogistic'], ['single-choice'])
        self.phm = phm.PHM(khemlani_phrase=khemlani_phrase)

        # Member variables
        self.o_heur_enabled = o_heur_enabled
        self.max_heur_enabled = max_heur_enabled
        self.direction_bias_enabled = direction_bias_enabled
        self.grid_fit = grid_fit

        self.history = []

        # Individualization parameters
        self.p_entailment = 0
        self.direction_bias = 0
        self.o_confidence = 0
        self.max_confidence = {'A': [1, 1], 'I': [1, 1], 'E': [1, 1], 'O': [1, 1]}
        self.default_confidence = self.max_confidence

    def end_participant(self, subj_id, **kwargs):
        print('Finalizing subject', subj_id)
        print('   p_entailm:', self.p_entailment)
        print('   direction:', self.direction_bias)
        print('   o_confide:', self.o_confidence)
        print('   max_confi:', self.max_confidence)
        print()

    def pre_train(self, data, **kwargs):
        dat = []
        for subj_data in data:
            for task_data in subj_data:
                enc_task = ccobra.syllogistic.encode_task(task_data['item'].task)
                enc_resp = ccobra.syllogistic.encode_response(task_data['response'], task_data['item'].task)

                # Obtain max premise
                max_prem = max_premise(enc_task)

                # Obtain PHM responses
                phm_pred = self.phm.predict(enc_task) + ['NVC']

                dat.append({
                    'subj_id': task_data['item'].identifier,
                    'syllogism': enc_task,
                    'response': enc_resp,
                    'max_prem': max_prem,
                    'is_phm_pred': enc_resp in phm_pred
                })

        dat_df = pd.DataFrame(dat)
        dat_df['is_nvc'] = dat_df['response'] == 'NVC'

        # Filter phm preds, i.e., only consider responses which lie in the scope of PHM's
        # predictions. All responses, PHM is unable to produce are ignored.
        dat_df = dat_df.loc[dat_df['is_phm_pred']]
        max_heur_thresholds = dict(dat_df.groupby('max_prem')['is_nvc'].agg('mean'))

        # Max-heuristic confidence values. First tuple value represents weighted proportion of
        # NVC responses from the pre-training data. Second value denotes the inverse, i.e., the
        # weighted proportion of non-NVC conclusions.
        self.max_confidence = {
            'A': [max_heur_thresholds['A'] * 10, (1 - max_heur_thresholds['A']) * 10],
            'I': [max_heur_thresholds['I'] * 10, (1 - max_heur_thresholds['I']) * 10],
            'E': [max_heur_thresholds['E'] * 10, (1 - max_heur_thresholds['E']) * 10],
            'O': [max_heur_thresholds['O'] * 10, (1 - max_heur_thresholds['O']) * 10],
        }

        self.default_confidence = self.max_confidence

    def person_train(self, dataset, **kwargs):
        for task_data in dataset:
            item = task_data['item']
            truth = task_data['response']
            self.history.append((item, truth))

            if not self.grid_fit:
                self.adapt_inc(item, truth)

        if self.grid_fit:
            self.adapt_grid()

    def generative_predict(self, item, **kwargs):
        task_enc = ccobra.syllogistic.encode_task(item.task)

        # Obtain predictions
        preds = self.phm.predict(task_enc)
        min_concls = preds[:(len(preds) // 2)]
        pent_concls = preds[(len(preds) // 2):]

        preds = min_concls
        if self.p_entailment > 0:
            preds = pent_concls

        pred = np.random.choice(preds)
        if self.direction_bias_enabled:
            pred = preds[0]
            if self.direction_bias < 0 and len(preds) == 2:
                pred = preds[1]

        return pred

    def predict(self, item, **kwargs):
        # Generate a prediction from min-heuristic, attachment-heuristic and p-entailment
        pred = self.generative_predict(item, **kwargs)

        # Apply O-heuristic, i.e., replace O response with NVC
        if self.o_heur_enabled and pred[0] == 'O' and self.o_confidence <= 0:
            pred = 'NVC'

        # Apply max-heuristic
        enc_task = ccobra.syllogistic.encode_task(item.task)
        max_prem = max_premise(enc_task)
        max_conf = self.max_confidence[max_prem]
        if self.max_heur_enabled and (max_conf[0] / sum(max_conf) >= 0.5):
            pred = 'NVC'

        return ccobra.syllogistic.decode_response(pred, item.task)

    def adapt(self, item, truth, **kwargs):
        self.history.append((item, truth))

        if self.grid_fit:
            self.adapt_grid()
        else:
            self.adapt_inc(item, truth, **kwargs)

    def adapt_grid(self):
        best_score = 0
        best_p_ent = 0
        best_dir_bias = 0
        best_o_conf = 0
        best_max_conf = self.default_confidence

        max_confidence_grid = [
            {'A': [1, 0], 'I': [1, 0], 'E': [1, 0], 'O': [1, 0]},
            {'A': [1, 0], 'I': [1, 0], 'E': [1, 0], 'O': [0, 1]},
            {'A': [1, 0], 'I': [1, 0], 'E': [0, 1], 'O': [1, 0]},
            {'A': [1, 0], 'I': [1, 0], 'E': [0, 1], 'O': [0, 1]},
            {'A': [1, 0], 'I': [0, 1], 'E': [1, 0], 'O': [1, 0]},
            {'A': [1, 0], 'I': [0, 1], 'E': [1, 0], 'O': [0, 1]},
            {'A': [1, 0], 'I': [0, 1], 'E': [0, 1], 'O': [1, 0]},
            {'A': [1, 0], 'I': [0, 1], 'E': [0, 1], 'O': [0, 1]},
            {'A': [0, 1], 'I': [1, 0], 'E': [1, 0], 'O': [1, 0]},
            {'A': [0, 1], 'I': [1, 0], 'E': [1, 0], 'O': [0, 1]},
            {'A': [0, 1], 'I': [1, 0], 'E': [0, 1], 'O': [1, 0]},
            {'A': [0, 1], 'I': [1, 0], 'E': [0, 1], 'O': [0, 1]},
            {'A': [0, 1], 'I': [0, 1], 'E': [1, 0], 'O': [1, 0]},
            {'A': [0, 1], 'I': [0, 1], 'E': [1, 0], 'O': [0, 1]},
            {'A': [0, 1], 'I': [0, 1], 'E': [0, 1], 'O': [1, 0]},
            {'A': [0, 1], 'I': [0, 1], 'E': [0, 1], 'O': [0, 1]}
        ]

        for p_ent in [1, -1]:
            for dir_bias in [1, -1]:
                for o_conf in [1, -1]:
                    for max_conf in max_confidence_grid:
                        self.p_entailment = p_ent
                        self.direction_bias = dir_bias
                        self.o_confidence = o_conf
                        self.max_confidence = max_conf

                        score = 0
                        for elem in self.history:
                            item = elem[0]
                            truth = elem[1]

                            pred = self.predict(item)

                            if pred == truth:
                                score += 1

                        if score >= best_score:
                            best_score = score
                            best_p_ent = p_ent
                            best_dir_bias = dir_bias
                            best_o_conf = o_conf
                            best_max_conf = max_conf

        self.p_entailment = best_p_ent
        self.direction_bias = best_dir_bias
        self.o_confidence = best_o_conf
        self.max_confidence = best_max_conf

    def adapt_inc(self, item, truth, **kwargs):
        task_enc = ccobra.syllogistic.encode_task(item.task)
        truth_enc = ccobra.syllogistic.encode_response(truth, item.task)

        # Obtain predictions
        preds = self.phm.predict(task_enc)
        min_concls = preds[:(len(preds) // 2)]
        pent_concls = preds[(len(preds) // 2):]
        singlepred = self.generative_predict(item)

        # Update O-heuristic
        if singlepred[0] == 'O':
            if truth_enc == 'NVC':
                self.o_confidence -= 1
            elif truth_enc[0] == 'O':
                self.o_confidence += 1

        # Check if p-entailment is correct
        if truth_enc[0] in [x[0] for x in pent_concls]:
            self.p_entailment += 1
        elif truth_enc[0] in [x[0] for x in min_concls]:
            self.p_entailment -= 1

        # Check directionality
        if 'ac' in truth_enc:
            self.direction_bias += 1
        elif 'ca' in truth_enc:
            self.direction_bias -= 1

        # Adapt max-heuristic
        max_prem = max_premise(task_enc)
        if truth_enc == 'NVC':
            self.max_confidence[max_prem][0] += 1
        elif truth_enc in preds:
            self.max_confidence[max_prem][1] += 1