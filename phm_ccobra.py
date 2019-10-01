import ccobra
import numpy as np
import pandas as pd

import sys
import os

import phm

class PHMModel(ccobra.CCobraModel):
    def __init__(self, name='PHM', khemlani_phrase=False, direction_bias_enabled=False, no_fit=False):
        super(PHMModel, self).__init__(name, ['syllogistic'], ['single-choice'])
        self.phm = phm.PHM(khemlani_phrase=khemlani_phrase)

        # Member variables
        self.direction_bias_enabled = direction_bias_enabled
        self.no_fit = no_fit

        # Individualization parameters
        self.history = []
        self.p_entailment = 0.04316547
        self.direction_bias = 0
        self.max_confidence = {'A': 0.88489209, 'I': 0.44604317, 'E': 0.25179856, 'O': 0.28776978}
        self.default_confidence = self.max_confidence

    def end_participant(self, subj_id, **kwargs):
        print('Finalizing subject', subj_id)
        print('   p_entailm:', self.p_entailment)
        print('   direction:', self.direction_bias)
        print('   max_confi:', self.max_confidence)
        print()

    def pre_train(self, data, **kwargs):
        if self.no_fit:
            return

        dat = []
        for subj_data in data:
            for task_data in subj_data:
                enc_task = ccobra.syllogistic.encode_task(task_data['item'].task)
                enc_resp = ccobra.syllogistic.encode_response(task_data['response'], task_data['item'].task)

                # Obtain max premise
                max_prem = phm.max_premise(enc_task)

                # Obtain PHM responses
                min_concls = self.phm.generate_conclusions(enc_task, False)
                pent_concls = self.phm.generate_conclusions(enc_task, True)
                phm_pred = min_concls + pent_concls + ['NVC']

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
            'A': 1 - max_heur_thresholds['A'],
            'I': 1 - max_heur_thresholds['I'],
            'E': 1 - max_heur_thresholds['E'],
            'O': 1 - max_heur_thresholds['O']
        }

        self.default_confidence = self.max_confidence

    def person_train(self, dataset, **kwargs):
        if self.no_fit:
            return

        for task_data in dataset:
            item = task_data['item']
            truth = task_data['response']
            self.history.append((item, truth))

        self.adapt_grid()

    def predict(self, item, **kwargs):
        task_enc = ccobra.syllogistic.encode_task(item.task)

        # Obtain predictions
        use_p_entailment = self.p_entailment >= 0.5
        preds = self.phm.generate_conclusions(task_enc, use_p_entailment)

        # Apply the possible direction bias
        pred = np.random.choice(preds)
        if self.direction_bias_enabled:
            pred = preds[0]
            if self.direction_bias < 0 and len(preds) == 2:
                pred = preds[1]

        # Apply max-heuristic
        if not self.phm.max_heuristic(task_enc, *[self.max_confidence[x] for x in ['A', 'I', 'E', 'O']]):
            pred = 'NVC'

        return ccobra.syllogistic.decode_response(pred, item.task)

    def adapt(self, item, truth, **kwargs):
        if self.no_fit:
            return

        self.history.append((item, truth))
        self.adapt_grid()

    def adapt_grid(self):
        best_score = 0
        best_p_ent = 0
        best_dir_bias = 0
        best_max_conf = self.default_confidence

        max_confidence_grid = [
            {'A': 1, 'I': 1, 'E': 1, 'O': 1},
            {'A': 1, 'I': 1, 'E': 1, 'O': 0},
            {'A': 1, 'I': 1, 'E': 0, 'O': 1},
            {'A': 1, 'I': 1, 'E': 0, 'O': 0},
            {'A': 1, 'I': 0, 'E': 0, 'O': 0},
            {'A': 0, 'I': 0, 'E': 0, 'O': 0}
        ]

        for p_ent in [1, 0]:
            for dir_bias in [1, 0]:
                for max_conf in max_confidence_grid:
                    self.p_entailment = p_ent
                    self.direction_bias = dir_bias
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
                        best_max_conf = max_conf

        self.p_entailment = best_p_ent
        self.direction_bias = best_dir_bias
        self.max_confidence = best_max_conf
