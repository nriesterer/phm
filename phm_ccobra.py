import ccobra
import numpy as np
import pandas as pd

import sys
import os

import phm

class PHMModel(ccobra.CCobraModel):
    def __init__(self, name='PyPHM', khemlani_phrase=False, n_samples=4):
        super(PHMModel, self).__init__(name, ['syllogistic'], ['single-choice'])
        self.phm = phm.PHM(khemlani_phrase=khemlani_phrase)

        # Model parameters
        self.n_samples = n_samples

        # Individualization parameters
        self.best_param_dicts = []
        self.p_entailment = 0.04316547
        self.max_confidence = {'A': 0.88489209, 'I': 0.44604317, 'E': 0.25179856, 'O': 0.28776978}
        self.default_confidence = self.max_confidence

        # Prepare for training
        self.n_pre_train_dudes = 0
        self.pre_train_data = np.zeros((64, 9))
        self.history = np.zeros((64, 9))

    def end_participant(self, subj_id, model_log, **kwargs):
        model_log['p_entailment'] = self.p_entailment
        model_log['A_conf'] = self.max_confidence['A']
        model_log['I_conf'] = self.max_confidence['I']
        model_log['E_conf'] = self.max_confidence['E']
        model_log['O_conf'] = self.max_confidence['O']
        model_log['best_params'] = self.max_confidence

        print('Finalizing subject', subj_id)
        print('   p_entailm:', self.p_entailment)
        print('   A_conf   :', self.max_confidence['A'])
        print('   I_conf   :', self.max_confidence['I'])
        print('   E_conf   :', self.max_confidence['E'])
        print('   O_conf   :', self.max_confidence['O'])
        print()

    def pre_train(self, dataset):
        """ Pre-trains the model by fitting PHM.

        Parameters
        ----------
        dataset : list(list(dict(str, object)))
            Training data.

        """

        # Extract the training data to fit mReasoner with
        self.n_pre_train_dudes = len(dataset)
        self.pre_train_data = np.zeros((64, 9))
        for subj_data in dataset:
            for task_data in subj_data:
                item = task_data['item']
                enc_task = ccobra.syllogistic.encode_task(item.task)
                enc_resp = ccobra.syllogistic.encode_response(task_data['response'], item.task)

                task_idx = ccobra.syllogistic.SYLLOGISMS.index(enc_task)
                resp_idx = ccobra.syllogistic.RESPONSES.index(enc_resp)
                self.pre_train_data[task_idx, resp_idx] += 1

        div_mask = (self.pre_train_data.sum(axis=1) != 0)
        self.pre_train_data[div_mask] /= self.pre_train_data[div_mask].sum(axis=1, keepdims=True)

        # Fit the model
        self.fit()

    def pre_train_person(self, dataset, **kwargs):
        """ Perform the person training of mReasoner.

        """

        # Extract the training data to fit mReasoner with
        for task_data in dataset:
            item = task_data['item']
            enc_task = ccobra.syllogistic.encode_task(item.task)
            enc_resp = ccobra.syllogistic.encode_response(task_data['response'], item.task)

            task_idx = ccobra.syllogistic.SYLLOGISMS.index(enc_task)
            resp_idx = ccobra.syllogistic.RESPONSES.index(enc_resp)
            self.history[task_idx, resp_idx] += 1

        # Fit the model
        self.fit()

    def fit(self):
        # Merge the training datasets
        history_copy = self.history.copy()
        div_mask = (history_copy.sum(axis=1) != 0)
        history_copy[div_mask] /= history_copy[div_mask].sum(axis=1, keepdims=True)

        train_data = self.pre_train_data
        train_data[div_mask] = history_copy[div_mask]

        best_score = 0
        best_param_dicts = []

        # Iterate over parameters
        max_confidence_grid = [
            {'A': 1, 'I': 1, 'E': 1, 'O': 1},
            {'A': 1, 'I': 1, 'E': 1, 'O': 0},
            {'A': 1, 'I': 1, 'E': 0, 'O': 1},
            {'A': 1, 'I': 1, 'E': 0, 'O': 0},
            {'A': 1, 'I': 0, 'E': 0, 'O': 0},
            {'A': 0, 'I': 0, 'E': 0, 'O': 0}
        ]

        # Prepare the optimization loop
        for p_ent in [1, 0]:
            for max_conf in max_confidence_grid:
                param_dict = {
                    'p_entailment': p_ent,
                    'max_confidences': max_conf
                }

                self.p_entailment = p_ent
                self.max_confidence = max_conf

                # Obtain prediction matrix
                pred_mat = np.zeros((64, 9))
                for syl_idx, syllog in enumerate(ccobra.syllogistic.SYLLOGISMS):
                    for _ in range(self.n_samples):
                        pred = self._predict(syllog)
                        pred_idx = ccobra.syllogistic.RESPONSES.index(pred)
                        pred_mat[syl_idx, pred_idx] += 1

                pred_mat /= pred_mat.sum(axis=1, keepdims=True)

                # Compute score
                pred_mask = (pred_mat == pred_mat.max(axis=1, keepdims=True))
                score = np.sum(np.mean(train_data * pred_mask, axis=1))

                if score > best_score:
                    best_score = score
                    best_param_dicts = [param_dict]
                elif score == best_score:
                    best_param_dicts.append(param_dict)

        # Store best parameter configurations
        self.best_param_dicts = best_param_dicts

        # Apply best parameterization
        rnd_best = best_param_dicts[int(np.random.randint(0, len(best_param_dicts)))]
        self.p_entailment = rnd_best['p_entailment']
        self.max_confidence = rnd_best['max_confidences']

    def _predict(self, task_enc):
        # Obtain predictions
        use_p_entailment = self.p_entailment >= 0.5
        preds = self.phm.generate_conclusions(task_enc, use_p_entailment)

        # Apply the possible direction bias
        pred = np.random.choice(preds)

        # Apply max-heuristic
        if not self.phm.max_heuristic(task_enc, *[self.max_confidence[x] for x in ['A', 'I', 'E', 'O']]):
            pred = 'NVC'

        return pred

    def predict(self, item, **kwargs):
        task_enc = ccobra.syllogistic.encode_task(item.task)
        pred = self._predict(task_enc)
        return ccobra.syllogistic.decode_response(pred, item.task)

    def adapt(self, item, truth, **kwargs):
        # Encode syllogistic information
        enc_task = ccobra.syllogistic.encode_task(item.task)
        enc_resp = ccobra.syllogistic.encode_response(truth, item.task)

        # Update history
        task_idx = ccobra.syllogistic.SYLLOGISMS.index(enc_task)
        resp_idx = ccobra.syllogistic.RESPONSES.index(enc_resp)
        self.history[task_idx, resp_idx] += 1

        # Perform training
        self.fit()
