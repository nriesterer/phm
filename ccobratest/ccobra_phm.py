import ccobra
import numpy as np

import sys
import os

sys.path.append(os.path.abspath(os.path.split(__file__)[0] + os.sep + '../phm'))
import phm

class PHMModel(ccobra.CCobraModel):
    def __init__(self, name='PHM', khemlani_phrase=True):
        super(PHMModel, self).__init__(name, ['syllogistic'], ['single-choice'])
        self.phm = phm.PHM(khemlani_phrase=khemlani_phrase)

        # Individualization parameters
        self.p_entailment = 0
        self.direction_bias = 0

    def end_participant(self, subj_id, **kwargs):
        print(subj_id, self.p_entailment)

    def predict(self, item, **kwargs):
        task_enc = ccobra.syllogistic.encode_task(item.task)

        # Obtain predictions
        preds = self.phm.predict(task_enc)
        min_concls = preds[:(len(preds) // 2)]
        pent_concls = preds[(len(preds) // 2):]

        preds = min_concls
        if self.p_entailment > 0:
            preds = pent_concls

        pred = preds[0]
        if self.direction_bias < 0 and len(preds) == 2:
            pred = preds[1]

        return ccobra.syllogistic.decode_response(pred, item.task)

    def adapt(self, item, truth, **kwargs):
        task_enc = ccobra.syllogistic.encode_task(item.task)
        truth_enc = ccobra.syllogistic.encode_response(truth, item.task)

        # Obtain predictions
        preds = self.phm.predict(task_enc)
        min_concls = preds[:(len(preds) // 2)]
        pent_concls = preds[(len(preds) // 2):]

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
