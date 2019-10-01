import unittest
import os
import sys

import pandas as pd

class KhemlaniTest(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        test_dir = os.path.split(os.path.abspath(__file__))[0]
        asset_dir = test_dir + os.sep + 'assets'
        self.khemlani_df = pd.read_csv(asset_dir + os.sep + 'Khemlani2012-PHM.csv')

        # Load PHM
        sys.path.append(test_dir + os.sep + '..')
        import phm
        self.phm = phm.PHM(khemlani_phrase=True)

    def test_khemlani(self):
        for _, row in self.khemlani_df.iterrows():
            task = row['Syllogism']
            truth = row['Prediction'].split(';')

            min_preds = self.phm.generate_conclusions(task, use_p_entailment=False)
            pent_preds = self.phm.generate_conclusions(task, use_p_entailment=True)
            preds = min_preds + pent_preds
            self.assertEqual(preds, truth, msg='Syllogism: {}'.format(task))

    def test_oaksford2001(self):
        task = 'IA1'
        truth = ['Iac', 'Oac']
        min_preds = self.phm.generate_conclusions(task, use_p_entailment=False)
        pent_preds = self.phm.generate_conclusions(task, use_p_entailment=True)
        prediction = min_preds + pent_preds
        self.assertEqual(prediction, truth, msg='Syllogism: {}'.format(task))

if __name__ == '__main__':
    unittest.main()
