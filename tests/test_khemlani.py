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
        sys.path.append(test_dir + os.sep + '..' + os.sep + 'phm')
        import phm
        self.phm = phm.PHM()

    def test_syllogisms(self):
        for _, row in self.khemlani_df.iterrows():
            task = row['Syllogism']
            truth = row['Prediction'].split(';')
            preds = self.phm.predict(task)
            self.assertEqual(preds, truth, msg='Syllogism: {}'.format(task))

if __name__ == '__main__':
    unittest.main()
