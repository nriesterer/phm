class PHM():
    def __init__(self):
        self.informativeness_order = [
            'A',
            'I',
            'E',
            'O'
        ]

    def more_informative_than(self, quant1, quant2):
        idx1 = self.informativeness_order.index(quant1)
        idx2 = self.informativeness_order.index(quant2)

        if idx1 <= idx2:
            return True
        return False

    def min_heuristic(self, premise_quants):
        quant1, quant2 = premise_quants

        if self.more_informative_than(quant1, quant2):
            return quant2
        return quant1

    def p_entails(self, quant):
        p_entailments = {
            'A': 'I',
            'E': 'O',
            'O': 'I',
            'I': 'O'
        }

        if quant in p_entailments:
            return p_entailments[quant]
        return None

    def p_entailment(self, min_quant):
        return self.p_entails(min_quant)

    def attachment(self, min_quant, prem_quants):
        prem1, prem2 = prem_quants
        prem1 = prem1.replace('O', 'I')
        prem2 = prem2.replace('O', 'I')
        min_quant = min_quant.replace('O', 'I')

        if prem1 == prem2:
            return None

        if min_quant == prem1:
            return 'ac'
        elif min_quant == prem2:
            return 'ca'

        assert False, 'Attachment failure with min_quant={}, prem_quants={}'.format(min_quant, prem_quants)
        return None

    def predict(self, task):
        quants = [task[0], task[1]]

        min_quant = self.min_heuristic(quants)
        p_ent = self.p_entailment(min_quant)

        # Unsure how to proceed.
        # Assumption 1: Attachment phrase only considers end-term (kill middle term)
        # Assumption 2: Attachment only on min-conclusion

        attachment_dir = self.attachment(min_quant, quants)
        if attachment_dir:
            return [min_quant + attachment_dir, p_ent + attachment_dir]

        return [min_quant + 'ac', min_quant + 'ca', p_ent + 'ac', p_ent + 'ca']

import ccobra
p = PHM()
for syllog in ccobra.syllogistic.SYLLOGISMS:
    print(syllog, p.predict(syllog))
