class PHM():
    def __init__(self):
        self.informativeness_order = ['A', 'I', 'E', 'O']
        self.p_entailments = {
            'A': 'I',
            'E': 'O',
            'O': 'I',
            'I': 'O'
        }

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
        if quant in self.p_entailments:
            return self.p_entailments[quant]
        return None

    def p_entailment(self, min_quant):
        return self.p_entails(min_quant)

    def attachment(self, min_quant, task_enc):
        # Treat Some not as Some (Oaksford, 2001)
        min_quant = min_quant#.replace('O', 'I')
        quant1, quant2 = task_enc[:-1]#.replace('O', 'I')
        figure = int(task_enc[-1])

        # Direction cannot be inferred for double-syllogisms
        if quant1 == quant2:
            return ['ac', 'ca']

        if min_quant == quant1:
            if figure % 2 == 1:
                return ['ac']
            return ['ca']

        if min_quant == quant2:
            if int(figure / 2) - 1 == 0:
                return ['ca']
            return ['ac']

        assert False, 'Shitness happened'

    def predict(self, task):
        quants = [task[0], task[1]]

        min_quant = self.min_heuristic(quants)
        p_ent = self.p_entailment(min_quant)

        # Unsure how to proceed.
        # Assumption 1: Attachment phrase only considers end-term (kill middle term)
        # Assumption 2: Attachment only on min-conclusion

        attachment_dir = self.attachment(min_quant, task)

        min_concls = [min_quant + direction for direction in attachment_dir]
        p_ent_concls = [p_ent + direction for direction in attachment_dir]
        return min_concls + p_ent_concls
