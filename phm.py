import warnings

def max_premise(task):
    """

    Parameters
    ----------
    task : str
        Task encoding

    """

    ordering = ['O', 'E', 'I', 'A']
    if ordering.index(task[0]) < ordering.index(task[1]):
        return task[1]
    else:
        return task[0]

class PHM():
    def __init__(self, khemlani_phrase=False):
        """ Initializes PHM by setting up the informativeness order and p_entailment definitions.

        Parameters
        ----------
        khemlani_phrase : bool
            Flag indicating whether Khemlani & Johnson-Laird's (2012) variant of PHM is to be used.
            "Some not" is not changed to "Some" phrases in this case which is in accordance to
            Chater & Oaksford (1999).
        """

        # Flag indicating how noun phrases are constructed. In Oaksford & Chater (2001), "Some not"
        # produces the same noun phrase as "Some". In order to reproduce Khemlani & Johnson-Laird's
        # (2012) prediction table, they have to be considered distinct (which is in accordance to
        # Chater & Oaksford, 1999).
        self.khemlani_phrase = khemlani_phrase

        self.informativeness_order = ['A', 'I', 'E', 'O']
        self.p_entailments = {'A': 'I', 'E': 'O', 'O': 'I', 'I': 'O'}

    @staticmethod
    def get_premises(task_enc):
        """ Extracts premises from a given syllogistic task encoding.

        Parameters
        ----------
        task_enc : str
            Syllogistic task encoding (e.g., 'AA1', 'OE4', etc.). Note that the encodings follow
            the specification of Khemlani & Johnson-Laird (2012) which use different numbers for
            the syllogistic figures.

        Returns
        -------
        prem1, prem2 : list(str)
            Tuple representations for both premises. Each contains three elements: quantifier and
            two terms.

        """

        figure = int(task_enc[2])

        prem1 = None
        prem2 = None
        if figure == 1:
            prem1 = [task_enc[0], 'A', 'B']
            prem2 = [task_enc[1], 'B', 'C']
        elif figure == 2:
            prem1 = [task_enc[0], 'B', 'A']
            prem2 = [task_enc[1], 'C', 'B']
        elif figure == 3:
            prem1 = [task_enc[0], 'A', 'B']
            prem2 = [task_enc[1], 'C', 'B']
        elif figure == 4:
            prem1 = [task_enc[0], 'B', 'A']
            prem2 = [task_enc[1], 'B', 'C']

        return prem1, prem2

    def noun_phrase(self, premise):
        """ Creates the noun phrase for a given premise.

        Parameters
        ----------
        premise : list(str)
            Tuple representation of a syllogistic premise. First element denotes quantifier,
            remaining two denote terms.

        Returns
        -------
        list(str)
            Two-element list containing the quantifier and subject term.

        """

        if self.khemlani_phrase:
            return [premise[0], premise[1]]
        else:
            return [premise[0].replace('O', 'I'), premise[1]]

    def gen_conclusions(self, quant, subject):
        """ Generate the list of conclusions for a given quantifier and subject.

        Parameters
        ----------
        quant : str
            Syllogistic quantifier (i.e., 'A', 'I', 'E', 'O')

        subject : str
            Syllogistic subject (i.e., 'A', 'C') or None if not specified. In this case,
            conclusions for both directions are returned.

        Returns
        -------
        list(str)
            List of conclusions.

        """

        if not subject:
            return [quant + 'ac', quant + 'ca']
        return [quant + subject.replace('A', 'ac').replace('C', 'ca')]

    def generate_conclusions(self, task, use_p_entailment):
        """ Computes the list of predictions for a given syllogistic task.

        Premises
        --------
        task : str
            Syllogism to predict for (e.g., "AA1", "AA2", ...). Note that the encodings follow
            the specification of Khemlani & Johnson-Laird (2012) which use different numbers for
            the syllogistic figures.

        use_p_entailment : bool

        Returns
        -------
        list(str)
            List of predictions.

        """

        # Complete the premises
        prem1, prem2 = self.get_premises(task)

        # Determine min and max premise
        min_prem = prem1
        max_prem = prem2
        if self.informativeness_order.index(prem1[0]) < self.informativeness_order.index(prem2[0]):
            min_prem = prem2
            max_prem = prem1

        # Obtain p-entailment quantifier
        p_ent_quant = self.p_entailments[min_prem[0]]

        # Compute direction via attachment
        min_concl_cands = [[min_prem[0], 'A', 'C'], [min_prem[0], 'C', 'A']]
        min_concl_cand_phrases = [self.noun_phrase(x) for x in min_concl_cands]

        prem1_phrase = self.noun_phrase(prem1)
        prem2_phrase = self.noun_phrase(prem2)
        prem_phrases = [prem1_phrase, prem2_phrase]

        subject = None
        if prem1[0] == prem2[0]:
            # If both premise quantifiers are equal, the direction cannot be inferred since
            # min- and max-premises are unspecified
            subject = None
        elif min_concl_cand_phrases[0] == prem1_phrase and min_concl_cand_phrases[1] not in prem_phrases:
            subject = min_concl_cand_phrases[0][1]
        elif min_concl_cand_phrases[0] == prem2_phrase and min_concl_cand_phrases[1] not in prem_phrases:
            subject = min_concl_cand_phrases[0][1]
        elif min_concl_cand_phrases[1] == prem1_phrase and min_concl_cand_phrases[0] not in prem_phrases:
            subject = min_concl_cand_phrases[1][1]
        elif min_concl_cand_phrases[1] == prem2_phrase and min_concl_cand_phrases[0] not in prem_phrases:
            subject = min_concl_cand_phrases[1][1]
        else:
            # Both or none match, use max premise subject as conclusion subject
            subject = ''.join(max_prem[1:]).replace('B', '')

        # Generate the conclusions
        if use_p_entailment:
            return self.gen_conclusions(p_ent_quant, subject)
        else:
            return self.gen_conclusions(min_prem[0], subject)

    def max_heuristic(self, task, a_conf, i_conf, e_conf, o_conf):
        """

        """

        # Validate that A>I>E~O
        if not (a_conf >= i_conf >= e_conf) or not (a_conf >= i_conf >= o_conf):
            warnings.warn(
                'max-quantifier confidences in incorrect ranking (should be A > I > E ~ O).')

        # Apply max-heuristic
        confidences = {
            'A': a_conf,
            'I': i_conf,
            'E': e_conf,
            'O': o_conf
        }

        max_prem = max_premise(task)
        max_conf = confidences[max_prem[0]]
        if max_conf >= 0.5:
            return True
        return False
