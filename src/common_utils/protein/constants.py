import enum

##
# Residue identities

non_standard_residue_substitutions = {
    '2AS':'ASP', '3AH':'HIS', '5HP':'GLU', 'ACL':'ARG', 'AGM':'ARG', 'AIB':'ALA', 'ALM':'ALA', 'ALO':'THR', 'ALY':'LYS', 'ARM':'ARG',
    'ASA':'ASP', 'ASB':'ASP', 'ASK':'ASP', 'ASL':'ASP', 'ASQ':'ASP', 'AYA':'ALA', 'BCS':'CYS', 'BHD':'ASP', 'BMT':'THR', 'BNN':'ALA',
    'BUC':'CYS', 'BUG':'LEU', 'C5C':'CYS', 'C6C':'CYS', 'CAS':'CYS', 'CCS':'CYS', 'CEA':'CYS', 'CGU':'GLU', 'CHG':'ALA', 'CLE':'LEU', 'CME':'CYS',
    'CSD':'ALA', 'CSO':'CYS', 'CSP':'CYS', 'CSS':'CYS', 'CSW':'CYS', 'CSX':'CYS', 'CXM':'MET', 'CY1':'CYS', 'CY3':'CYS', 'CYG':'CYS',
    'CYM':'CYS', 'CYQ':'CYS', 'DAH':'PHE', 'DAL':'ALA', 'DAR':'ARG', 'DAS':'ASP', 'DCY':'CYS', 'DGL':'GLU', 'DGN':'GLN', 'DHA':'ALA',
    'DHI':'HIS', 'DIL':'ILE', 'DIV':'VAL', 'DLE':'LEU', 'DLY':'LYS', 'DNP':'ALA', 'DPN':'PHE', 'DPR':'PRO', 'DSN':'SER', 'DSP':'ASP',
    'DTH':'THR', 'DTR':'TRP', 'DTY':'TYR', 'DVA':'VAL', 'EFC':'CYS', 'FLA':'ALA', 'FME':'MET', 'GGL':'GLU', 'GL3':'GLY', 'GLZ':'GLY',
    'GMA':'GLU', 'GSC':'GLY', 'HAC':'ALA', 'HAR':'ARG', 'HIC':'HIS', 'HIP':'HIS', 'HMR':'ARG', 'HPQ':'PHE', 'HTR':'TRP', 'HYP':'PRO',
    'IAS':'ASP', 'IIL':'ILE', 'IYR':'TYR', 'KCX':'LYS', 'LLP':'LYS', 'LLY':'LYS', 'LTR':'TRP', 'LYM':'LYS', 'LYZ':'LYS', 'MAA':'ALA', 'MEN':'ASN',
    'MHS':'HIS', 'MIS':'SER', 'MLE':'LEU', 'MPQ':'GLY', 'MSA':'GLY', 'MSE':'MET', 'MVA':'VAL', 'NEM':'HIS', 'NEP':'HIS', 'NLE':'LEU',
    'NLN':'LEU', 'NLP':'LEU', 'NMC':'GLY', 'OAS':'SER', 'OCS':'CYS', 'OMT':'MET', 'PAQ':'TYR', 'PCA':'GLU', 'PEC':'CYS', 'PHI':'PHE',
    'PHL':'PHE', 'PR3':'CYS', 'PRR':'ALA', 'PTR':'TYR', 'PYX':'CYS', 'SAC':'SER', 'SAR':'GLY', 'SCH':'CYS', 'SCS':'CYS', 'SCY':'CYS',
    'SEL':'SER', 'SEP':'SER', 'SET':'SER', 'SHC':'CYS', 'SHR':'LYS', 'SMC':'CYS', 'SOC':'CYS', 'STY':'TYR', 'SVA':'SER', 'TIH':'ALA',
    'TPL':'TRP', 'TPO':'THR', 'TPQ':'ALA', 'TRG':'LYS', 'TRO':'TRP', 'TYB':'TYR', 'TYI':'TYR', 'TYQ':'TYR', 'TYS':'TYR', 'TYY':'TYR'
}


ressymb_to_resindex = {
    'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4,
    'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9,
    'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14,
    'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19,
    'X': 20,
}


class AA(enum.IntEnum):
    ALA = 0; CYS = 1; ASP = 2; GLU = 3; PHE = 4
    GLY = 5; HIS = 6; ILE = 7; LYS = 8; LEU = 9
    MET = 10; ASN = 11; PRO = 12; GLN = 13; ARG = 14
    SER = 15; THR = 16; VAL = 17; TRP = 18; TYR = 19
    UNK = 20

    @classmethod
    def _missing_(cls, value):
        if isinstance(value, str) and len(value) == 3:      # three representation
            if value in non_standard_residue_substitutions:
                value = non_standard_residue_substitutions[value]
            if value in cls._member_names_:
                return getattr(cls, value)
        elif isinstance(value, str) and len(value) == 1:    # one representation
            if value in ressymb_to_resindex:
                return cls(ressymb_to_resindex[value])

        return super()._missing_(value)

    def __str__(self):
        return self.name

    @classmethod
    def is_aa(cls, value):
        return (value in ressymb_to_resindex) or \
            (value in non_standard_residue_substitutions) or \
            (value in cls._member_names_)


num_aa_types = len(AA)

##
# Atom identities

class BBHeavyAtom(enum.IntEnum):
    N = 0; CA = 1; C = 2; O = 3; CB = 4; OXT=14;

max_num_heavyatoms = 15
max_num_hydrogens = 16
max_num_allatoms = max_num_heavyatoms + max_num_hydrogens

restype_to_heavyatom_names = {
    AA.ALA: ['N', 'CA', 'C', 'O', 'CB', '',    '',    '',    '',    '',    '',    '',    '',    '', 'OXT'],
    AA.ARG: ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD',  'NE',  'CZ',  'NH1', 'NH2', '',    '',    '', 'OXT'],
    AA.ASN: ['N', 'CA', 'C', 'O', 'CB', 'CG',  'OD1', 'ND2', '',    '',    '',    '',    '',    '', 'OXT'],
    AA.ASP: ['N', 'CA', 'C', 'O', 'CB', 'CG',  'OD1', 'OD2', '',    '',    '',    '',    '',    '', 'OXT'],
    AA.CYS: ['N', 'CA', 'C', 'O', 'CB', 'SG',  '',    '',    '',    '',    '',    '',    '',    '', 'OXT'],
    AA.GLN: ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD',  'OE1', 'NE2', '',    '',    '',    '',    '', 'OXT'],
    AA.GLU: ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD',  'OE1', 'OE2', '',    '',    '',    '',    '', 'OXT'],
    AA.GLY: ['N', 'CA', 'C', 'O', '',   '',    '',    '',    '',    '',    '',    '',    '',    '', 'OXT'],
    AA.HIS: ['N', 'CA', 'C', 'O', 'CB', 'CG',  'ND1', 'CD2', 'CE1', 'NE2', '',    '',    '',    '', 'OXT'],
    AA.ILE: ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2', 'CD1', '',    '',    '',    '',    '',    '', 'OXT'],
    AA.LEU: ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD1', 'CD2', '',    '',    '',    '',    '',    '', 'OXT'],
    AA.LYS: ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD',  'CE',  'NZ',  '',    '',    '',    '',    '', 'OXT'],
    AA.MET: ['N', 'CA', 'C', 'O', 'CB', 'CG',  'SD',  'CE',  '',    '',    '',    '',    '',    '', 'OXT'],
    AA.PHE: ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD1', 'CD2', 'CE1', 'CE2', 'CZ',  '',    '',    '', 'OXT'],
    AA.PRO: ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD',  '',    '',    '',    '',    '',    '',    '', 'OXT'],
    AA.SER: ['N', 'CA', 'C', 'O', 'CB', 'OG',  '',    '',    '',    '',    '',    '',    '',    '', 'OXT'],
    AA.THR: ['N', 'CA', 'C', 'O', 'CB', 'OG1', 'CG2', '',    '',    '',    '',    '',    '',    '', 'OXT'],
    AA.TRP: ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD1', 'CD2', 'NE1', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2', 'OXT'],
    AA.TYR: ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD1', 'CD2', 'CE1', 'CE2', 'CZ',  'OH',  '',    '', 'OXT'],
    AA.VAL: ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2', '',    '',    '',    '',    '',    '',    '', 'OXT'],
    AA.UNK: ['',  '',   '',  '',  '',   '',    '',    '',    '',    '',    '',    '',    '',    '',    ''],
}
for names in restype_to_heavyatom_names.values(): assert len(names) == max_num_heavyatoms


##
# Torsion geometry and ideal coordinates

class Torsion(enum.IntEnum):
    Backbone = 0
    Omega = 1
    Phi = 2
    Psi = 3
    Chi1 = 4
    Chi2 = 5
    Chi3 = 6
    Chi7 = 7


chi_angles_atoms = {
    AA.ALA: [],
    # Chi5 in arginine is always 0 +- 5 degrees, so ignore it.
    AA.ARG: [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD'],
            ['CB', 'CG', 'CD', 'NE'], ['CG', 'CD', 'NE', 'CZ']],
    AA.ASN: [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'OD1']],
    AA.ASP: [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'OD1']],
    AA.CYS: [['N', 'CA', 'CB', 'SG']],
    AA.GLN: [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD'],
            ['CB', 'CG', 'CD', 'OE1']],
    AA.GLU: [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD'],
            ['CB', 'CG', 'CD', 'OE1']],
    AA.GLY: [],
    AA.HIS: [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'ND1']],
    AA.ILE: [['N', 'CA', 'CB', 'CG1'], ['CA', 'CB', 'CG1', 'CD1']],
    AA.LEU: [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD1']],
    AA.LYS: [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD'],
            ['CB', 'CG', 'CD', 'CE'], ['CG', 'CD', 'CE', 'NZ']],
    AA.MET: [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'SD'],
            ['CB', 'CG', 'SD', 'CE']],
    AA.PHE: [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD1']],
    AA.PRO: [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD']],
    AA.SER: [['N', 'CA', 'CB', 'OG']],
    AA.THR: [['N', 'CA', 'CB', 'OG1']],
    AA.TRP: [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD1']],
    AA.TYR: [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD1']],
    AA.VAL: [['N', 'CA', 'CB', 'CG1']],
}

num_chi_angles = {
    k: len(v) for k, v in chi_angles_atoms.items()
}


chi_angles_mask = {
    AA.ALA: [False, False, False, False],  # ALA
    AA.ARG: [True , True , True , True ],  # ARG
    AA.ASN: [True , True , False, False],  # ASN
    AA.ASP: [True , True , False, False],  # ASP
    AA.CYS: [True , False, False, False],  # CYS
    AA.GLN: [True , True , True , False],  # GLN
    AA.GLU: [True , True , True , False],  # GLU
    AA.GLY: [False, False, False, False],  # GLY
    AA.HIS: [True , True , False, False],  # HIS
    AA.ILE: [True , True , False, False],  # ILE
    AA.LEU: [True , True , False, False],  # LEU
    AA.LYS: [True , True , True , True ],  # LYS
    AA.MET: [True , True , True , False],  # MET
    AA.PHE: [True , True , False, False],  # PHE
    AA.PRO: [True , True , False, False],  # PRO
    AA.SER: [True , False, False, False],  # SER
    AA.THR: [True , False, False, False],  # THR
    AA.TRP: [True , True , False, False],  # TRP
    AA.TYR: [True , True , False, False],  # TYR
    AA.VAL: [True , False, False, False],  # VAL
    AA.UNK: [False, False, False, False],  # UNK
}


chi_pi_periodic = {
    AA.ALA: [False, False, False, False],  # ALA
    AA.ARG: [False, False, False, False],  # ARG
    AA.ASN: [False, False, False, False],  # ASN
    AA.ASP: [False, True , False, False],  # ASP
    AA.CYS: [False, False, False, False],  # CYS
    AA.GLN: [False, False, False, False],  # GLN
    AA.GLU: [False, False, True , False],  # GLU
    AA.GLY: [False, False, False, False],  # GLY
    AA.HIS: [False, False, False, False],  # HIS
    AA.ILE: [False, False, False, False],  # ILE
    AA.LEU: [False, False, False, False],  # LEU
    AA.LYS: [False, False, False, False],  # LYS
    AA.MET: [False, False, False, False],  # MET
    AA.PHE: [False, True , False, False],  # PHE
    AA.PRO: [False, False, False, False],  # PRO
    AA.SER: [False, False, False, False],  # SER
    AA.THR: [False, False, False, False],  # THR
    AA.TRP: [False, False, False, False],  # TRP
    AA.TYR: [False, True , False, False],  # TYR
    AA.VAL: [False, False, False, False],  # VAL
    AA.UNK: [False, False, False, False],  # UNK
}
