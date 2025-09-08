# Target column in the data
target_col = "Rang_rescaled"


decision_tree_grid = {
    "max_depth": [5, 10, 13, 15],
    "min_samples_split": [150],
    "min_samples_leaf": [100],
    "max_features": ["sqrt", None],
}

random_forest_grid = {
    "n_estimators": [50, 200],
    "max_depth": [10, 13],
    "min_samples_split": [150],
    "min_samples_leaf": [100],
    "max_features": [None],
}

xgboosst_grid = {
    "n_estimators": [300, 1000],
    "max_depth": [10],
    "eta": [0.1, 0.3],  # learning rate
    "colsample_bytree": [0.8],
}
# decision_tree_grid, random_forest_grid, xgboosst_grid = {}

# Remove deduced variables + redundant variables + donor variables (already included in matching functions)
variables_deduced = ["ICAR", "SCORE_CCB", "SCORE_CCP", "SCORE_C"]

variables_redundant_sup0 = [
    "AppABO",
    "AppSC",
    "IND_POSTGRF",
    "AppAge",
    "SCR",
    "URGENCE",
    "KXPC",
    "XPC",
    "DelaiURG",
    "DelaiDRG2",
    "DelaiCAT2",
    "BNP2",
    "DelaiBNP2",
    "PROBNP2",
    "DelaiPROBNP2",
    "DelaiBILI2",
    "BNP_AVI",
    "PBN_AVI",
    "MG",
    "CAT2",
    "nb_pat_for_idd",
]
variables_donneur = ["AGED", "ABOD", "SEXD", "TAILLED", "POIDSD", "SCD"]

map_col_name = {
    "IDD": "ID du donneur",
    "IDR": "ID du receveur",
    "TimelineD": "Temps relatif du donneur",
    "AGED": "Age du donneur",
    "ABOD": "Groupe sanguin donneur",
    "SEXD": "Sexe donneur",
    "TAILLED": "Taille donneur",
    "POIDSD": "Poids donneur",
    "IndALD": "Index points bonus assistance gauche (+750 pts)",
    "AGER": "Age receveur",
    "ABOR": "Groupe sanguin receveur",
    "SEXER": "Sexe receveur",
    "TAILLER": "Taille receveur",
    "POIDSR": "Poids receveur",
    "SCR": "Surface corporelle receveur",
    "DelaiINSC": "Délai d'inscription",
    "DelaiURG": "Délai d'urgence (en j)",
    "XPC": "Délai max points urgence (0/1/2/3)",
    "MAL": "Maladie receveur",
    "MALADI": "Maladie 1 receveur",
    "MALADI2": "Maladie 2 receveur",
    "DRG2": "Drogues inotropes receveur",
    "DelaiDRG2": "Délai drogues inotropes (en j)",
    "CEC2": "Circulation Extra-Corporelle (oui/non)",
    "DelaiCEC2": "Délai CEC2 receveur (en j)",
    "CEC2_12": "CEC inférieure à 12 j (oui/non)",
    "SIAV2": "Type d'assistance ventriculaire",
    "DelaiAV2": "Délai ass. ventriculaire",
    "CAT2": "Cœur Artificiel Total",
    "CAT_BV": "Cœur Artificiel Total (ou BV)",
    "DelaiCAT2": "Délai CAT2 receveur (en j)",
    "BNP2": "BNP receveur (peptides)",
    "DelaiBNP2": "Délai BNP",
    "PROBNP2": "NT PROBNP receveur (peptides)",
    "DelaiPROBNP2": "Délai PROBNP",
    "DecilePEPT": "Décile Peptides",
    "DecilePEPT_CEC": "Décile Peptides + CEC",
    "DecilePEPT_IMP": "Décile Peptides",
    "DIA2": "Dialyse receveur (oui/non)",
    "BILI2": "Bilirubine receveur (µmol/L)",
    "DelaiBILI2": "Délai Bilirubine (en j)",
    "DIA_AVI": "Dialyse avant CEC2 ou DRG2 (oui/non)",
    "CRE_AVI": "Créatinine avant CEC2 ou DRG2",
    "BILI_AVI": "Bilirubine avant CEC2 ou DRG2",
    "BNP_AVI": "BNP (peptides) avant CEC2 ou DRG2",
    "PBN_AVI": "NT PROBNP (peptides) avant CEC2 ou DRG2",
    "DecilePEPT_AVI": "Décile Peptides avant CEC2 ou DRG2 (1 à 10)",
    "ICAR": "Index ICAR",
    "ALLOC": "Composante score (adulte/pédiatrie +/- urgence)",
    "AppAge": "Appariement âge (0-1)",
    "AppABO": "Appariement groupe sanguin (0-1)",
    "AppSC": "Appariement surface corporelle (0/1)",
    "IND_POSTGRF": "Survie post-greffe (0/1)",
    "DIST": "Distance (en minutes)",
    "MG": "Modèle géographique (0-1)",
    "Rang_rescaled": "Rang_rescaled",
    "nb_pat_for_idd": "nb patients classés sur ce greffon",
    "DFG": "Débit de filtration glomérulaire",
    "DFG_AVI": "Débit de filtration glomérulaire avant CEC2 ou DRG2",
    "observation_weight": "Pondération de l'observation",
    "nb_nonzeros_propositions_for_IDR": "nb_nonzeros_propositions_for_IDR",
}
