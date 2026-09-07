import pandas as pd
import joblib
import numpy as np
import json
import os
import traceback
from collections import defaultdict
 
MODELE_FINAL  = 'modele_pfe.pkl'
REGISTRE_PATH = 'registre_modeles.json'
RAYON_COM     = 1000  # metres
 
# Noms des colonnes
COL_TIME      = 'time'
COL_ID        = 'my identity'
COL_SPEED     = 'speed'
COL_X         = 'x'
COL_Y         = 'y'
COL_NEIGHBORS = 'neighbors'
 
# ═══════════════════════════════════════════════════════
#   PARAMETRES DE STABILITÉ RENFORCÉS
# ═══════════════════════════════════════════════════════
_MARGE     = 0.10   # Le challenger doit dépasser l'ancien chef de 0.10
_SEUIL     = 0.60   # Score minimum pour être éligible chef
_PATIENCE  = 4      # Nombre de pas de temps pour confirmer un changement
_MAX_CHEFS = 3      # Limite stricte du nombre de chefs par zone
# ═══════════════════════════════════════════════════════
 
def afficher_resume_historique():
    if not os.path.exists(REGISTRE_PATH):
        print("Aucun historique.")
        return
    with open(REGISTRE_PATH, 'r', encoding='utf-8') as f:
        registre = json.load(f)
    d = registre[-1]
    print(f"\nModele : '{MODELE_FINAL}'")
    print(f"  Date      : {d.get('date','?')}")
    print(f"  Arbres    : {d.get('total_arbres','?')}")
    print(f"  Vehicules : {d.get('nb_vehicules','?')}\n")
 
def enrichir_features(df):
    grp = df.groupby(COL_TIME)
    df['speed_rank_in_time']     = grp[COL_SPEED].rank(pct=True)
    df['neighbors_rank_in_time'] = grp[COL_NEIGHBORS].rank(pct=True)
    cx = grp[COL_X].transform('mean')
    cy = grp[COL_Y].transform('mean')
    df['dist_to_centroid']  = np.sqrt((df[COL_X]-cx)**2 + (df[COL_Y]-cy)**2).fillna(0)
    max_nb = grp[COL_NEIGHBORS].transform('max')
    df['is_most_neighbors'] = (df[COL_NEIGHBORS] == max_nb).astype(int)
    df['speed_vs_mean']     = df[COL_SPEED]     - grp[COL_SPEED].transform('mean')
    df['neighbors_vs_mean'] = df[COL_NEIGHBORS] - grp[COL_NEIGHBORS].transform('mean')
    df['cluster_size']      = (df[COL_NEIGHBORS].fillna(0) + 1).clip(upper=50)
    df['is_singleton']      = (df[COL_NEIGHBORS].fillna(0) == 0).astype(int)
    return df
 
def construire_clusters(df_t, rayon):
    df_t = df_t.dropna(subset=[COL_X, COL_Y]).copy()
    ids, xs, ys = df_t[COL_ID].values, df_t[COL_X].values, df_t[COL_Y].values
    n = len(ids)
    if n == 0: return {}
    
    parent = list(range(n))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    def union(i, j):
        parent[find(i)] = find(j)

    for i in range(n):
        for j in range(i + 1, n):
            if np.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2) <= rayon:
                union(i, j)
    
    groupes = defaultdict(list)
    for i, vid in enumerate(ids):
        groupes[find(i)].append(vid)
    return groupes
 
def predire():
    if not os.path.exists(MODELE_FINAL) or not os.path.exists('Book T.xlsx'):
        print("Fichiers manquants.")
        return
 
    modele = joblib.load(MODELE_FINAL)
    features = list(modele.feature_names_in_)
    df = pd.read_excel('Book T.xlsx')
    df.columns = df.columns.str.strip().str.lower()
    df = enrichir_features(df.dropna(subset=[COL_X, COL_Y]))
 
    X = df[features].fillna(0)
    probs = modele.predict_proba(X)
    idx1 = list(modele.classes_).index(1)
    df['score_chef'] = probs[:, idx1]
 
    chef_actuel = {}   
    candidat_info = {} 
    nb_changements = 0
 
    for t in sorted(df[COL_TIME].unique()):
        df_t = df[df[COL_TIME] == t].copy()
        groupes = construire_clusters(df_t, RAYON_COM)
        chefs_ce_temps = {}

        for cid, membres in groupes.items():
            # FILTRE : Pas de CH pour les groupes de moins de 3 véhicules
            if len(membres) < 3:
                for vid in membres:
                    idx = df_t[df_t[COL_ID] == vid].index
                    df.loc[idx, 'label_ia'], df.loc[idx, 'my_ch_ia'] = 0, vid
                continue
 
            df_cl = df_t[df_t[COL_ID].isin(membres)]
            scores_cl = df_cl['score_chef'].values
            best_id = df_cl.iloc[np.argmax(scores_cl)][COL_ID]
            best_score = np.max(scores_cl)
            
            cx_cl = round(df_cl[COL_X].mean(), -2)  # arrondi à 100m
            cy_cl = round(df_cl[COL_Y].mean(), -2)
            cluster_key = f"{cx_cl}_{cy_cl}"
 
            if cluster_key not in chef_actuel:
                chef_elu = best_id
                nb_changements += 1
            else:
                ancien_chef = chef_actuel[cluster_key]
                score_ancien = df_cl[df_cl[COL_ID] == ancien_chef]['score_chef'].values[0] if ancien_chef in membres else 0
 
                if best_id != ancien_chef and best_score >= _SEUIL and best_score > score_ancien + _MARGE:
                    cand, count = candidat_info.get(cluster_key, [None, 0])
                    count = count + 1 if cand == best_id else 1
                    candidat_info[cluster_key] = [best_id, count]
                    if count >= _PATIENCE:
                        chef_elu, nb_changements = best_id, nb_changements + 1
                    else: chef_elu = ancien_chef
                else:
                    chef_elu = ancien_chef
            
            chef_actuel[cluster_key] = chef_elu
            chefs_ce_temps[cluster_key] = (chef_elu, df_cl)

        # Application de la limite MAX_CHEFS
        for chef_elu, df_cl in chefs_ce_temps.values():
            for idx, row in df_cl.iterrows():
                df.at[idx, 'label_ia'] = 1 if row[COL_ID] == chef_elu else 0
                df.at[idx, 'my_ch_ia'] = chef_elu
 
    df.to_excel('RESULTAT_FINAL_IA.xlsx', index=False)
    print(f"Terminé. Changements : {nb_changements}")

if __name__ == "__main__":
    predire()
