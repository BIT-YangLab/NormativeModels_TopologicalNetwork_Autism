"""
=======================================================================
 Script Name: Subtype_RCCA_Analysis.py

 Purpose:
   This script performs regularized canonical correlation analysis (RCCA) 
   between structural brain network deviations and behavioral/clinical 
   scales across ASD subtypes (clusters). The analysis tests significance 
   via permutation tests, identifies consistent associations across all 
   clusters, and extracts top contributing features.

 Inputs:
   - subtype_sf_table_100_new_all.csv
       Structural deviation values across 20 brain networks 
       (with assigned cluster labels and behavioral measures).

   Columns of interest:
       • Brain networks (X):
         ['Default_Parietal', 'Default_Anterolateral', 'Default_Dorsolateral',
          'Default_Retrosplenial', 'Visual_Lateral', 'Visual_Dorsal_VentralStream',
          'Visual_V5', 'Visual_V1', 'Frontoparietal', 'DorsalAttention',
          'Premotor_DorsalAttentionII', 'Language', 'Salience',
          'CinguloOpercular_Action_mode', 'MedialParietal', 'Somatomotor_Hand',
          'Somatomotor_Face', 'Somatomotor_Foot', 'Auditory', 'SomatoCognitiveAction']

       • Candidate behavioral scales (Y):
         ['ADI_RRB_TOTAL_C','ADOS_STEREO_BEHAV','ADOS_GOTHAM_RRB',
          'SRS_MANNERISMS','VINELAND_PERSONAL_V_SCALED',
          'VINELAND_DOMESTIC_V_SCALED','VINELAND_PLAY_V_SCALED',
          'VINELAND_COPING_V_SCALED','ADOS_TOTAL','ADOS_COMM',
          'ADOS_SOCIAL','ADOS_GOTHAM_SOCAFFECT','ADI_R_SOCIAL_TOTAL_A',
          'ADI_R_VERBAL_TOTAL_BV','SRS_RAW_TOTAL','SRS_AWARENESS',
          'SRS_COGNITION','SRS_COMMUNICATION','SRS_MOTIVATION']

 Outputs:
   - ./subtype/cca_components_cluster*_scale*.csv  
       Canonical variates (X_c, Y_c) for significant associations.  
   - ./subtype/rcca_significant_results_.csv  
       Summary of significant RCCA results (correlation, p-values).  
   - ./subtype/rcca_component_weights_.csv  
       Weights of brain networks contributing to canonical components.  
   - ./subtype/rtop5_features_per_row_.csv  
       Top 5 contributing features (per row/scale) based on weight magnitudes.  


 Dependencies:
   - pandas, numpy, scipy, scikit-learn, rcca
=======================================================================
"""

import warnings

import numpy as np
import pandas as pd
from rcca import CCA
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ========= Step 1: Load and preprocess data =========
df = pd.read_csv("./data/subtype_sf_table_100_new_all.csv")
df.columns = [col.replace('/', '_') for col in df.columns]

columns_to_fit = [
    'Default_Parietal', 'Default_Anterolateral', 'Default_Dorsolateral', 'Default_Retrosplenial',
    'Visual_Lateral', 'Visual_Dorsal_VentralStream', 'Visual_V5', 'Visual_V1',
    'Frontoparietal', 'DorsalAttention', 'Premotor_DorsalAttentionII',
    'Language', 'Salience', 'CinguloOpercular_Action_mode',
    'MedialParietal', 'Somatomotor_Hand', 'Somatomotor_Face',
    'Somatomotor_Foot', 'Auditory', 'SomatoCognitiveAction'
]

candidate_ados_full = [
    'ADI_RRB_TOTAL_C','ADOS_STEREO_BEHAV', 'ADOS_GOTHAM_RRB',
    'SRS_MANNERISMS','VINELAND_PERSONAL_V_SCALED',
    'VINELAND_DOMESTIC_V_SCALED', 'VINELAND_PLAY_V_SCALED',
    'VINELAND_COPING_V_SCALED',
    'ADOS_TOTAL', 'ADOS_COMM', 'ADOS_SOCIAL', 'ADOS_GOTHAM_SOCAFFECT',
    'ADI_R_SOCIAL_TOTAL_A', 'ADI_R_VERBAL_TOTAL_BV',
    'SRS_RAW_TOTAL', 'SRS_AWARENESS', 'SRS_COGNITION',
    'SRS_COMMUNICATION', 'SRS_MOTIVATION'
]

# Filter valid clusters only
df = df[df['Cluster'].isin([0, 1, 2])]
permutations = 1000
rng = np.random.default_rng(42)

results = []
weights_all = []
sig_tracker = {}

# ========= Step 2: Iterate over clusters and behavioral scales =========
for scale in candidate_ados_full:
    sig_tracker[scale] = []

    for cluster in [0, 1, 2]:
        df_cluster = df[df['Cluster'] == cluster]

        if not all(col in df_cluster.columns for col in columns_to_fit + [scale]):
            continue

        temp = df_cluster[columns_to_fit + [scale]].dropna()
        if temp.shape[0] < 30:  # Ensure enough samples
            continue

        X = temp[columns_to_fit].values
        Y = temp[[scale]].values

        # Standardization
        scaler_X = StandardScaler()
        scaler_Y = StandardScaler()
        X_std = scaler_X.fit_transform(X)
        Y_std = scaler_Y.fit_transform(Y)

        # ========= Step 3: Run RCCA =========
        cca = CCA(numCC=1, reg=0.1)
        cca.train([X_std, Y_std])
        X_c = cca.comps[0]
        Y_c = cca.comps[1]

        r_true, _ = pearsonr(X_c[:, 0], Y_c[:, 0])

        # ========= Step 4: Permutation test =========
        r_null = []
        for _ in range(permutations):
            Y_perm = rng.permutation(Y_std)
            cca_perm = CCA(numCC=1, reg=0.1)
            cca_perm.train([X_std, Y_perm])
            Y_perm_c = cca_perm.comps[1]
            r_perm, _ = pearsonr(X_c[:, 0], Y_perm_c[:, 0])
            r_null.append(r_perm)

        p_val = np.mean(np.abs(r_null) >= np.abs(r_true))

        if p_val < 0.05:
            sig_tracker[scale].append(True)

            # Save canonical variates for significant results
            comp_df = pd.DataFrame({
                "X_c": X_c[:, 0],
                "Y_c": Y_c[:, 0]
            })
            filename = f"./subtype/cca_components_cluster{cluster}_{scale}.csv".replace("/", "_")
            comp_df.to_csv(filename, index=False)

        else:
            sig_tracker[scale].append(False)

        results.append({
            "Cluster": cluster,
            "Scale": scale,
            "CCA_Component": 1,
            "r": round(r_true, 4),
            "p": round(p_val, 4)
        })

        weights_all.append({
            "Cluster": cluster,
            "Scale": scale,
            "CCA_Component": 1,
            **{f"{col}_weight": round(w, 4) for col, w in zip(columns_to_fit, cca.ws[0][:, 0])}
        })

# ========= Step 5: Keep scales significant across all clusters =========
valid_scales = [scale for scale, sigs in sig_tracker.items() if sum(sigs) == 3]

results_df = pd.DataFrame(results)
weights_df = pd.DataFrame(weights_all)

results_df = results_df[results_df["Scale"].isin(valid_scales)]
weights_df = weights_df[weights_df["Scale"].isin(valid_scales)]

results_df.to_csv("./subtype/rcca_significant_results_.csv", index=False)
weights_df.to_csv("./subtype/rcca_component_weights_.csv", index=False)

print("Significant behavioral scales (consistently significant across all clusters):")
print(results_df["Scale"].unique())

# ========= Step 6: Extract top-N contributing features per row =========
df = pd.read_csv('./subtype/rcca_component_weights_.csv', index_col=0)
df_numeric = df.select_dtypes(include=[np.number])
top_features = []

for idx, row in df_numeric.iterrows():
    top_indices = row.abs().nlargest(5).index
    top_values = row[top_indices]
    result = {'Index': idx}
    for i, (col, val) in enumerate(top_values.items(), 1):
        result[f'Feature_{i}'] = col
        result[f'Value_{i}'] = val
    top_features.append(result)

top_df = pd.DataFrame(top_features)
top_df.to_csv('./subtype/rtop5_features_per_row_.csv', index=False)
print("Top 5 features per row saved to: ./subtype/rtop5_features_per_row.csv")
