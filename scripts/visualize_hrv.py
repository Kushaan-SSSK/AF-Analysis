import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os



def generate_hrv_plots():
    # Load analysis results and ground truth annotations
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    results_dir = os.path.join(base_dir, "Results")


    out_dir = os.path.join(results_dir, "hrv_visualizations")
    os.makedirs(out_dir, exist_ok=True)

    # Needs the full joined dataframe that `train_classifier.py` builds in memory
    # Quick re-implementation of the merge to get proper manual labels:

    summary_path = os.path.join(results_dir, "all_results_summary.csv")
    annot_path = os.path.join(base_dir, "EKG_Annotations.xlsx")

    
    if not os.path.exists(summary_path) or not os.path.exists(annot_path):
        print("Required files not found. Ensure all_results_summary.csv and EKG_Annotations exist.")
        return

        
    df = pd.read_csv(summary_path)
    gt_df = pd.read_excel(annot_path)

    # Clean GT
    gt_df.columns = [c.strip() for c in gt_df.columns]
    file_col = gt_df.columns[0]
    label_col = next((c for c in gt_df.columns if c.strip() == 'AF' or 'AF' in c), None)
    ep_start_col = next((c for c in gt_df.columns if 'Episode Start' in c), None)
    ep_end_col = next((c for c in gt_df.columns if 'Episode End' in c), None)

    

    valid_labels = {'yes', 'no', 'sustained', 'paroxysmal'}
    gt_map = {}

    
    for _, row in gt_df.iterrows():
        fname = str(row[file_col]).strip()
        label_str = str(row[label_col]).lower().strip()

        if not any(v in label_str for v in valid_labels):
            continue

        base_fname = os.path.splitext(os.path.basename(fname))[0].lower().strip()


        start_t = float(row[ep_start_col]) if ep_start_col and pd.notna(row[ep_start_col]) else None
        end_t = float(row[ep_end_col]) if ep_end_col and pd.notna(row[ep_end_col]) else None

        

        if 'sustained' in label_str: is_af, af_type = True, 'sustained'
        elif 'paroxysmal' in label_str: is_af, af_type = True, 'paroxysmal'
        elif 'yes' in label_str: is_af, af_type = True, 'af'
        else: is_af, af_type = False, 'no_af'

        

        if base_fname not in gt_map:
            gt_map[base_fname] = {'is_af': is_af, 'af_type': af_type, 'episodes': []}

            

        if start_t is not None and end_t is not None:
             gt_map[base_fname]['episodes'].append((start_t, end_t))

             

    def normalize_name(name):
        return os.path.splitext(os.path.basename(name))[0].replace('_Export', '').replace('.iwxdata', '').lower().strip()

        

    def get_window_label(row):
        norm_row = normalize_name(row['file'])

        

        file_ann = gt_map.get(norm_row)
        if not file_ann:
            for k, v in gt_map.items():
                if norm_row in k or k in norm_row:
                    file_ann = v; break

                    

        if not file_ann: return None

        w_start, w_end = row['start_time'], row['end_time']
        is_af, af_type, eps = file_ann['is_af'], file_ann['af_type'], file_ann.get('episodes', [])

        

        if not is_af: return 'No-AF'

        if is_af and af_type == 'sustained': return 'AF'

        if is_af and af_type == 'paroxysmal':
            if not eps: return 'AF'
            for ep_s, ep_e in eps:
                overlap = max(0, min(w_end, ep_e) - max(w_start, ep_s))
                if overlap >= 5.0: return 'AF'

            return 'No-AF'

            

        return None


    df['Label'] = df.apply(get_window_label, axis=1)
    df_clean = df.dropna(subset=['Label'])

    

    # Plotting
    features_to_plot = ['pNN20', 'NN20', 'pRR_3.25']


    sns.set_theme(style="whitegrid")

    
    for feature in features_to_plot:
        if feature not in df_clean.columns: continue

    
        plt.figure(figsize=(8, 6))

        

        # Draw Boxplot
        ax = sns.boxplot(x='Label', y=feature, data=df_clean, palette={'No-AF': '#2ecc71', 'AF': '#e74c3c'})

        

        # Calculate P-Value roughly for annotation
        af_vals = df_clean[df_clean['Label'] == 'AF'][feature].dropna()
        noaf_vals = df_clean[df_clean['Label'] == 'No-AF'][feature].dropna()

        

        from scipy import stats
        t_stat, p_val = stats.ttest_ind(af_vals, noaf_vals, equal_var=False)
        p_str = "p < 0.001" if p_val < 0.001 else f"p = {p_val:.3f}"

        

        plt.title(f"{feature} Variation\nAF vs Normal Sinus Rhythm ({p_str})", fontsize=14, fontweight='bold')
        plt.ylabel(feature, fontsize=12)
        plt.xlabel("Rhythm Type", fontsize=12)

        

        plt.tight_layout()
        out_file = os.path.join(out_dir, f"{feature}_difference.png")
        plt.savefig(out_file, dpi=300)
        plt.close()

        

        print(f"Generated {out_file}")



if __name__ == "__main__":
    generate_hrv_plots()

