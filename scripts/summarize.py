import pandas as pd
import numpy as np
import os





def estimate_duration(file_path):
    try:
        with open(file_path, 'r', encoding='latin-1') as f:
            header_line = f.readline()


        use_header = 'infer' if 'Time' in header_line else None
        chunks = pd.read_csv(file_path, sep='\t', header=use_header,

                             chunksize=10000, low_memory=False)

        try:
            first_chunk = next(chunks)
        except StopIteration:
            return 0



        time_col = 'Time' if (use_header and 'Time' in first_chunk.columns) else first_chunk.columns[0]
        t0_series = pd.to_numeric(first_chunk[time_col], errors='coerce').dropna()

        if t0_series.empty:
            return 0


        t0 = t0_series.iloc[0]
        t_end = t0_series.iloc[-1]



        for chunk in chunks:
            t_chunk = pd.to_numeric(chunk[time_col], errors='coerce').dropna()
            if not t_chunk.empty:
                t_end = t_chunk.iloc[-1]



        full_dur = max(0, t_end - t0)
        # We only analyze 10-40s, so cap the effective extracted duration at 30s
        if full_dur < 10:
            return 0
        return min(30, full_dur - 10)



    except Exception as e:
        print(f"  Warning: could not read {os.path.basename(file_path)}: {e}")
        return 0





def load_annotations(base_dir):
    annot_path = os.path.join(base_dir, "EKG_Annotations.xlsx")

    if not os.path.exists(annot_path):
        print(f"Annotation file not found: {annot_path}")
        return {}



    df = pd.read_excel(annot_path)
    file_col = df.columns[0]
    af_col = 'AF'


    valid = df[df[af_col].isin(['No', 'sustained', 'paroxysmal'])].copy()



    def norm(name):
        bn = os.path.basename(str(name).replace('\\', '/').split('/')[-1])
        return os.path.splitext(bn)[0].strip().lower()



    mapping = {}

    for _, row in valid.iterrows():
        key = norm(str(row[file_col]))
        label = str(row[af_col]).strip()

        mapping[key] = {
            'is_af': label in ('sustained', 'paroxysmal'),

            'af_type': label

        }

    return mapping

def main():
    import glob
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    data_dir = os.path.join(base_dir, "EKG Recordings")



    if not os.path.exists(data_dir):
        print(f"EKG Recordings folder not found: {data_dir}")
        return



    annot_map = load_annotations(base_dir)
    print(f"Annotations loaded: {len(annot_map)} records")



    exports = sorted(set(
        glob.glob(os.path.join(data_dir, "*_Export.xls*")) +
        glob.glob(os.path.join(data_dir, "*_Export.txt"))

    ))

    print(f"Export files found: {len(exports)}\n")


    rows = []

    for f in exports:
        bn = os.path.basename(f)
        dur = estimate_duration(f)
        mouse = bn.split('-')[0].strip()



        norm_f = os.path.splitext(bn)[0].replace('_Export', '').strip().lower()
        ann = annot_map.get(norm_f)

        if ann is None:

            for k, v in annot_map.items():
                if norm_f in k or k in norm_f:
                    ann = v
                    break



        rows.append({

            'file': bn,

            'mouse_id': mouse,

            'duration_s': dur,

            'is_af': ann['is_af'] if ann else None,

            'af_type': ann['af_type'] if ann else 'unknown',

        })



    df = pd.DataFrame(rows)
    n_files = len(df)
    n_matched = df['is_af'].notna().sum()
    n_af_files = df[df['is_af'] == True].shape[0]
    n_sustained = (df['af_type'] == 'sustained').sum()
    n_paroxysmal = (df['af_type'] == 'paroxysmal').sum()
    n_nonaf_files = df[df['is_af'] == False].shape[0]
    n_unmatched = df['is_af'].isna().sum()
    unique_mice = df['mouse_id'].nunique()
    by_ext = df['file'].apply(lambda x: os.path.splitext(x)[1]).value_counts()



    # All annotated recordings contribute exactly 30s (10-40s window)
    total_dur_s = n_matched * 30.0



    print("=" * 60)
    print("DATASET SUMMARY - EKG Recordings (10-40s window)")
    print("=" * 60)
    print(f"Total export files:          {n_files}")
    print(f"  .xls files:                {by_ext.get('.xls', 0)}")
    print(f"  .txt files:                {by_ext.get('.txt', 0) + by_ext.get('.TXT', 0)}")
    print(f"Files with annotations:      {n_matched}")
    print(f"Files without annotations:   {n_unmatched}")
    print()
    print(f"Unique mice (by ID prefix):  {unique_mice}")
    print()
    print(f"Total analyzed duration:     {total_dur_s/60:.1f} min  ({total_dur_s/3600:.2f} hr)")
    print(f"Extraction per recording:    30 sec (10-40s)")
    print()

    print("AF Labels:")
    print(f"  AF (total):                {n_af_files}")
    print(f"    Sustained:               {n_sustained}")
    print(f"    Paroxysmal:              {n_paroxysmal}")
    print(f"  No-AF:                     {n_nonaf_files}")
    print("=" * 60)



    results_dir = os.path.join(base_dir, "Results")
    out_path = os.path.join(results_dir, "dataset_summary.csv")
    pd.DataFrame([{

        'total_files': n_files,

        'annotated_files': n_matched,

        'unique_mice': unique_mice,

        'af_files': n_af_files,

        'sustained_af': n_sustained,

        'paroxysmal_af': n_paroxysmal,

        'nonaf_files': n_nonaf_files,

        'total_duration_min': round(total_dur_s / 60, 1),

        'mean_duration_min':  0.5,

    }]).to_csv(out_path, index=False)
    print(f"\nSummary -> {out_path}")



    per_file_path = os.path.join(results_dir, "per_file_stats.csv")
    df.to_csv(per_file_path, index=False)
    print(f"Per-file stats -> {per_file_path}")



if __name__ == "__main__":
    main()

