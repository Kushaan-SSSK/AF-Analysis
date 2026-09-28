# Preparing new recordings

This guide takes EKG recordings from a lab's own mice to the format of the published dataset, so they can be scored with `predict.py` or added to the training data. It takes three steps: export the recordings as text, fill in one metadata row per recording, and run `prepare_dataset.py`.

## What you start with

Recordings made on an iWorx system with LabScribe, saved as `.iwxdata` files. Each recording has two EKG leads, which LabScribe calls `Raw Ch 1` and `Raw Ch 2`. The model needs at least 40 s of recording and reads the leads at up to 1,000 Hz. Any iWorx sampling rate works (the dataset has 500, 2,000 and 5,000 Hz recordings).

## 1. Export each recording as text

Open the recording in LabScribe and export the whole recording as a tab-separated text file. The export must include the `Time` column and both raw channels. Extra channels (such as LabScribe's computed `Comp Ch` channels) are fine and are ignored.

Menu names differ between LabScribe versions. The export is right if the first lines of the file look like this:

```
Time	Raw Ch 1	Raw Ch 2
0	-12.1302	4.24404
0.0002	-12.1302	4.24404
```

Keep LabScribe's default file name (for example `134405-131592-1-TBX5H2B_Export.txt`); the script matches exports to the metadata by that name. LabScribe sometimes saves these files with a `.xls` extension even though they are plain text; both `.txt` and `.xls` are accepted.

## 2. Fill in the metadata

Copy `docs/metadata_template.csv` and add one row per recording:

| Column | What to enter |
|---|---|
| `original_filename` | the recording's file name, with or without `.iwxdata` or `_Export` |
| `animal_id` | the mouse's ID; use the same ID for every recording of the same mouse |
| `session_date` | recording date as YYYY-MM-DD, if known |
| `sex`, `age`, `genotype`, `treatment`, `disease_or_control` | as you record them; leave blank if unknown |
| `rhythm` | one of: AF, Paroxysmal AF, AF/atrial flutter, Sinus rhythm, Junctional rhythm, Atrial bigeminy, Unknown |
| `annotator_notes` | anything useful, such as episode times for paroxysmal AF |

AF, Paroxysmal AF and AF/atrial flutter become `af_label` 1; Sinus rhythm, Junctional rhythm and Atrial bigeminy become 0. A recording marked Unknown or left blank is kept in the dataset but not used for training.

The animal ID matters. Cross-validation keeps all recordings of one mouse on the same side of each split, so recordings of the same mouse must share an ID.

## 3. Run the script

```
python prepare_dataset.py --exports path/to/exports --metadata metadata.csv --lab XX --out path/to/dataset
```

`--lab` is a short code for your lab (the published dataset uses JJ, SF and MS). The script:
- matches every metadata row to its export;
- checks that the export has `Time` and `Raw Ch 1` and reads its duration and sampling rate;
- copies it to `recordings/xx/XX-001.txt`, continuing the numbering if the dataset already has recordings from your lab;
- appends a row to `recordings.csv`, in the same columns as the published dataset.

Exports without a metadata row, and metadata rows without an export, are listed at the end and left out. Use `--label-source` to record where the rhythm calls came from (the default is "XX lab spreadsheet").

## 4. Check and use the result

`path/to/dataset` now has the same layout as the published dataset:

```
recordings.csv
recordings/xx/XX-001.txt
recordings/xx/XX-002.txt
...
```

To score the new recordings without retraining:

```
python predict.py path/to/dataset/recordings/xx/*.txt --mode sensitive --report review.html
```

To see every recording with its metadata in one page, run `databank_report.py --databank path/to/dataset`.

To add your recordings to the published dataset, point `--out` at the folder where you unzipped it. The script appends to its `recordings.csv` and numbers your recordings after any existing ones from your lab. Then retrain with `train.py --databank path/to/dataset`.
