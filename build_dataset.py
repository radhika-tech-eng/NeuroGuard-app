
import re
import os
import glob
import numpy as np
import mne

BASE_DIR = "data/raw/chbmit"
OUTPUT_DIR = "data/processed"
WINDOW_SECONDS = 5

X_all = []
y_all = []
patient_ids = []
recording_ids = []

reference_channels = None
skipped = []

for patient_dir in sorted(glob.glob(os.path.join(BASE_DIR, "chb*"))):
    if not os.path.isdir(patient_dir):
        continue

    patient = os.path.basename(patient_dir)
    summary_path = os.path.join(patient_dir, f"{patient}-summary.txt")

    if not os.path.isfile(summary_path):
        print(f"Skipping {patient}: summary missing")
        continue

    with open(summary_path, encoding="utf-8") as f:
        summary = f.read()

    blocks = re.split(r"(?=File Name:)", summary)

    edf_files = {
        os.path.basename(p): p
        for p in glob.glob(os.path.join(patient_dir, "*.edf"))
        if os.path.isfile(p)
    }

    for block in blocks:
        match = re.search(r"File Name:\s*(\S+)", block)

        if not match:
            continue

        filename = match.group(1)
        path = edf_files.get(filename)

        if path is None:
            continue

        count_match = re.search(
            r"Number of Seizures in File:\s*(\d+)", block
        )

        if not count_match:
            print(f"Skipping {filename}: seizure annotation missing")
            skipped.append(filename)
            continue

        count = int(count_match.group(1))

        starts = [
            int(v) for v in re.findall(
                r"Seizure(?:\s+\d+)? Start Time:\s*(\d+)", block
            )
        ]

        ends = [
            int(v) for v in re.findall(
                r"Seizure(?:\s+\d+)? End Time:\s*(\d+)", block
            )
        ]

        if len(starts) != count or len(ends) != count:
            print(f"Skipping {filename}: annotation mismatch")
            skipped.append(filename)
            continue

        print(f"\nProcessing {patient}/{filename}")

        try:
            raw = mne.io.read_raw_edf(
                path,
                preload=True,
                verbose="ERROR"
            )

            raw.pick("eeg", exclude=[])

            if len(raw.ch_names) == 0:
                print(f"Skipping {filename}: no EEG channels")
                skipped.append(filename)
                continue

            raw.filter(1.0, 40.0, verbose="ERROR")

            channels = list(raw.ch_names)

            if reference_channels is None:
                reference_channels = channels
                print("Reference channels:", len(reference_channels))

            elif channels != reference_channels:
                print(f"Skipping {filename}: channel layout differs")
                skipped.append(filename)
                continue

            data = raw.get_data()
            fs = int(raw.info["sfreq"])
            window = WINDOW_SECONDS * fs

            if window <= 0:
                print(f"Skipping {filename}: invalid sampling rate")
                skipped.append(filename)
                continue

            file_windows = 0

            for i in range(0, data.shape[1] - window + 1, window):
                start_sec = i / fs
                end_sec = (i + window) / fs

                label = int(any(
                    start_sec < seizure_end and end_sec > seizure_start
                    for seizure_start, seizure_end in zip(starts, ends)
                ))

                X_all.append(data[:, i:i + window])
                y_all.append(label)
                patient_ids.append(patient)
                recording_ids.append(filename)

                file_windows += 1

            print(f"Finished {filename}: {file_windows} windows")

        except Exception as exc:
            skipped.append(filename)
            print(f"Error processing {filename}: {exc}")

if not X_all:
    raise RuntimeError(
        "No windows generated. Check EDF files, summaries, "
        "permissions, and channel compatibility."
    )

X = np.stack(X_all).astype(np.float32)
y = np.asarray(y_all, dtype=np.int64)
patient_ids = np.asarray(patient_ids, dtype=str)
recording_ids = np.asarray(recording_ids, dtype=str)

# Verify that all arrays have matching lengths.
assert len(X) == len(y) == len(patient_ids) == len(recording_ids)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Save to separate files; preserve existing datasets.
outputs = {
    "eeg_windows_new.npy": X,
    "eeg_labels_new.npy": y,
    "patient_ids_new.npy": patient_ids,
    "recording_ids_new.npy": recording_ids,
}

for filename, array in outputs.items():
    output_path = os.path.join(OUTPUT_DIR, filename)
    temp_path = output_path + ".tmp.npy"

    np.save(temp_path, array)
    os.replace(temp_path, output_path)

print("\nNew dataset built successfully.")
print("X shape:", X.shape)
print("y shape:", y.shape)
print("Patient IDs shape:", patient_ids.shape)
print("Recording IDs shape:", recording_ids.shape)
print("Label counts:", dict(zip(*np.unique(y, return_counts=True))))
print("Patients included:", np.unique(patient_ids).tolist())
print("Skipped recordings:", skipped)
print("Saved new dataset files:")
for filename in outputs:
    print(os.path.join(OUTPUT_DIR, filename))
