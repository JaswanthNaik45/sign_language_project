import os
import pandas as pd

base_dir = "own_gestures"
all_data = []

for gesture in os.listdir(base_dir):
    gesture_path = os.path.join(base_dir, gesture)
    csv_path = os.path.join(gesture_path, "landmarks.csv")

    if os.path.isfile(csv_path):
        df = pd.read_csv(csv_path)
        df["label"] = gesture
        all_data.append(df)

final_df = pd.concat(all_data, ignore_index=True)

os.makedirs("final_dataset", exist_ok=True)
final_df.to_csv("final_dataset/gestures.csv", index=False)

print("Merged dataset saved as final_dataset/gestures.csv")
print("Total samples:", final_df.shape[0])
print("Total features:", final_df.shape[1])