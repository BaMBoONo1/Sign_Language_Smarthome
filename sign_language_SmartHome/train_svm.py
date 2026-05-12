import os
import glob
import csv
import json
import numpy as np
import cv2

# ================= Configuration =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'gesture_dataset')
TARGET_FRAMES = 20
MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'gesture_svm_model.xml')
LABEL_MAP_PATH = os.path.join(BASE_DIR, 'gesture_labels.json')
# =================================================


def extract_features(data_rows, target_frames=TARGET_FRAMES):
    """
    Interpolate the time-series angle and position data to a fixed number of frames.
    data_rows: [ [rx, ry, lx, ly, r_angle0, ... l_angle14, ...], ... ]
    """
    data = np.array(data_rows, dtype=np.float32)
    
    # 1. Scaling: Normalize angles (0~180) to 0~1 range to match coordinates (0~1)
    # This ensures that SVM doesn't ignore motion (x, y) which has smaller variance.
    data[:, 4:] = data[:, 4:] / 180.0
    
    # 2. Relative Position Normalization
    # Find the first frame where at least one hand is detected to use as reference
    ref_x, ref_y = 0.0, 0.0
    for frame in data:
        if frame[0] != 0 or frame[1] != 0: # Right hand first
            ref_x, ref_y = frame[0], frame[1]
            break
        elif frame[2] != 0 or frame[3] != 0: # Left hand fallback
            ref_x, ref_y = frame[2], frame[3]
            break
            
    if ref_x != 0 or ref_y != 0:
        # Subtract reference from all detected hands to get relative motion
        data[:, 0] = np.where(data[:, 0] != 0, data[:, 0] - ref_x, 0)
        data[:, 1] = np.where(data[:, 1] != 0, data[:, 1] - ref_y, 0)
        data[:, 2] = np.where(data[:, 2] != 0, data[:, 2] - ref_x, 0)
        data[:, 3] = np.where(data[:, 3] != 0, data[:, 3] - ref_y, 0)
    
    num_frames = data.shape[0]
    num_features = data.shape[1]
    
    if num_frames == target_frames:
        resampled = data
    else:
        resampled = np.zeros((target_frames, num_features), dtype=np.float32)
        original_indices = np.linspace(0, 1, num_frames)
        target_indices = np.linspace(0, 1, target_frames)
        
        for i in range(num_features):
            resampled[:, i] = np.interp(target_indices, original_indices, data[:, i])
            
    return resampled.flatten()

def main():
    print("Searching for gesture data CSV files...")
    csv_files = glob.glob(os.path.join(DATA_DIR, "*_data_*.csv"))
    
    if not csv_files:
        print("No CSV files found in the current directory.")
        print("Make sure you are running this from the directory where the CSV files are saved.")
        return
        
    X = []
    y_str = []
    
    print(f"Found {len(csv_files)} gesture files. Processing...")
    for file in csv_files:
        basename = os.path.basename(file)
        gesture_name = basename.split('_data_')[0]
        
        try:
            data = []
            with open(file, 'r', newline='') as f:
                reader = csv.reader(f)
                header = next(reader)
                num_cols = len(header)
                
                is_old_format = False
                if num_cols == 16: # time_sec + 15 angles
                    is_old_format = True
                elif num_cols < 18:
                    print(f"Skipping {basename} (Unknown format: {num_cols} columns)")
                    continue

                for row in reader:
                    vals = [float(val) for val in row[1:]]
                    if is_old_format:
                        # Insert 0.0 for x and y at the beginning
                        vals = [0.0, 0.0] + vals
                    data.append(vals)
                    
            if len(data) < 5:
                print(f"Skipping {basename} (Sequence too short)")
                continue
                
            features = extract_features(data)
            X.append(features)
            y_str.append(gesture_name)
        except Exception as e:
            print(f"Error processing {file}: {e}")
            
    # OpenCV SVM requires integer labels
    unique_labels = sorted(list(set(y_str)))
    label_to_int = {lbl: i for i, lbl in enumerate(unique_labels)}
    int_to_label = {i: lbl for i, lbl in enumerate(unique_labels)}
    
    y = [label_to_int[lbl] for lbl in y_str]
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    
    print(f"Loaded {len(X)} samples with {X.shape[1]} features each.")
    print("\nClass distribution:")
    for lbl in unique_labels:
        count = y_str.count(lbl)
        print(f" - {lbl}: {count} samples")
    
    if len(X) < 10:
        print("Dataset is too small to split meaningfully! Collect more data.")
        return
        
    # Split data into training and test sets
    indices = np.arange(len(X))
    np.random.seed(42)
    np.random.shuffle(indices)
    split_idx = int(0.8 * len(X))
    train_idx, test_idx = indices[:split_idx], indices[split_idx:]
    
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    
    print("\nTraining OpenCV SVM model...")
    # scikit-learn의 기본 gamma='scale' 방식을 모방 (분산 역순)
    var = np.var(X_train) if np.var(X_train) > 0 else 1.0
    gamma = 1.0 / (X_train.shape[1] * var)
    
    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(cv2.ml.SVM_RBF)
    svm.setC(50.0) # 과적합을 허용하므로 마진 페널티를 크게 설정
    svm.setGamma(gamma)
    
    svm.train(X_train, cv2.ml.ROW_SAMPLE, y_train)
    
    # Evaluate model on the train and test split
    _, y_pred_train = svm.predict(X_train)
    train_acc = np.mean(y_pred_train.flatten().astype(np.int32) == y_train)
    
    _, y_pred_test = svm.predict(X_test)
    test_acc = np.mean(y_pred_test.flatten().astype(np.int32) == y_test)
    
    print(f"\nModel Evaluation:")
    print(f" - Train Accuracy (과적합 지표): {train_acc*100:.2f}%")
    print(f" - Test Accuracy (검증 지표): {test_acc*100:.2f}%")
    
    # 성능 극대화를 위해 *전체 데이터*로 최종 재학습 (과적합 허용)
    print("Retraining on all data for maximum deployment performance...")
    svm.train(X, cv2.ml.ROW_SAMPLE, y)
    
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    svm.save(MODEL_SAVE_PATH)
    print(f"Model successfully saved to {MODEL_SAVE_PATH}")
    
    with open(LABEL_MAP_PATH, 'w') as f:
        json.dump(int_to_label, f)
    print(f"Label assignments saved to {LABEL_MAP_PATH}")

if __name__ == '__main__':
    main()
