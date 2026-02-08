"""
Simple Baseline Test - Check if ANY predictive signal exists in the data
Uses simple ML models (not deep learning) to establish baseline performance.
If simple models can't beat random, complex models won't either.
"""

import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("BASELINE TEST: Does ANY predictive signal exist?")
print("="*80)

# Load data
print("\nLoading windowed data...")
with open('data/processed/indian_windowed.pkl', 'rb') as f:
    data = pickle.load(f)

# Extract features and labels
X_train = data['train']['windows']  # (N, 20, 52)
y_train = data['train']['targets_movement']

X_val = data['val']['windows']
y_val = data['val']['targets_movement']

X_test = data['test']['windows']
y_test = data['test']['targets_movement']

print(f"Train: {len(X_train):,} samples")
print(f"Val:   {len(X_val):,} samples")  
print(f"Test:  {len(X_test):,} samples")

# Flatten windows to 2D or use last timestep
# Option 1: Use only the last timestep (most recent features)
X_train_last = X_train[:, -1, :]  # Shape: (N, 52)
X_val_last = X_val[:, -1, :]
X_test_last = X_test[:, -1, :]

# Option 2: Use aggregated statistics over all timesteps
def extract_stats(X):
    """Extract statistics from time series"""
    # Mean, std, min, max, last value for each feature
    mean = X.mean(axis=1)
    std = X.std(axis=1)
    last = X[:, -1, :]
    first = X[:, 0, :]
    trend = last - first  # Change over window
    return np.concatenate([mean, std, last, trend], axis=1)

X_train_stats = extract_stats(X_train)
X_val_stats = extract_stats(X_val)
X_test_stats = extract_stats(X_test)

print(f"\nFeatures (last timestep): {X_train_last.shape[1]}")
print(f"Features (statistics): {X_train_stats.shape[1]}")

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_last)
X_val_scaled = scaler.transform(X_val_last)
X_test_scaled = scaler.transform(X_test_last)

scaler_stats = StandardScaler()
X_train_stats_scaled = scaler_stats.fit_transform(X_train_stats)
X_val_stats_scaled = scaler_stats.transform(X_val_stats)
X_test_stats_scaled = scaler_stats.transform(X_test_stats)

# Class distribution
pos_ratio_train = y_train.mean()
pos_ratio_test = y_test.mean()
print(f"\nClass distribution:")
print(f"  Train positive ratio: {pos_ratio_train:.2%}")
print(f"  Test positive ratio:  {pos_ratio_test:.2%}")

# Random baseline
print("\n" + "="*80)
print("BASELINE 1: Random Guessing")
print("="*80)
random_preds = np.random.randint(0, 2, len(y_test))
print(f"Accuracy: {accuracy_score(y_test, random_preds):.4f}")
print(f"Balanced Accuracy: {balanced_accuracy_score(y_test, random_preds):.4f}")

# Always predict majority class
print("\n" + "="*80)
print("BASELINE 2: Always predict majority class")
print("="*80)
majority_class = 1 if pos_ratio_test > 0.5 else 0
majority_preds = np.full(len(y_test), majority_class)
print(f"Accuracy: {accuracy_score(y_test, majority_preds):.4f}")
print(f"Balanced Accuracy: {balanced_accuracy_score(y_test, majority_preds):.4f}")

# Logistic Regression
print("\n" + "="*80)
print("MODEL 1: Logistic Regression (last timestep)")
print("="*80)
lr = LogisticRegression(max_iter=1000, C=0.1, class_weight='balanced', n_jobs=-1)
lr.fit(X_train_scaled, y_train)
lr_preds = lr.predict(X_test_scaled)
lr_probs = lr.predict_proba(X_test_scaled)[:, 1]
print(f"Accuracy: {accuracy_score(y_test, lr_preds):.4f}")
print(f"Balanced Accuracy: {balanced_accuracy_score(y_test, lr_preds):.4f}")
print(f"AUC-ROC: {roc_auc_score(y_test, lr_probs):.4f}")
print(f"Pred positive ratio: {lr_preds.mean():.2%}")

# Logistic Regression with statistics
print("\n" + "="*80)
print("MODEL 2: Logistic Regression (statistics)")
print("="*80)
lr_stats = LogisticRegression(max_iter=1000, C=0.1, class_weight='balanced', n_jobs=-1)
lr_stats.fit(X_train_stats_scaled, y_train)
lr_stats_preds = lr_stats.predict(X_test_stats_scaled)
lr_stats_probs = lr_stats.predict_proba(X_test_stats_scaled)[:, 1]
print(f"Accuracy: {accuracy_score(y_test, lr_stats_preds):.4f}")
print(f"Balanced Accuracy: {balanced_accuracy_score(y_test, lr_stats_preds):.4f}")
print(f"AUC-ROC: {roc_auc_score(y_test, lr_stats_probs):.4f}")
print(f"Pred positive ratio: {lr_stats_preds.mean():.2%}")

# Random Forest
print("\n" + "="*80)
print("MODEL 3: Random Forest")
print("="*80)
rf = RandomForestClassifier(
    n_estimators=100, 
    max_depth=10, 
    min_samples_leaf=50,
    class_weight='balanced',
    n_jobs=-1,
    random_state=42
)
rf.fit(X_train_stats_scaled, y_train)
rf_preds = rf.predict(X_test_stats_scaled)
rf_probs = rf.predict_proba(X_test_stats_scaled)[:, 1]
print(f"Accuracy: {accuracy_score(y_test, rf_preds):.4f}")
print(f"Balanced Accuracy: {balanced_accuracy_score(y_test, rf_preds):.4f}")
print(f"AUC-ROC: {roc_auc_score(y_test, rf_probs):.4f}")
print(f"Pred positive ratio: {rf_preds.mean():.2%}")

# Gradient Boosting (more powerful)
print("\n" + "="*80)
print("MODEL 4: Gradient Boosting")
print("="*80)
# Use subset for faster training
n_subset = min(50000, len(X_train_stats_scaled))
idx = np.random.choice(len(X_train_stats_scaled), n_subset, replace=False)
gb = GradientBoostingClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    random_state=42
)
gb.fit(X_train_stats_scaled[idx], y_train[idx])
gb_preds = gb.predict(X_test_stats_scaled)
gb_probs = gb.predict_proba(X_test_stats_scaled)[:, 1]
print(f"Accuracy: {accuracy_score(y_test, gb_preds):.4f}")
print(f"Balanced Accuracy: {balanced_accuracy_score(y_test, gb_preds):.4f}")
print(f"AUC-ROC: {roc_auc_score(y_test, gb_probs):.4f}")
print(f"Pred positive ratio: {gb_preds.mean():.2%}")

# Feature importance from Random Forest
print("\n" + "="*80)
print("FEATURE IMPORTANCE (Top 20)")
print("="*80)
feature_names = ['mean_' + str(i) for i in range(52)] + \
                ['std_' + str(i) for i in range(52)] + \
                ['last_' + str(i) for i in range(52)] + \
                ['trend_' + str(i) for i in range(52)]
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1][:20]
for i, idx_ in enumerate(indices):
    print(f"  {i+1:2d}. {feature_names[idx_]:15s}: {importances[idx_]:.4f}")

# Summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print("""
| Model                  | Accuracy | Bal.Acc | AUC    |
|------------------------|----------|---------|--------|
| Random                 | ~0.50    | 0.50    | 0.50   |
| Majority class         | {:.4f}   | 0.50    | -      |
| Logistic Reg (last)    | {:.4f}   | {:.4f}  | {:.4f} |
| Logistic Reg (stats)   | {:.4f}   | {:.4f}  | {:.4f} |
| Random Forest          | {:.4f}   | {:.4f}  | {:.4f} |
| Gradient Boosting      | {:.4f}   | {:.4f}  | {:.4f} |
""".format(
    accuracy_score(y_test, majority_preds),
    accuracy_score(y_test, lr_preds), balanced_accuracy_score(y_test, lr_preds), roc_auc_score(y_test, lr_probs),
    accuracy_score(y_test, lr_stats_preds), balanced_accuracy_score(y_test, lr_stats_preds), roc_auc_score(y_test, lr_stats_probs),
    accuracy_score(y_test, rf_preds), balanced_accuracy_score(y_test, rf_preds), roc_auc_score(y_test, rf_probs),
    accuracy_score(y_test, gb_preds), balanced_accuracy_score(y_test, gb_preds), roc_auc_score(y_test, gb_probs)
))

if roc_auc_score(y_test, gb_probs) < 0.52:
    print("""
⚠️  CONCLUSION: Very weak predictive signal
The best model achieves AUC < 0.52, which is barely better than random.
This suggests:
1. Next-day stock direction is very hard to predict from technical indicators
2. The efficient market hypothesis may be at play
3. Consider:
   - Different features (sentiment, news, fundamentals)
   - Longer prediction horizons (weekly returns instead of daily)
   - Different markets or asset classes
""")
elif roc_auc_score(y_test, gb_probs) < 0.55:
    print("""
📊 CONCLUSION: Weak but detectable signal
The best model achieves AUC between 0.52-0.55.
There IS some signal, but it's weak. Deep learning may help extract it.
Continue training with more epochs and hyperparameter tuning.
""")
else:
    print("""
✓ CONCLUSION: Meaningful signal detected!
The best model achieves AUC > 0.55.
Deep learning should be able to leverage this signal.
Focus on model architecture and training optimization.
""")
