import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics.pairwise import rbf_kernel
from qiskit_algorithms.utils import algorithm_globals
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_ibm_runtime import QiskitRuntimeService

# Set random seed for reproducibility
algorithm_globals.random_seed = 123

# Step 1: Setup IBM Quantum backend (no fallback to ensure 9-min quota usage)
print("Setting up IBM Quantum backend...")
try:
    service = QiskitRuntimeService(name='Paul')  # Your IBM Quantum account
    backend = service.backend('ibm_torino')
    print(f"✓ IBM Torino backend selected: {backend}")
    quantum_backend = True
    backend_name = "IBM Torino"
except Exception as e:
    raise Exception(f"IBM backend failed: {e}. Please check token, update qiskit-ibm-runtime, or wait for quota reset on Oct 1, 2025.")

# Step 2: Load and preprocess the large dataset
print("\nPreprocessing large heart disease dataset...")
data_large = pd.read_csv('Heart_Disease_and_Hospitals.csv')
print(f"Dataset shape: {data_large.shape}")

# Select numerical features and target
numerical_features = ['age', 'blood_pressure', 'cholesterol', 'bmi', 'glucose_level']
X = data_large[numerical_features].values
y = data_large['heart_disease'].values

print(f"Selected features: {numerical_features}")
print(f"Features shape: {X.shape}")
print(f"Overall class distribution: {np.bincount(y)} ({np.mean(y):.1%} disease)")

# Handle missing values (impute with column mean)
X = np.nan_to_num(X, nan=np.nanmean(X, axis=0))

# Step 3: Stratified subsampling for balanced classes
print("\nUsing STRATIFIED SAMPLING for balanced classes...")
X_small, _, y_small, _ = train_test_split(X, y, train_size=200, random_state=456, stratify=y)  # Fits in 9 min

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_small)

# Split data (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_small, test_size=0.2, random_state=456, stratify=y_small
)

print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")
print(f"Training class distribution: {np.bincount(y_train)} ({np.mean(y_train):.1%} disease)")
print(f"Test class distribution: {np.bincount(y_test)} ({np.mean(y_test):.1%} disease)")

# Step 4: Set up quantum feature map for 2 features (age, blood_pressure)
print("\nSetting up quantum feature map...")
feature_map = ZZFeatureMap(feature_dimension=2, reps=2, entanglement='linear')

# Step 5: Create quantum kernel
print("Creating quantum kernel...")
quantum_kernel = FidelityQuantumKernel(feature_map=feature_map)
print("Quantum kernel created successfully!")

# Step 6: Compute quantum and classical kernel matrices
print("\nComputing hybrid quantum-classical kernel...")
# Quantum kernel for age, blood_pressure (indices 0, 1)
quantum_train = quantum_kernel.evaluate(X_train[:, [0, 1]])
quantum_test_train = quantum_kernel.evaluate(X_test[:, [0, 1]], X_train[:, [0, 1]])

# Classical RBF kernel for cholesterol, bmi, glucose_level (indices 2, 3, 4)
classical_train = rbf_kernel(X_train[:, [2, 3, 4]])
classical_test_train = rbf_kernel(X_test[:, [2, 3, 4]], X_train[:, [2, 3, 4]])

# Combine kernels (weighted: 70% classical, 30% quantum)
hybrid_train = 0.7 * classical_train + 0.3 * quantum_train
hybrid_test_train = 0.7 * classical_test_train + 0.3 * quantum_test_train

# Step 7: Train hybrid SVM with combined kernel
print("Training Hybrid Quantum-Classical SVM...")
svc = SVC(kernel='precomputed')
svc.fit(hybrid_train, y_train)

# Step 8: Evaluate
score = svc.score(hybrid_test_train, y_test)
print(f"Hybrid Quantum-Classical SVC Test Accuracy: {score:.2f}")

# Step 9: Predict on a sample
sample = X_test[:1]
sample_quantum = quantum_kernel.evaluate(sample[:, [0, 1]], X_train[:, [0, 1]])
sample_classical = rbf_kernel(sample[:, [2, 3, 4]], X_train[:, [2, 3, 4]])
sample_hybrid = 0.7 * sample_classical + 0.3 * sample_quantum
prediction = svc.predict(sample_hybrid)
print(f"Sample Prediction: {prediction[0]} (0: No disease, 1: Disease)")

# Step 10: Classical comparison (RBF kernel on all features)
print("\n" + "="*50)
print("CLASSICAL SVM COMPARISON")
print("="*50)
classical_svc = SVC(kernel='rbf')
classical_svc.fit(X_train, y_train)
classical_score = classical_svc.score(X_test, y_test)
print(f"Classical SVC Test Accuracy: {classical_score:.2f}")

print(f"\nHybrid vs Classical: {score:.3f} vs {classical_score:.3f}")
if score > classical_score:
    print("🎉 Hybrid Quantum-Classical SVM outperformed classical!")
else:
    print("Classical SVM performed better")

# Step 11: Full dataset classical benchmark
print("\n" + "="*50)
print("CLASSICAL BENCHMARK ON FULL DATASET")
print("="*50)
X_full_scaled = scaler.fit_transform(X)
X_full_train, X_full_test, y_full_train, y_full_test = train_test_split(
    X_full_scaled, y, test_size=0.2, random_state=456, stratify=y
)
full_classical_svc = SVC(kernel='rbf')
full_classical_svc.fit(X_full_train, y_full_train)
full_score = full_classical_svc.score(X_full_test, y_full_test)
print(f"Classical SVM on full 10,000 samples: {full_score:.3f}")

# Step 12: Results
print("\n" + "="*50)
print("HYBRID QUANTUM HEART DISEASE PREDICTION")
print("="*50)
print(f"Backend: {backend_name}")
print(f"Balanced training: {np.bincount(y_train)} samples with {X_train.shape[1]} features")
print(f"Hybrid Quantum-Classical Test accuracy: {score:.3f}")
print(f"Classical Test accuracy: {classical_score:.3f}")
print(f"Full dataset Classical accuracy: {full_score:.3f}")
print(f"Sample prediction: {'HEART DISEASE' if prediction[0] == 1 else 'NO HEART DISEASE'}")
print(f"Quantum advantage: {score - classical_score:+.3f}")
