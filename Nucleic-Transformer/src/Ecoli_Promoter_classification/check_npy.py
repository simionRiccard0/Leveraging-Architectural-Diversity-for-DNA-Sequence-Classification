import numpy as np

# Carica i file
logits = np.load('independent_test_results_ensemble_logits.npy')
gt = np.load('independent_test_results_ground_truths.npy')

print("=== ENSEMBLE LOGITS ===")
print("Dimensione (Shape):", logits.shape)
print("Primi 5 campioni (Logit Classe 0, Logit Classe 1):\n", logits[:5])

print("\n=== GROUND TRUTHS ===")
print("Dimensione (Shape):", gt.shape)
print("Prime 5 etichette reali:", gt[:5])
