import torch
import torchvision.models as models
import time
import matplotlib.pyplot as plt
import os

# --- Configuration du Projet ---
NOMBRE_INFERENCES = 100
DIM_INPUT = (3, 224, 224)  # Sans batch dimension
BATCH_SIZES = [1, 10, 25, 50, 100]  # Différentes tailles de batch à tester

def charger_modele():
    """
    Charge le modèle ResNet18 pré-entraîné.
    PyTorch utilise automatiquement le multithreading interne pour les opérations matricielles.
    """
    try:
        print("[INFO] Chargement du modèle ResNet18 pré-entraîné...")
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.eval()
        torch.set_grad_enabled(False)
        
        # Permettre à PyTorch d'utiliser tous les cœurs disponibles
        num_threads = torch.get_num_threads()
        print(f"[INFO] PyTorch utilise {num_threads} threads pour les opérations.")
        
        return model
    except Exception as e:
        print(f"[ERREUR] Lors du chargement du modèle : {e}")
        exit(1)

def approche_sequentielle(modele, data_list):
    """
    Traitement séquentiel : une image à la fois (baseline lent).
    """
    print("\n[PROCESS] Approche Séquentielle (1 image à la fois)...")
    start_time = time.perf_counter()
    
    _ = [modele(data.unsqueeze(0)) for data in data_list]  # unsqueeze pour ajouter batch dim
    
    end_time = time.perf_counter()
    return end_time - start_time

def approche_batch(modele, data_list, batch_size):
    """
    Traitement par BATCH : grouper les images et les traiter ensemble.
    C'est LA méthode optimale pour l'inférence sur CPU/GPU.
    PyTorch parallélise automatiquement les calculs matriciels sur plusieurs cœurs.
    """
    print(f"\n[PROCESS] Approche Batch (batch_size={batch_size})...")
    start_time = time.perf_counter()
    
    # Créer les batches
    for i in range(0, len(data_list), batch_size):
        batch = torch.stack(data_list[i:i + batch_size])  # Combine en un seul tenseur
        _ = modele(batch)  # Inférence sur tout le batch en une fois
    
    end_time = time.perf_counter()
    return end_time - start_time

def generer_graphique(resultats):
    """
    Génère un histogramme comparatif des performances.
    """
    labels = list(resultats.keys())
    temps = list(resultats.values())
    
    # Couleurs : rouge pour séquentiel, dégradé de vert pour les batches
    colors = ['#e74c3c'] + ['#2ecc71', '#27ae60', '#1abc9c', '#16a085'][:len(labels)-1]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(labels, temps, color=colors)
    
    plt.ylabel('Temps d\'exécution (secondes)')
    plt.xlabel('Méthode')
    plt.title('Performance : Séquentiel vs Batched Inference (Vraie Optimisation)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.4f}s', 
                 ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # Nommage incrémental
    base_name = 'performance_graph'
    extension = '.png'
    counter = 0
    nom_fichier = f"{base_name}{extension}"
    
    while os.path.exists(nom_fichier):
        counter += 1
        nom_fichier = f"{base_name}_{counter}{extension}"

    plt.tight_layout()
    plt.savefig(nom_fichier)
    print(f"\n[INFO] Graphique sauvegardé : {nom_fichier}")

def main():
    print("="*60)
    print("OPTIMISATION D'INFÉRENCE : BATCHED INFERENCE")
    print("="*60)
    print("\nNOTE : Multiprocessing n'est PAS optimal pour cette tâche.")
    print("       PyTorch utilise le multithreading interne pour les calculs matriciels.")
    print("       La vraie optimisation est le BATCHING des données.\n")

    # 1. Préparation
    modele = charger_modele()
    print(f"[INFO] Génération de {NOMBRE_INFERENCES} images factices...")
    donnees = [torch.randn(*DIM_INPUT) for _ in range(NOMBRE_INFERENCES)]

    resultats = {}

    # 2. Test Séquentiel (baseline)
    t_seq = approche_sequentielle(modele, donnees)
    resultats['Séquentiel\n(batch=1)'] = t_seq
    print(f"[OK] Séquentiel terminé en : {t_seq:.4f}s")

    # 3. Tests avec différentes tailles de batch
    for batch_size in BATCH_SIZES[1:]:  # Skip batch_size=1 (c'est le séquentiel)
        t_batch = approche_batch(modele, donnees, batch_size)
        resultats[f'Batch\n(size={batch_size})'] = t_batch
        print(f"[OK] Batch size={batch_size} terminé en : {t_batch:.4f}s")

    # 4. Analyse
    print("\n" + "="*50)
    print("RÉSUMÉ DES PERFORMANCES")
    print("="*50)
    
    t_best = min(resultats.values())
    best_method = [k for k, v in resultats.items() if v == t_best][0]
    speedup = t_seq / t_best
    
    print(f"Temps Séquentiel     : {t_seq:.4f}s")
    print(f"Meilleur temps       : {t_best:.4f}s ({best_method.replace(chr(10), ' ')})")
    print(f"Accélération (Speedup): {speedup:.2f}x")
    
    if speedup > 1.5:
        print("\n✅ SUCCÈS : Le batching offre une accélération significative!")
    else:
        print("\n⚠️  Le gain est modéré. C'est normal pour les petits modèles sur CPU.")

    # 5. Visualisation
    generer_graphique(resultats)
    print("\n[FIN] Analyse terminée.")

if __name__ == "__main__":
    main()
