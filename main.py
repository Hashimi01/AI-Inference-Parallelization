import torch
import torchvision.models as models
import time
import concurrent.futures
import matplotlib.pyplot as plt
import os

# --- Configuration du Projet ---
# IMPORTANT : Assurez-vous d'avoir alloué suffisamment de cœurs (ex: 8) à votre environnement
NOMBRE_INFERENCES = 100
NOMBRE_PROCESSUS = 8 
DIM_INPUT = (1, 3, 224, 224)

# Variable globale pour stocker le modèle dans chaque processus (évite la sérialisation inutile)
model_worker = None

def init_worker():
    """
    Fonction d'initialisation exécutée une fois au démarrage de chaque processus.
    Charge le modèle en mémoire locale du processus pour éviter le surcoût de transfert (Pickling).
    """
    global model_worker
    
    # INDISPENSABLE : Empêche PyTorch d'utiliser le multithreading interne,
    # ce qui entrerait en conflit avec notre multiprocessing et réduirait les performances.
    torch.set_num_threads(1)
    
    try:
        # Chargement du modèle (poids par défaut)
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.eval()
        # Désactivation des gradients pour l'inférence seule
        torch.set_grad_enabled(False)
        model_worker = model
    except Exception as e:
        print(f"[ERREUR] Échec de l'initialisation du worker : {e}")
        exit(1)

def tache_inference_batch(data_batch):
    """
    Exécute l'inférence sur un BATCH de données (au lieu d'une seule).
    CRITIQUE : Réduire le nombre de transferts inter-processus en traitant par lots.
    """
    global model_worker
    # Traite chaque élément du batch
    return [model_worker(data) for data in data_batch]

def approche_sequentielle(data_list):
    """
    Exécution séquentielle classique pour servir de référence (baseline).
    """
    print("\n[PROCESS] Début de l'exécution séquentielle...")
    
    # Chargement d'un modèle local pour le test séquentiel
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.eval()
    torch.set_grad_enabled(False)
    
    start_time = time.perf_counter()
    
    _ = [model(data) for data in data_list]
    
    end_time = time.perf_counter()
    return end_time - start_time

def approche_multiprocess_optimisee(data_list, num_workers):
    """
    Version ULTRA-OPTIMISÉE utilisant le traitement par BATCH.
    Au lieu d'envoyer 100 tâches individuelles (overhead énorme), on divise en 8 lots.
    Cela réduit drastiquement le coût de communication inter-processus.
    """
    print(f"\n[PROCESS] Début du parallélisme optimisé ({num_workers} processus)...")
    start_time = time.perf_counter()
    
    # Division intelligente des données en batches (un par worker)
    batch_size = len(data_list) // num_workers
    batches = []
    
    for i in range(num_workers):
        start_idx = i * batch_size
        # Le dernier batch prend tous les éléments restants
        if i == num_workers - 1:
            batches.append(data_list[start_idx:])
        else:
            batches.append(data_list[start_idx:start_idx + batch_size])

    # ProcessPoolExecutor avec initializer : charge le modèle AVANT de traiter les tâches
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers, initializer=init_worker) as executor:
        # On passe des BATCHES entiers, pas des éléments individuels (réduction majeure de l'overhead)
        futures = [executor.submit(tache_inference_batch, batch) for batch in batches]
        
        # Attente et récupération des résultats
        resultats = [f.result() for f in concurrent.futures.as_completed(futures)]

    end_time = time.perf_counter()
    return end_time - start_time

def generer_graphique(temps_seq, temps_mp):
    """
    Génère un histogramme comparatif des temps d'exécution.
    """
    labels = ['Séquentiel', f'Multiprocess ({NOMBRE_PROCESSUS} cœurs)']
    temps = [temps_seq, temps_mp]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, temps, color=['#e74c3c', '#2ecc71'])
    
    plt.ylabel('Temps d\'exécution (secondes)')
    plt.title('Comparaison de Performance : Séquentiel vs Multiprocessing Optimisé')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.4f}s', 
                 ha='center', va='bottom', fontweight='bold')
    
    # Logique pour le nommage incrémental
    base_name = 'performance_graph'
    extension = '.png'
    counter = 0
    nom_fichier = f"{base_name}{extension}"
    
    while os.path.exists(nom_fichier):
        counter += 1
        nom_fichier = f"{base_name}_{counter}{extension}"

    plt.savefig(nom_fichier)
    print(f"\n[INFO] Graphique sauvegardé : {nom_fichier}")

def main():
    print("="*60)
    print("SYSTÈME D'INFÉRENCE PARALLÈLE : OPTIMISATION MAXIMALE")
    print("="*60)

    # 1. Préparation des données (Simulation)
    print(f"[INFO] Génération de {NOMBRE_INFERENCES} entrées factices...")
    donnees = [torch.randn(*DIM_INPUT) for _ in range(NOMBRE_INFERENCES)]

    # 2. Test Séquentiel
    t_seq = approche_sequentielle(donnees)
    print(f"[OK] Séquentiel terminé en : {t_seq:.4f}s")

    # 3. Test Multiprocess Optimisé
    t_mp = approche_multiprocess_optimisee(donnees, NOMBRE_PROCESSUS)
    print(f"[OK] Multiprocess Optimisé terminé en : {t_mp:.4f}s")

    # 4. Analyse des gains
    if t_mp > 0:
        speedup = t_seq / t_mp
        print("\n" + "="*30)
        print(f"ANALYSE : Speedup de {speedup:.2f}x")
        
        if speedup > 1.5:
            print("RÉSULTAT : Excellente accélération. L'optimisation est réussie.")
        else:
            print("ATTENTION : Gain modéré. Vérifiez le nombre de cœurs physiques disponibles.")
    else:
        print("[ERREUR] Temps d'exécution nul.")

    # 5. Visualisation
    generer_graphique(t_seq, t_mp)

if __name__ == "__main__":
    # Protection nécessaire pour Windows/Linux lors de la création de sous-processus
    main()
