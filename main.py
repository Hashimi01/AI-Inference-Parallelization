import torch
torch.set_num_threads(1)
import torchvision.models as models
import time
import concurrent.futures
import matplotlib.pyplot as plt
import os

# --- Configuration du Projet ---
# Nombre de tâches d'inférence à exécuter
NOMBRE_INFERENCES = 100
# Nombre de threads pour l'approche parallèle
NOMBRE_THREADS = 8
# Dimensions des données d'entrée (batch_size, channels, height, width)
DIM_INPUT = (1, 3, 224, 224)

def charger_modele():
    """
    Charge le modèle ResNet18 pré-entraîné.
    Le mode eval() désactive les couches de dropout et de batch normalization en mode entraînement.
    """
    try:
        print("[INFO] Chargement du modèle ResNet18 pré-entraîné...")
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.eval()
        # Désactiver le calcul des gradients pour l'inférence
        torch.set_grad_enabled(False)
        print(f"[INFO] Modèle chargé. PyTorch utilise {torch.get_num_threads()} threads internes.")
        return model
    except Exception as e:
        print(f"[ERREUR] Lors du chargement du modèle : {e}")
        exit(1)

def generer_donnees(nombre):
    """
    Génère des tenseurs aléatoires pour simuler des images d'entrée.
    """
    print(f"[INFO] Génération de {nombre} échantillons de données...")
    return [torch.randn(*DIM_INPUT) for _ in range(nombre)]

def tache_inference(modele, data):
    """
    Effectue une inférence sur une donnée avec le modèle.
    Cette fonction est appelée par chaque thread.
    """
    return modele(data)

def approche_sequentielle(modele, data_list):
    """
    Exécute les inférences de manière séquentielle (une par une).
    C'est notre référence de base (baseline).
    """
    print("\n[PROCESS] Exécution SÉQUENTIELLE (1 thread)...")
    start_time = time.perf_counter()
    
    resultats = []
    for data in data_list:
        res = tache_inference(modele, data)
        resultats.append(res)
    
    end_time = time.perf_counter()
    duree = end_time - start_time
    print(f"[OK] Approche séquentielle terminée en {duree:.4f} secondes.")
    return duree

def approche_multithread(modele, data_list, num_threads):
    """
    Exécute les inférences en parallèle avec un pool de THREADS.
    
    NOTE TECHNIQUE : 
    - Python a un verrou global (GIL) qui empêche l'exécution parallèle pure.
    - MAIS PyTorch libère le GIL pendant les calculs matriciels (C/C++).
    - Pour maximiser les gains, on utilise des BATCHES par thread.
    """
    print(f"\n[PROCESS] Exécution MULTI-THREAD ({num_threads} threads)...")
    start_time = time.perf_counter()
    
    # Diviser les données en lots (un par thread) pour réduire l'overhead
    batch_size = len(data_list) // num_threads
    batches = []
    for i in range(num_threads):
        start_idx = i * batch_size
        if i == num_threads - 1:
            batches.append(data_list[start_idx:])  # Dernier batch prend le reste
        else:
            batches.append(data_list[start_idx:start_idx + batch_size])
    
    def traiter_batch(batch):
        """Traite un batch de données dans un thread."""
        return [modele(data) for data in batch]
    
    # ThreadPoolExecutor : gestion automatique du pool de threads
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(traiter_batch, batch) for batch in batches]
        resultats = [f.result() for f in concurrent.futures.as_completed(futures)]
    
    end_time = time.perf_counter()
    duree = end_time - start_time
    print(f"[OK] Approche Multi-thread terminée en {duree:.4f} secondes.")
    return duree

def generer_graphique(temps_seq, temps_mt):
    """
    Génère un graphique comparatif des performances.
    """
    labels = ['Séquentiel\n(1 thread)', f'Multi-thread\n({NOMBRE_THREADS} threads)']
    temps = [temps_seq, temps_mt]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, temps, color=['#e74c3c', '#3498db'])
    
    plt.ylabel('Temps d\'exécution (secondes)')
    plt.title('Parallélisation d\'Inférence d\'IA avec des Threads')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f'{yval:.4f}s', 
                 ha='center', va='bottom', fontweight='bold')
    
    # Calcul et affichage du speedup
    speedup = temps_seq / temps_mt if temps_mt > 0 else 0
    plt.figtext(0.5, 0.01, f'Speedup : {speedup:.2f}x', ha='center', fontsize=12, 
                fontweight='bold', color='green' if speedup > 1 else 'red')
    
    # Nommage incrémental du fichier
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
    print("PROJET : PARALLÉLISATION D'INFÉRENCE D'IA AVEC DES THREADS")
    print("="*60)
    
    # 1. Préparation des ressources
    modele = charger_modele()
    donnees = generer_donnees(NOMBRE_INFERENCES)
    
    # 2. Exécution Séquentielle (baseline)
    temps_seq = approche_sequentielle(modele, donnees)
    
    # 3. Exécution Multi-thread
    temps_mt = approche_multithread(modele, donnees, NOMBRE_THREADS)
    
    # 4. Analyse des résultats
    speedup = temps_seq / temps_mt if temps_mt > 0 else 0
    
    print("\n" + "="*50)
    print("RÉSUMÉ DES PERFORMANCES")
    print("="*50)
    print(f"  Temps Séquentiel   : {temps_seq:.4f} s")
    print(f"  Temps Multi-thread : {temps_mt:.4f} s")
    print(f"  Accélération       : {speedup:.2f}x")
    
    if speedup > 1.2:
        print("\n✅ SUCCÈS : Le multi-threading accélère l'inférence!")
    elif speedup > 0.9:
        print("\n⚠️  Performances similaires (overhead du threading).")
    else:
        print("\n❌ Le threading ralentit l'exécution (GIL ou contention des ressources).")
    
    # 5. Génération du graphique
    generer_graphique(temps_seq, temps_mt)
    print("\n[FIN] Exécution terminée.")

if __name__ == "__main__":
    main()
