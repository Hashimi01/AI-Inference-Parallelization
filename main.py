import torch
import torchvision.models as models
import time
import concurrent.futures
import matplotlib.pyplot as plt
import numpy as np
import os

# --- Configuration du Projet ---
# Nombre de tâches d'inférence à simuler
NOMBRE_INFERENCES = 100
# Nombre de threads à utiliser pour l'approche parallèle
NOMBRE_THREADS = 8
# Dimensions des données d'entrée (batch_size, channels, height, width)
DIM_INPUT = (1, 3, 224, 224)

def charger_modele():
    """
    Charge le modèle ResNet18 pré-entraîné de torchvision.
    Le modèle est mis en mode évaluation (.eval()).
    """
    try:
        print("[INFO] Chargement du modèle ResNet18 pré-entraîné...")
        # Utilisation des poids par défaut pour ResNet18
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.eval()
        # Désactiver le calcul des gradients pour économiser de la mémoire et du temps
        torch.set_grad_enabled(False)
        print("[INFO] Modèle chargé avec succès.")
        return model
    except Exception as e:
        print(f"[ERREUR] Lors du chargement du modèle : {e}")
        exit(1)

def generer_donnees_factices(nombre):
    """
    Génère une liste de tenseurs aléatoires pour simuler un lot de données.
    """
    print(f"[INFO] Génération de {nombre} échantillons de données factices...")
    return [torch.randn(*DIM_INPUT) for _ in range(nombre)]

def tache_inference(modele, data):
    """
    Effectue une seule passe d'inférence sur une donnée.
    """
    return modele(data)

def approche_sequentielle(modele, data_list):
    """
    Exécute les tâches d'inférence de manière séquentielle.
    """
    print("\n[PROCESS] Début de l'approche séquentielle...")
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
    Exécute les tâches d'inférence en utilisant un pool de threads.
    """
    print(f"\n[PROCESS] Début de l'approche Multi-thread ({num_threads} threads)...")
    start_time = time.perf_counter()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        # On soumet toutes les tâches au pool de threads
        futures = [executor.submit(tache_inference, modele, data) for data in data_list]
        # On attend que toutes les tâches soient terminées
        resultats = [f.result() for f in concurrent.futures.as_completed(futures)]
        
    end_time = time.perf_counter()
    duree = end_time - start_time
    print(f"[OK] Approche Multi-thread terminée en {duree:.4f} secondes.")
    return duree

def generer_graphique(temps_seq, temps_mt):
    """
    Génère et sauvegarde un graphique comparatif des performances.
    """
    labels = ['Séquentiel', f'Multi-thread ({NOMBRE_THREADS} threads)']
    temps = [temps_seq, temps_mt]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, temps, color=['#e74c3c', '#3498db'])
    
    plt.ylabel('Temps d\'exécution (secondes)')
    plt.title('Comparaison des performances : Séquentiel vs Multi-thread')
    
    # Ajouter les valeurs au-dessus des barres
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.4f}s', ha='center', va='bottom', fontweight='bold')
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Sauvegarde du graphique
    output_path = 'performance_graph.png'
    plt.savefig(output_path)
    print(f"\n[INFO] Graphique sauvegardé sous : {output_path}")
    plt.close()

def main():
    print("="*50)
    print("PROJET : PARALLÉLISATION D'INFÉRENCE D'IA")
    print("="*50)
    
    # 1. Préparation
    modele = charger_modele()
    donnees = generer_donnees_factices(NOMBRE_INFERENCES)
    
    # 2. Exécution Séquentielle
    temps_seq = approche_sequentielle(modele, donnees)
    
    # 3. Exécution Multi-thread
    temps_mt = approche_multithread(modele, donnees, NOMBRE_THREADS)
    
    # 4. Analyse des résultats
    speedup = temps_seq / temps_mt
    print("\n" + "="*30)
    print("RÉSUMÉ DES PERFORMANCES")
    print("="*30)
    print(f"Temps Séquentiel  : {temps_seq:.4f} s")
    print(f"Temps Multi-thread : {temps_mt:.4f} s")
    print(f"Accélération (Speedup) : {speedup:.2f}x")
    
    if speedup > 1:
        print(f"L'approche Multi-thread est {speedup:.2f} fois plus rapide.")
    else:
        print("L'approche Multi-thread n'a pas apporté d'amélioration (possible dû au GIL ou overhead).")
    
    # 5. Génération du graphique
    generer_graphique(temps_seq, temps_mt)
    print("\n[FIN] Fin du script.")

if __name__ == "__main__":
    main()
