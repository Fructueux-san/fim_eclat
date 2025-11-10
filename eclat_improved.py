"""
Implémentation optimisée de l'algorithme ECLAT
Focus sur la performance et la consommation mémoire réduite
"""

from collections import defaultdict
from typing import Dict, Set, List, Tuple
import time
import argparse
import sys
import gc

class ECLATMiner:
    def __init__(self, min_support: float = 0.2, verbose: bool = True):
        """
        Initialise le mineur ECLAT
        
        Args:
            min_support: Support minimum (en pourcentage, ex: 0.2 pour 20%)
            verbose: Afficher les informations de progression
        """
        self.min_support_ratio = min_support
        self.min_support_count = 0
        self.nb_transactions = 0
        self.nb_frequent_itemsets = 0
        self.verbose = verbose
        
    def load_dataset(self, filepath: str) -> Dict[str, Set[int]]:
        """
        Charge un dataset et génère les tidsets (format vertical)
        Optimisé pour réduire la consommation mémoire
        
        Args:
            filepath: Chemin vers le fichier dataset
            
        Returns:
            Dictionnaire {item -> set de transaction IDs}
        """
        item_tidset = defaultdict(set)
        
        try:
            with open(filepath, 'r') as file:
                for tid, line in enumerate(file):
                    items = line.strip().split()
                    for item in items:
                        if item:
                            item_tidset[item].add(tid)
            
            self.nb_transactions = tid + 1
            self.min_support_count = int(self.nb_transactions * self.min_support_ratio)
            
            if self.verbose:
                print(f"Dataset: {filepath}")
                print(f"Transactions: {self.nb_transactions}")
                print(f"Items distincts: {len(item_tidset)}")
                print(f"Support minimum: {self.min_support_count} ({self.min_support_ratio*100:.1f}%)")
                
        except FileNotFoundError:
            print(f"ERREUR: Fichier '{filepath}' introuvable", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"ERREUR lors de la lecture du fichier: {e}", file=sys.stderr)
            sys.exit(1)
            
        return item_tidset
    
    def eclat_recursive(self, prefix: Tuple[str, ...], items: List[Tuple[str, Set[int]]]):
        """
        Fonction récursive ECLAT optimisée pour la mémoire
        - Utilise des tuples au lieu de listes pour le préfixe (moins de mémoire)
        - Ne stocke pas tous les itemsets, juste le compteur
        - Libère la mémoire des tidsets dès que possible
        
        Args:
            prefix: Préfixe de l'itemset courant (tuple immutable)
            items: Liste de tuples (item, tidset) à explorer
        """
        for i in range(len(items)):
            item, tidset = items[i]
            support = len(tidset)
            
            if support >= self.min_support_count:
                self.nb_frequent_itemsets += 1
                
                # Construire le suffix avec intersection
                suffix = []
                for j in range(i + 1, len(items)):
                    other_item, other_tidset = items[j]
                    intersection = tidset & other_tidset
                    
                    if len(intersection) >= self.min_support_count:
                        suffix.append((other_item, intersection))
                
                # Appel récursif si suffix non vide
                if suffix:
                    new_prefix = prefix + (item,)
                    self.eclat_recursive(new_prefix, suffix)
                    
                # Libérer la mémoire du suffix
                del suffix
    
    def mine(self, filepath: str) -> int:
        """
        Lance l'extraction des motifs fréquents
        Retourne uniquement le nombre d'itemsets (pas les itemsets eux-mêmes)
        
        Args:
            filepath: Chemin vers le dataset
            
        Returns:
            Nombre d'itemsets fréquents trouvés
        """
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"Algorithme ECLAT - Benchmark de performance")
            print(f"{'='*70}\n")
        
        start_time = time.time()
        
        # Charger le dataset
        item_tidset = self.load_dataset(filepath)
        load_time = time.time() - start_time
        
        # Filtrer les items qui ne respectent pas le support minimum
        frequent_1_itemsets = [
            (item, tidset) 
            for item, tidset in item_tidset.items() 
            if len(tidset) >= self.min_support_count
        ]
        
        # Libérer la mémoire de item_tidset
        del item_tidset
        gc.collect()
        
        if self.verbose:
            print(f"Items fréquents (1-itemsets): {len(frequent_1_itemsets)}")
            print(f"Temps de chargement: {load_time:.3f}s\n")
            print(f"Extraction en cours...")
        
        # Trier par support croissant (optimisation)
        frequent_1_itemsets.sort(key=lambda x: len(x[1]))
        
        # Compter les 1-itemsets
        self.nb_frequent_itemsets = len(frequent_1_itemsets)
        
        # Lancer ECLAT
        mining_start = time.time()
        self.eclat_recursive((), frequent_1_itemsets)
        mining_time = time.time() - mining_start
        
        total_time = time.time() - start_time
        
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"RÉSULTATS")
            print(f"{'='*70}")
            print(f"Itemsets fréquents trouvés: {self.nb_frequent_itemsets}")
            print(f"Temps de chargement: {load_time:.3f}s")
            print(f"Temps d'extraction: {mining_time:.3f}s")
            print(f"Temps total: {total_time:.3f}s")
            print(f"{'='*70}\n")
        else:
            # Mode compact pour benchmarking
            print(f"{filepath},{self.min_support_ratio},{self.nb_frequent_itemsets},{total_time:.3f}")
        
        return self.nb_frequent_itemsets


def main():
    """
    Point d'entrée principal avec arguments en ligne de commande
    """
    parser = argparse.ArgumentParser(
        description='ECLAT Algorithm - Frequent Itemset Mining',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  python eclat.py datasets/chess.dat 0.2
  python eclat.py datasets/retail.dat 0.01
  python eclat.py datasets/chess.dat 0.3 --quiet
  python eclat.py datasets/retail.dat 0.005 -q

Le mode --quiet affiche: dataset,support,nb_itemsets,temps(s)
        """
    )
    
    parser.add_argument(
        'dataset',
        type=str,
        help='Chemin vers le fichier dataset'
    )
    
    parser.add_argument(
        'min_support',
        type=float,
        help='Support minimum (entre 0 et 1, ex: 0.2 pour 20%%)'
    )
    
    parser.add_argument(
        '-q', '--quiet',
        action='store_true',
        help='Mode silencieux (affiche uniquement les résultats en CSV)'
    )
    
    parser.add_argument(
        '--timeout',
        type=int,
        default=1800,
        help='Timeout en secondes (défaut: 1800s = 30min)'
    )
    
    args = parser.parse_args()
    
    # Validation du support
    if not 0 < args.min_support <= 1:
        print("ERREUR: Le support minimum doit être entre 0 et 1", file=sys.stderr)
        sys.exit(1)
    
    # Header CSV pour mode quiet
    if args.quiet:
        print("dataset,min_support,nb_itemsets,time_seconds")
    
    # Exécuter ECLAT
    try:
        miner = ECLATMiner(min_support=args.min_support, verbose=not args.quiet)
        miner.mine(args.dataset)
        
    except KeyboardInterrupt:
        print("\n\nInterruption par l'utilisateur", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\nERREUR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
