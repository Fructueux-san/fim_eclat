"""
Implémentation améliorée de l'algorithme ECLAT
pour l'extraction de motifs fréquents
"""

from collections import defaultdict
from typing import Dict, Set, List, Tuple
import time

class ECLATMiner:
    def __init__(self, min_support: float = 0.2):
        """
        Initialise le mineur ECLAT
        
        Args:
            min_support: Support minimum (en pourcentage, ex: 0.2 pour 20%)
        """
        self.min_support_ratio = min_support
        self.min_support_count = 0
        self.nb_transactions = 0
        self.frequent_itemsets = {}
        
    def load_dataset(self, filepath: str) -> Dict[str, Set[int]]:
        """
        Charge un dataset et génère les tidsets (format vertical)
        
        Args:
            filepath: Chemin vers le fichier dataset
            
        Returns:
            Dictionnaire {item -> set de transaction IDs}
        """
        item_tidset = defaultdict(set)
        
        with open(filepath, 'r') as file:
            for tid, line in enumerate(file):
                # Nettoie la ligne et split par espace
                items = line.strip().split()
                for item in items:
                    if item:  # Ignore les chaînes vides
                        item_tidset[item].add(tid)
        
        self.nb_transactions = tid + 1  # Nombre total de transactions
        self.min_support_count = int(self.nb_transactions * self.min_support_ratio)
        
        print(f"Dataset chargé: {self.nb_transactions} transactions")
        print(f"Support minimum: {self.min_support_count} ({self.min_support_ratio*100}%)")
        print(f"Nombre d'items distincts: {len(item_tidset)}")
        
        return item_tidset
    
    def eclat_recursive(self, prefix: List[str], items: List[Tuple[str, Set[int]]]):
        """
        Fonction récursive ECLAT
        
        Args:
            prefix: Préfixe de l'itemset courant
            items: Liste de tuples (item, tidset) à explorer
        """
        for i, (item, tidset) in enumerate(items):
            support = len(tidset)
            
            # Si le support est suffisant
            if support >= self.min_support_count:
                # Créer le nouvel itemset
                new_itemset = prefix + [item]
                self.frequent_itemsets[frozenset(new_itemset)] = support
                
                # Construire le suffix en intersectant avec les items suivants
                suffix = []
                for j in range(i + 1, len(items)):
                    other_item, other_tidset = items[j]
                    intersection = tidset & other_tidset
                    
                    if len(intersection) >= self.min_support_count:
                        suffix.append((other_item, intersection))
                
                # Appel récursif si le suffix n'est pas vide
                if suffix:
                    self.eclat_recursive(new_itemset, suffix)
    
    def mine(self, filepath: str) -> Dict[frozenset, int]:
        """
        Lance l'extraction des motifs fréquents
        
        Args:
            filepath: Chemin vers le dataset
            
        Returns:
            Dictionnaire {itemset -> support}
        """
        print(f"\n{'='*60}")
        print(f"Démarrage de l'algorithme ECLAT")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        
        # Charger le dataset
        item_tidset = self.load_dataset(filepath)
        
        # Filtrer les items qui ne respectent pas le support minimum
        frequent_1_itemsets = [
            (item, tidset) 
            for item, tidset in item_tidset.items() 
            if len(tidset) >= self.min_support_count
        ]
        
        print(f"Items fréquents (1-itemsets): {len(frequent_1_itemsets)}")
        
        # Trier par support croissant (optimisation)
        frequent_1_itemsets.sort(key=lambda x: len(x[1]))
        
        # Initialiser le dictionnaire des itemsets fréquents
        self.frequent_itemsets = {}
        
        # Ajouter les 1-itemsets fréquents
        for item, tidset in frequent_1_itemsets:
            self.frequent_itemsets[frozenset([item])] = len(tidset)
        
        # Lancer ECLAT
        print(f"\nExtraction des motifs fréquents en cours...")
        self.eclat_recursive([], frequent_1_itemsets)
        
        elapsed_time = time.time() - start_time
        
        print(f"\n{'='*60}")
        print(f"Extraction terminée en {elapsed_time:.2f} secondes")
        print(f"Nombre total d'itemsets fréquents: {len(self.frequent_itemsets)}")
        print(f"{'='*60}\n")
        
        return self.frequent_itemsets
    
    def print_results(self, max_itemsets: int = 50, sort_by_size: bool = True):
        """
        Affiche les résultats
        
        Args:
            max_itemsets: Nombre maximum d'itemsets à afficher
            sort_by_size: Si True, trie par taille d'itemset puis par support
        """
        if not self.frequent_itemsets:
            print("Aucun itemset fréquent trouvé.")
            return
        
        # Statistiques par taille
        itemsets_by_size = defaultdict(int)
        for itemset in self.frequent_itemsets.keys():
            itemsets_by_size[len(itemset)] += 1
        
        print("Distribution des itemsets par taille:")
        for size in sorted(itemsets_by_size.keys()):
            print(f"  {size}-itemsets: {itemsets_by_size[size]}")
        
        print(f"\n{'='*60}")
        print(f"Top {max_itemsets} itemsets fréquents:")
        print(f"{'='*60}\n")
        
        # Trier les itemsets
        if sort_by_size:
            sorted_itemsets = sorted(
                self.frequent_itemsets.items(),
                key=lambda x: (-len(x[0]), -x[1], sorted(list(x[0])))
            )
        else:
            sorted_itemsets = sorted(
                self.frequent_itemsets.items(),
                key=lambda x: -x[1]
            )
        
        # Afficher les résultats
        for i, (itemset, support) in enumerate(sorted_itemsets[:max_itemsets], 1):
            support_pct = (support / self.nb_transactions) * 100
            items_str = '{' + ', '.join(sorted(list(itemset))) + '}'
            print(f"{i:3d}. {items_str:<40} => support: {support:5d} ({support_pct:5.2f}%)")
        
        if len(self.frequent_itemsets) > max_itemsets:
            print(f"\n... et {len(self.frequent_itemsets) - max_itemsets} autres itemsets")


# ============================================================================
# Exemple d'utilisation
# ============================================================================

if __name__ == "__main__":
    # Test avec le dataset chess
    print("=" * 80)
    print("TEST AVEC LE DATASET CHESS")
    print("=" * 80)
    
    miner_chess = ECLATMiner(min_support=0.2)
    frequent_itemsets_chess = miner_chess.mine("datasets/chess.dat")
    miner_chess.print_results(max_itemsets=30)
    
    # Test avec le dataset retail (décommentez si vous avez le fichier)
    # print("\n\n")
    # print("=" * 80)
    # print("TEST AVEC LE DATASET RETAIL")
    # print("=" * 80)
    # 
    # miner_retail = ECLATMiner(min_support=0.01)  # Support plus faible pour retail
    # frequent_itemsets_retail = miner_retail.mine("datasets/retail.dat")
    # miner_retail.print_results(max_itemsets=30)
