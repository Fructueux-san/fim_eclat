# Implémentation de eclat trouvé sur geeksforgeeks
# link : https://www.geeksforgeeks.org/machine-learning/ml-eclat-algorithm/

from collections import defaultdict
from itertools import combinations
import math

transactions = {
    "T1": ["Bread", "Butter", "Jam"],
    "T2": ["Butter", "Coke"],
    "T3": ["Butter", "Milk"],
    "T4": ["Bread", "Butter", "Coke"],
    "T5": ["Bread", "Milk"],
    "T6": ["Butter", "Milk"],
    "T7": ["Bread", "Milk"],
    "T8": ["Bread", "Butter", "Milk", "Jam"],
    "T9": ["Bread", "Butter", "Milk"]
}



# Transformation de la base en db verticale

def generate_tidsets(transactions):
    item_tidset = defaultdict(set)

    for tid, items in transactions.items():
        for item in items:
            item_tidset[item].add(tid)
    return item_tidset

def generate_tidsets_from_dataset(filepath:str):

    item_tidset = defaultdict(set)

    with open(filepath, 'r') as file:
        transactions = file.readlines()
        # Les items sont séparés par des spaces dans le fichier
        for tid, tset_line in enumerate(transactions):
            for item in tset_line.split(' '):
                item_tidset[item].add(tid)
    return item_tidset
        
        
def get_dataset_length(filepath):
    l = 0
    with open(filepath, 'r') as file:
        l = len(file.readlines())
    return l


def eclat(prefix, items, min_support, frequent_itemsets):
    while items:
        item, tidset = items.pop()
        support = len(tidset)

        if support >= min_support:
            new_itemset = prefix + [item]
            frequent_itemsets[frozenset(new_itemset)] = support
            suffix = []

            for other_item, other_tidset in items:
                intersection = tidset & other_tidset
                if len(intersection) >= min_support:
                    suffix.append((other_item, intersection))
            suffix = sorted(suffix, key=lambda x: len(x[1]))
            eclat(new_itemset, suffix, min_support, frequent_itemsets)


item_tidset = generate_tidsets_from_dataset("datasets/chess.dat")

# item_tidset = generate_tidsets(transactions)
#
# for item, tidset in item_tidset.items():
#     print(item, ':', sorted(tidset), '\n')
#
# tri croissant (par ordre du nombre de transactions)

# 20 %
min_support = math.floor(get_dataset_length("datasets/chess.dat") * 0.2)

items = sorted(item_tidset.items(), key=lambda x: len(x[1]))
# print('\n', items)

print(f" \nECLAT start with support at {min_support} \n")

frequent_itemsets = {}
eclat([], items, min_support, frequent_itemsets)

print("Frequent itemsets (as list) -> support count")
for itemset, support in sorted(frequent_itemsets.items(), key=lambda x: (-len(x[0]), -x[1], sorted(list(x[0])))):
    print(list(itemset), '=>', support)

