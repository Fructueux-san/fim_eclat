#! /bin/bash
# Tester différents supports
for support in 0.98 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1; do
    time python eclat_improved.py datasets/chess.dat $support
done
