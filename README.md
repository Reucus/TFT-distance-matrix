# TFT-doodles

Currently using data from Riot's Teamfight Tactics, Set 6: Gizmos & Gadgets

Datasets are one-hot-encoded with origins/classes as traits
Distance matrix is calculated with these traits

Notes:
- This doesn't take into account the number of units required to activate a trait
- Jayce/Yuumi/Tahm Kench each have a unique trait that activates with only one unit (themselves) (Transformer/Cuddly/Glutton) — the distance matrix is unweighted and doesn't take this into account



This project is licensed under CC-BY-NC-SA.
