# Natural Language Sentiment Classification with Deep Learning
Assignment 2 (Sentiment Classification with Deep Learning) for the course "Natural Language Processing 1" at the University of Amsterdam

### Open ideas
- [x] Train standard LSTMs with attention modules 
  - _Result_: no (significant) improvement
  - _Problem_: too few data points to learn deep context / ignoring subtree annotation!
  - [ ] Train LSTM on short sequences/subtrees as well 
- [x] Training Tree LSTMs with subtrees
  - [x] Start with small trees/single words, and increase average tree height over time
  - [x] Record loss for subtrees, and take those with higher probability that have higher loss (over last _N_ iterations)
    - This list is shared across the same words and different trees
  - _Problem_: strongly imbalanced dataset, especially for small tree => MAF (see below)
  - [ ] Moving Average Filters for Loss Weighting
  - Which heuristic for determining the probabilities of tree heights at each iteration?
- [ ] Use context-free grammar rules to distinguish between different binary tree combinations (adjective and noun, verb and object, ...) and train different weights (might help)
