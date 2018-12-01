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
  - [x] Strong attention module/mechanism in Tree LSTMs
- [ ] Use context-free grammar rules to distinguish between different binary tree combinations (adjective and noun, verb and object, ...) and train different weights (might help) <= Need analysis of dataset first!
- [x] More detailed analysis during evaluation:
  - [x] How do we perform depending on the class?
  - [x] How do we perform depending on height fo the tree (and class)?
  - [x] Print out (small) examples during eval to understand prediction behaviour
  - [x] Print out the hardest/easiest training examples
- [ ] More detailed analysis of dataset
  - [x] Print example distribution over height
  - [x] Print class distribution depending on tree height
  - [ ] Consider frequency of examples when printing the class distributions
  - [ ] Print very common phrases/subtrees. Is there a pattern?
  - [ ] Try to annotate examples by POS tag, and find grammar rules for combining by binary tree. Is there something common/a pattern when a word (left or right) is more important?
  - [ ] Analyse the similarity of words (preinitialized values). Can we find synonyms between words? (High cosine-similarity and same humanly annotation?)
  - [ ] Check whether there are different annotations for same subtree in dataset (check for bugs in dataset).
