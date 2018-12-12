# Efficient bottom-up learning of tree-structured neural networks for sentiment classification
We present a novel approach for   efficient   loss   weighting   in   a   tree-structured neural network.  Current methods consider either only the top-node prediction for loss calculation,  or weight all nodes equally yielding to a strongly imbalanced class loss.  Our method progresses through the tree, starting at word level, to focus the loss on misclassified nodes.  We propose  three  different  heuristics  for  determining  such  misclassifications  and  investigate their effect and performance on the Stanford Sentiment Treebank based on a binary Tree-LSTM model. The results show a significant improvement compared to  previous  models  concerning  accuracy and overfitting.

This paper was written in the context of the second partical assignment (Sentiment Classification with Deep Learning) for the course "Natural Language Processing 1" at the University of Amsterdam.

## Code structure

The code is structured into two jupyter notebooks. 

### Mandatory_model.ipynb

This notebook summarizes all experiments and models from the mandatory assignment part (BOW, CBOW, Deep CBOW, LSTM, Tree-LSTM). In addition, we create all plots which are shown in the paper in this notebook. The results for other models are taken into account by the text files provided next to the notebooks.

### TreeLSTMs.ipynb

This notebook contains all experiments from our proposed models. Please note that executing this notebook will take a longer time as the training is set to 50,000 iterations. 

## Open ideas
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
