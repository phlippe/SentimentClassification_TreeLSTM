# Efficient bottom-up learning of tree-structured neural networks for sentiment classification
We present a novel approach for   efficient   loss   weighting   in   a   tree-structured neural network.  Current methods consider either only the top-node prediction for loss calculation,  or weight all nodes equally yielding to a strongly imbalanced class loss.  Our method progresses through the tree, starting at word level, to focus the loss on misclassified nodes.  We propose  three  different  heuristics  for  determining  such  misclassifications  and  investigate their effect and performance on the Stanford Sentiment Treebank based on a binary Tree-LSTM model. The results show a significant improvement compared to  previous  models  concerning  accuracy and overfitting. The figure below visualizes the concept.

![alt text](https://raw.githubusercontent.com/phlippe/NLP_Project/master/paper/general_concept_extended.png)

This paper was written in the context of the second partical assignment (Sentiment Classification with Deep Learning) for the course "Natural Language Processing 1" at the University of Amsterdam. The full paper can be found [here](paper/NLP1_Paper_Lippe_Halm.pdf).

## Code structure

The code is structured into two jupyter notebooks. 

### Mandatory_model.ipynb

This notebook summarizes all experiments and models from the mandatory assignment part (BOW, CBOW, Deep CBOW, LSTM, Tree-LSTM). In addition, we create all plots which are shown in the paper in this notebook. The results for other models are taken into account by the text files provided next to the notebooks.

### TreeLSTMs.ipynb

This notebook contains all experiments from our proposed models. Please note that executing this notebook will take a longer time as the training is set to 50,000 iterations. We therefore advise to use the pretrained models (see below) for test purpose.

### Pretrained models

Pretrained models can be found here (_will be added soon_). For plotting, the needed test predictions/accuracies are provided in text file which are saved in the notebook folder. 
