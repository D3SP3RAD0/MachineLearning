import numpy as np
from matplotlib.colors import ListedColormap

class Perceptron(object):
    """Perceptron Class.
    Parameters
    ----------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Number of passes over the training dataset.
    random_state : int
      Random Number generator seed for random weight initialization

    Attributes
    ----------
    w_ : 1d-array 
      Weights after fitting. 
    errors_ : list
        Number of misclassifications (updates) in each epoch. 
    """
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

   
    def fit(self, X, y):
        """Fit Training Data. (Update weights)
            Parameters
            ----------
            X : {array-like}, shape = [n_examples, n_features]
              Training vectors, where n_examples is the number of examples 
              and n_features is the number of features.
            y : {array-like}, shape = [n_examples]
              Target Values
            
            Returns
            -------
            self : object
        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale = 0.1, size=1 + X.shape[1]) #NumPy random number generator pulling from normal distribution
        self.errors_ = []
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X,y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        """Calculate the net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1 , -1)

def plot_decision_regions(X, y, classifier, resolution=0.02):
  #setup marker generator and color map
  markers = ('s', 'x', 'o', '^', 'v')
  colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
  cmap = ListedColormap(colors[:len(np.unique(y))])

  #plot the decision surface
  x1_min, x1_max = X[:, 0].min() - 1
