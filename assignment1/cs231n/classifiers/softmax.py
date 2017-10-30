import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################

  N = X.shape[0]
  C = W.shape[1]
  y_enc = np.eye(C)[y]

  for i in xrange(N):
    f = X[i].dot(W)
    # numeric stability fix: shift the values of f so that the highest number is 0
    f -= np.max(f)
    loss += - f[y[i]] + np.log(np.sum(np.exp(f)))
    d = np.exp(f)
    d /= np.sum(d)
    for j in xrange(C):
        dW[:, j] += X[i] * (d[j] - y_enc[i][j])

  loss /= N
  loss += reg * np.sum(W * W)

  dW /= N
  dW += reg * 2 * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################

  N = X.shape[0]
  C = W.shape[1]

  y_enc = np.eye(C)[y]
  F = X.dot(W)
  # numeric stability fix
  F -= np.max(F)
  F_exp = np.exp(F)
  F_exp_sum = np.sum(F_exp, axis=1, keepdims=True)

  loss = np.sum(- np.sum(F * y_enc, axis=1, keepdims=True) + np.log(F_exp_sum))
  loss /= N
  loss += reg * np.sum(W * W)

  s = F_exp / F_exp_sum
  df = s - y_enc

  dW = X.T.dot(df) / N + reg * 2 * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

