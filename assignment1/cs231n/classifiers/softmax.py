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
  D = X.shape[1]
  C = W.shape[1]

  for i in xrange(N):
    f = X[i].dot(W)
    # numeric stability fix: shift the values of f so that the highest number is 0
    f -= np.max(f)

    f_exp = np.exp(f)
    f_exp_sum = np.sum(f_exp)

    loss += -np.log(f_exp[y[i]] / f_exp_sum)

    df = f_exp / f_exp_sum
    df[y[i]] -= 1

    # for j in xrange(C):
    #     dW[:, j] += X[i] * df[j]
    dW += np.dot(X[i].reshape(D, 1), df.reshape(1, C))

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

  F = X.dot(W)
  # numeric stability fix
  F -= np.max(F)

  F_exp = np.exp(F)
  F_exp_sum = np.sum(F_exp, axis=1, keepdims=True)
  F_exp_y = F_exp[range(N), y].reshape(N, 1)

  loss = np.sum(-np.log(F_exp_y / F_exp_sum)) / N
  loss += .5 * reg * np.sum(W * W)

  dF = F_exp / F_exp_sum
  dF[range(N), y] -= 1

  dW = X.T.dot(dF) / N + reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

