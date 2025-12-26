"""ridge.py: pytorch implementation of grid-searching ridge regression.

Ridge solutions for multiple regularization values are computed efficiently by
using the Woodbury identity. With X (n x d) representing the feature matrix, and
y (n x 1) the outcomes, the ridge solution is given by

    β = (X'X + l*I)^{-1}X'y

where l is the regularization coefficient. This can be reduced to

    (1/l)*(X'y - X'V(e + l*I)^{-1}(X'V)'X'y)

where Ue^{1/2}V' is the singular-value decomposition of X'. Since (e + lI) is a
diagonal matrix, its inverse can be computed efficiently simply by taking the
reciprocal of the diagonal elements. Then, (X'V)'X'y is a vector; so it can be
multiplied by (e + lI)^{-1} just by scalar multiplication.
"""

import torch


def _validate_ls(ls):
    """Ensure that ls is a 1-dimensional torch float/double tensor."""
    try:
        assert isinstance(ls, torch.Tensor)
        assert ls.dtype is torch.float or ls.dtype is torch.double
        assert len(ls.shape) == 1
    except AssertionError:
        raise AttributeError(
            "invalid ls: should be 1-dimensional torch float/double tensor"
        )


def _validate_XY(X, Y):
    """Ensure that X and Y are 2-dimensional torch float/double tensors, with
    proper sizes."""
    try:
        for inp in [X, Y]:
            assert isinstance(inp, torch.Tensor)
            assert inp.dtype is torch.float or inp.dtype is torch.double
            assert len(inp.shape) == 2
        assert X.dtype is Y.dtype
        assert X.shape[0] == Y.shape[0]
    except AssertionError:
        raise AttributeError(
            "invalid inputs: X and Y should be float/double tensors of shape "
            "(n, d) and (n, m) respectively, where n is the number of samples, "
            "d is the number of features, and m is the number of outputs"
        )


class MultiRidge:

    """Ridge model for multiple outputs and regularization strengths. A separate
    model is fit for each (output, regularization) pair."""

    def __init__(self, ls, scale_X=True, scale_thresh=1e-8):
        """
        Arguments:
                       ls: 1-dimensional torch tensor of regularization values.
                  scale_X: whether or not to scale X.
             scale_thresh: when standardizing, standard deviations below this
                           value are not used (i.e. they are set to 1).
        """
        self.ls = ls
        self.scale_X = scale_X
        self.scale_thresh = scale_thresh
        self.X_t = None
        self.Xm = None
        self.Xs = None
        self.e = None
        self.Q = None
        self.Y = None
        self.Ym = None

    def fit(self, X, Y):
        """
        Arguments:
            X: 2-dimensional torch tensor of shape (n, d) where n is the number
               of samples, and d is the number of features.
            Y: 2-dimensional tensor of shape (n, m) where m is the number of
               targets.
        """
        self.Xm = X.mean(dim=0, keepdim=True) # get the mean of each feature
        X = X - self.Xm # center the X
        if self.scale_X: # when scale_X set to True
            self.Xs = X.std(dim=0, keepdim=True) 
            self.Xs[self.Xs < self.scale_thresh] = 1 # cap at 1 for st below a threshod
            X = X / self.Xs # standardize the X


        # -------- Precompute SVD of X -----------
        self.X_t = X.t() # transpose X to get d x n (d: number of features, n: number of samples)
        _, S, V = self.X_t.svd() # SVD of X': X' = U S V'; V is n x n, S is vector of singular values
        self.e = S.pow_(2) # square the singular values to get e
        self.Q = self.X_t @ V # V is orthogonal, so X'V = U S; Q is d x n, which each column is a singular vector scaled by its singular value
        # ---------------------------------------

        self.Y = Y # n x m (m: number of targets); outputs
        self.Ym = Y.mean(dim=0) # get the mean of each target

        return self

    def _compute_pred_interms(self, y_idx, X_te_p): # y_idx: index of target, X_te_p: X_te @ Q
        Y_j, Ym_j = self.Y[:, y_idx], self.Ym[y_idx] # get the j-th target and its mean (y_idx: index of target)

        p_j = self.X_t @ (Y_j - Ym_j) # centered Y_j; this computes X' (Y_j - Ym_j), which means projecting the centered target onto the feature space

        r_j = self.Q.t() @ p_j # Q is d x n, so Q' is n x d; this computes (X'V)'X'(Y_j - Ym_j), which means projecting p_j onto the space spanned by the right singular vectors of X'

        N_te_j = X_te_p @ p_j # X_te_p is X_te @ Q; this computes X_te @ Q @ X' (Y_j - Ym_j); X_te is the test feature matrix

        return Ym_j, r_j, N_te_j

    def _predict_single(self, l, M_te, Ym_j, r_j, N_te_j):
        Yhat_te_j = (1 / l) * (N_te_j - M_te @ (r_j / (self.e + l))) + Ym_j
        return Yhat_te_j

    def _compute_single_beta(self, l, y_idx): # l: regularization value, y_idx: index of target
        Y_j, Ym_j = self.Y[:, y_idx], self.Ym[y_idx]
        beta = (1 / l) * (
            self.X_t @ (Y_j - Ym_j)
            - self.Q / (self.e + l) @ self.Q.t() @ self.X_t @ (Y_j - Ym_j) 
        ) # this is just the ridge regression formula as given in the file docstring
        return beta # d x 1, where d is the number of features, for the y_idx-th target

    def get_model_weights_and_bias(self, l_idxs): # l_idxs: list of indexes of regularization values
        betas = torch.zeros((self.X_t.shape[0], len(l_idxs))) # d x M, where M is the number of regularization values and d is the number of features
        for j, l_idx in enumerate(l_idxs):
            l = self.ls[l_idx] # self.ls is the list of regularization values
            betas[:, j] = self._compute_single_beta(l, j) # store the beta for the j-th target with regularization value l
        return betas, self.Ym

    def get_prediction_scores(self, X_te, Y_te, scoring):
        """Compute predictions for each (regulariztion, output) pair and return
        the scores as produced by the given scoring function.

        Arguments:
               X_te: 2-dimensional torch tensor of shape (n, d) where n is the
                     number of samples, and d is the number of features.
               Y_te: 2-dimensional tensor of shape (n, m) where m is the
                     number of targets.
            scoring: scoring function with signature scoring(y, yhat).

        Returns a (m, M) torch tensor of scores, where M is the number of
        regularization values.
        """
        X_te = X_te - self.Xm # center the test features with training mean (why not use test mean?)
        if self.scale_X:
            X_te = X_te / self.Xs # standardize the test features with training std
        M_te = X_te @ self.Q # M_te size n x n (what is this matrix representing?)

        scores = torch.zeros(Y_te.shape[1], len(self.ls), dtype=X_te.dtype) # m x M, where m is the number of targets, M is the number of regularization values
        for j, Y_te_j in enumerate(Y_te.t()): # iterate over each target; ???
            Ym_j, r_j, N_te_j = self._compute_pred_interms(j, X_te)
            for k, l in enumerate(self.ls):
                Yhat_te_j = self._predict_single(l, M_te, Ym_j, r_j, N_te_j)
                scores[j, k] = scoring(Y_te_j, Yhat_te_j).item()
        return scores

    def predict_single(self, X_te, l_idxs):
        """Compute a single prediction corresponding to each output.

        Arguments:
              X_te: 2-dimensional torch tensor of shape (n, d) where n is the
                    number of samples, and d is the number of features.
            l_idxs: iterable of length m (number of targets), with indexes
                    specifying the l value to use for each of the targets.

        Returns a (n, m) tensor of predictions.
        """
        X_te = X_te - self.Xm
        if self.scale_X:
            X_te = X_te / self.Xs

        M_te = X_te @ self.Q

        Yhat_te = []
        for j, l_idx in enumerate(l_idxs):
            Ym_j, r_j, N_te_j = self._compute_pred_interms(j, X_te)
            l = self.ls[l_idx]
            Yhat_te_j = self._predict_single(l, M_te, Ym_j, r_j, N_te_j)
            Yhat_te.append(Yhat_te_j)

        Yhat_te = torch.stack(Yhat_te, dim=1)
        return Yhat_te


class RidgeCVEstimator: # this class performs cross-validated ridge regression to select the best regularization parameter with best cross-validation score
    def __init__(self, ls, cv, scoring, scale_X=True, scale_thresh=1e-8):
        """Cross-validated ridge estimator.

        Arguments:
                       ls: 1-dimensional torch tensor of regularization values.
                       cv: cross-validation object implementing split.
                  scoring: scoring function with signature scoring(y, yhat).
                  scale_X: whether or not to scale X.
             scale_thresh: when standardizing, standard deviations below this
                           value are not used (i.e. they are set to 1).
        """
        _validate_ls(ls)
        self.ls = ls
        self.cv = cv
        self.scoring = scoring
        self.scale_X = scale_X
        self.scale_thresh = scale_thresh
        self.base_ridge = None
        self.mean_cv_scores = None
        self.best_l_scores = None
        self.best_l_idxs = None

    def fit(self, X, Y, groups=None):
        """Fit ridge model to given data.

        Arguments:
                 X: 2-dimensional torch tensor of shape (n, d) where n is the
                    number of samples, and d is the number of features.
                 Y: 2-dimensional tensor of shape (n, m) where m is the number
                    of targets.
            groups: groups used for cross-validation; passed directly to
                    cv.split.

        A separate model is learned for each target i.e. Y[:, j].
        """
        _validate_XY(X, Y)
        cv_scores = []

        for idx_tr, idx_val in self.cv.split(X, Y, groups):
            X_tr, X_val = X[idx_tr], X[idx_val]
            Y_tr, Y_val = Y[idx_tr], Y[idx_val]

            self.base_ridge = MultiRidge(self.ls, self.scale_X, self.scale_thresh)
            self.base_ridge.fit(X_tr, Y_tr)
            split_scores = self.base_ridge.get_prediction_scores(
                X_val, Y_val, self.scoring
            )
            cv_scores.append(split_scores)
            del self.base_ridge

        cv_scores = torch.stack(cv_scores)
        self.mean_cv_scores = cv_scores.mean(dim=0)
        self.best_l_scores, self.best_l_idxs = self.mean_cv_scores.max(dim=1)
        self.base_ridge = MultiRidge(self.ls, self.scale_X, self.scale_thresh)
        self.base_ridge.fit(X, Y)
        return self

    def predict(self, X):
        """Predict using cross-validated model.

        Arguments:
            X_te: 2-dimensional torch tensor of shape (n, d) where n is the
                  number of samples, and d is the number of features.

        Returns a (n, m) matrix of predictions.
        """
        if self.best_l_idxs is None:
            raise RuntimeError("cannot predict without fitting")
        return self.base_ridge.predict_single(X, self.best_l_idxs)

    def get_model_weights_and_bias(self):
        if self.best_l_idxs is None:
            raise RuntimeError("cannot return weight without fitting")
        return self.base_ridge.get_model_weights_and_bias(self.best_l_idxs)
