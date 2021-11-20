import numpy as np


def add_row(A, k, i, j):
    """
    Add k times row j to row i.
    """
    n = A.shape[0]
    E = np.eye(n)
    E[i, j] += k
    return E @ A


def scale_row(A, k, i):
    """
    Multiply row i by k.
    """
    n = A.shape[0]
    E = np.eye(n)
    E[i, i] = k
    return E @ A


def switch_rows(A, i, j):
    """
    Switch row i and row j.
    """
    n = A.shape[0]
    E = np.eye(n)
    E[[i, j]] = E[[j, i]]
    return E @ A


def compute_rank(A):
    A = rref(A)
    return sum(np.count_nonzero(A, axis=1) != 0)


def rref(A):
    """ Return Row Echelon Form of matrix A """

    # if matrix A has no columns or rows,
    # it is already in REF, so we return itself
    r, c = A.shape
    if r == 0 or c == 0:
        return A

    # we search for non-zero element in the first column
    for i in range(len(A)):
        if A[i, 0] != 0:
            break
        else:
            # if all elements in the first column is zero,
            # we perform REF on matrix from second column
            B = rref(A[:, 1:])
            # and then add the first zero-column back
            return np.hstack([A[:, :1], B])

    # if non-zero element happens not in the first row,
    # we switch rows
    if i > 0:
        A = switch_rows(A, i, 0)

    # we divide first row by first element in it
    A = scale_row(A, 1/A[0, 0], 0)
    # we subtract all subsequent rows with first row (it has 1 now as first element)
    # multiplied by the corresponding element in the first column
    A[1:] -= A[0] * A[1:, 0:1]

    # we perform REF on matrix from second row, from second column
    B = rref(A[1:, 1:])

    # we add first row and first (zero) column, and return
    return np.vstack([A[:1], np.hstack([A[1:, :1], B])])