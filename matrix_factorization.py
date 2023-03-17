from csc311.project_311.utils import *
from scipy.linalg import sqrtm

import numpy as np
import matplotlib.pyplot as plt

def svd_reconstruct(matrix, k):
    """ Given the matrix, perform singular value decomposition
    to reconstruct the matrix.

    :param matrix: 2D sparse matrix
    :param k: int
    :return: 2D matrix
    """
    # First, you need to fill in the missing values (NaN) to perform SVD.
    # Fill in the missing values using the average on the current item.
    # Note that there are many options to do fill in the
    # missing values (e.g. fill with 0).
    new_matrix = matrix.copy()
    mask = np.isnan(new_matrix)
    masked_matrix = np.ma.masked_array(new_matrix, mask)
    item_means = np.mean(masked_matrix, axis=0)
    new_matrix = masked_matrix.filled(item_means)

    # Next, compute the average and subtract it.
    item_means = np.mean(new_matrix, axis=0)
    mu = np.tile(item_means, (new_matrix.shape[0], 1))
    new_matrix = new_matrix - mu

    # Perform SVD.
    Q, s, Ut = np.linalg.svd(new_matrix, full_matrices=False)
    s = np.diag(s)

    # Choose top k eigenvalues.
    s = s[0:k, 0:k]
    Q = Q[:, 0:k]
    Ut = Ut[0:k, :]
    s_root = sqrtm(s)

    # Reconstruct the matrix.
    reconst_matrix = np.dot(np.dot(Q, s_root), np.dot(s_root, Ut))
    reconst_matrix = reconst_matrix + mu
    return np.array(reconst_matrix)


def squared_error_loss(data, u, z):
    """ Return the squared-error-loss given the data.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param u: 2D matrix
    :param z: 2D matrix
    :return: float
    """
    loss = 0
    for i, q in enumerate(data["question_id"]):
        loss += (data["is_correct"][i]
                 - np.sum(u[data["user_id"][i]] * z[q])) ** 2.
    return 0.5 * loss


def update_u_z(train_data, lr, u, z):
    """ Return the updated U and Z after applying
    stochastic gradient descent for matrix completion.

    :param train_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param u: 2D matrix
    :param z: 2D matrix
    :return: (u, z)
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    # Randomly select a pair (user_id, question_id).
    i = \
        np.random.choice(len(train_data["question_id"]), 1)[0]
    c = train_data["is_correct"][i]
    n = train_data["user_id"][i]
    q = train_data["question_id"][i]
    temp_u_n = u[n].reshape(-1, 1)
    temp_z_q = z[q].reshape(-1, 1)
    new_u = temp_u_n + lr * (c - np.transpose(temp_u_n) @ temp_z_q) * temp_z_q
    new_z = temp_z_q + lr * (c - np.transpose(temp_u_n) @ temp_z_q) * temp_u_n
    u[n] = new_u.reshape(len(u[n]),)
    z[q] = new_z.reshape(len(z[q]),)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return u, z


def als(train_data, k, lr, num_iteration):
    """ Performs ALS algorithm. Return reconstructed matrix.

    :param train_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :param lr: float
    :param num_iteration: int
    :return: 2D reconstructed Matrix.
    """
    # Initialize u and z
    u = np.random.uniform(low=0, high=1 / np.sqrt(k),
                          size=(len(set(train_data["user_id"])), k))
    z = np.random.uniform(low=0, high=1 / np.sqrt(k),
                          size=(len(set(train_data["question_id"])), k))

    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    sqr_error_list = []
    for num_it in range(num_iteration):
        new_u, new_z = update_u_z(train_data, lr, u, z)
        u = new_u
        z = new_z
        if num_it % 1000 == 0:
            sqr_error_list.append(squared_error_loss(train_data, u, z))
            print(squared_error_loss(train_data, u, z))
        # print("num_iteration " + str(num_it))
    y_axis = np.arange(1, num_iteration / 1000 + 1, 1)
    plt.plot(y_axis, sqr_error_list, label="squared-error-loss")
    plt.xlabel('num_iterations')
    plt.ylabel('error-loss')
    plt.legend()
    plt.show()
    mat = u @ np.transpose(z)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return mat


def svd_test(matrix, k_list):
    """
    report the construction error of matrix and its svd matrix for all values k
    in k_list
    """
    constr_error = []
    for k in k_list:
        svd_matrix = svd_reconstruct(matrix, k)
        temp = matrix - svd_matrix
        total = 0
        for i in range(len(temp)):
            row = temp[i]
            for j in range(len(row)):
                if not np.isnan(row[j]):
                    total += row[j] * row[j]
        constr_error.append(total / 2)
    return constr_error

def main():
    train_matrix = load_train_sparse("../data").toarray()
    train_data = load_train_csv("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    #####################################################################
    # TODO:
    # hyper_parameter values of k:
    mat = als(train_data, 7, 0.04, 90000)
    # mat = als(val_data, 7, 0.04, 90000)
    round_matrix = np.round(mat)
    total = 0
    num_correct = 0
    for i in range(len(val_data["user_id"])):
        # continue
        user = val_data["user_id"][i]
        q = val_data["question_id"][i]
        is_correct = val_data["is_correct"][i]
        pred = round_matrix[user][q]
        if pred == is_correct:
            num_correct += 1
        total += 1
    # print(num_correct / total)

    total = 0
    num_correct = 0
    for i in range(len(test_data["user_id"])):
        continue
        user = test_data["user_id"][i]
        q = test_data["question_id"][i]
        is_correct = test_data["is_correct"][i]
        pred = round_matrix[user][q]
        if pred == is_correct:
            num_correct += 1
        total += 1
    # print(num_correct / total)
    # (SVD) Try out at least 5 different k and select the best k        #
    # using the validation set.                                         #
    #####################################################################
    # pass
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:
    # FYI: in order to run this code uncomment continue for each of the loops.
    k_list = [2, 5, 7, 8, 10, 20, 25, 30, 40, 50]
    for k in k_list:
        continue
        svd_matrix = svd_reconstruct(train_matrix, k)
        round_matrix = np.round(svd_matrix)
        total = 0
        num_correct = 0
        for i in range(len(val_data["user_id"])):
            user = val_data["user_id"][i]
            q = val_data["question_id"][i]
            is_correct = val_data["is_correct"][i]
            pred = round_matrix[user][q]
            if pred == is_correct:
                num_correct += 1
            total += 1
        print("accuracy for k = " + str(k) + ": " + str(num_correct / total))
    k_best = [2, 5, 7, 8, 10]
    for k in k_best:
        continue
        svd_matrix = svd_reconstruct(train_matrix, k)
        round_matrix = np.round(svd_matrix)
        total = 0
        num_correct = 0
        for i in range(len(test_data["user_id"])):
            user = test_data["user_id"][i]
            q = test_data["question_id"][i]
            is_correct = test_data["is_correct"][i]
            pred = round_matrix[user][q]
            if pred == is_correct:
                num_correct += 1
            total += 1
        print(
            "test accuracy for k = " + str(k) + ": " + str(num_correct / total))
    # (ALS) Try out at least 5 different k and select the best k        #
    # using the validation set.                                         #
    #####################################################################
    # pass
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
