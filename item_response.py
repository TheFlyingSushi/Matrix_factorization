from utils import *

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    y = theta - beta
    z = -np.log(1 + np.exp(y))   
    #before_sum = np.multiply(data, x) + np.multiply(1 - data, 1 - x)
    before_sum = np.multiply(data, (y + z)) + np.multiply(1 - data, z)
    before_sum[np.isnan(before_sum)] = 0
    log_lklihood = np.sum(before_sum)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood

def neg_log_likelihood_dict(data, theta, beta):

    y = theta - beta
    z = -np.log(1 + np.exp(y)) 
    a = y + z

    log_lklihood = 0
    for i in range(len(data["is_correct"])):
        user_index = data["user_id"][i]
        question_index = data["question_id"][i]
        log_lklihood += data["is_correct"][i] * (a[user_index][question_index]) + (1 - data["is_correct"][i])*(z[user_index][question_index])

    return -log_lklihood

def update_theta_beta(data, lr, theta, beta):
    """ Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    #updating theta
    b = -(sigmoid(theta - beta))
    a = 1 + b
    new_theta = np.multiply(data, a) + np.multiply((1-data),b)
    new_theta[np.isnan(new_theta)] = 0
    theta += lr*np.sum(new_theta, axis=1)

    #updating beta
    b = (sigmoid(theta - beta))
    a = -1 + b
    new_beta = np.multiply(data, a) + np.multiply((1-data),b)
    new_beta[np.isnan(new_beta)] = 0
    beta += lr*np.sum(new_beta, axis=0)

    #print(theta.shape, beta.shape)
    #print((theta - beta).shape, data.shape)


    #find how many questions got answered? Not sure if i need it
    #students = (~np.isnan(theta)).sum(1) #rows
    #questions = (~np.isnan(theta)).sum(0) #columns
    #beta = np.nan_to_num(beta)
    #theta = np.nan_to_num(theta)
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta


def irt(data, val_data, lr, iterations):
    """ Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    # TODO: Initialize theta and beta.
    theta = np.zeros((data.shape[0], 1))
    beta = np.zeros((1, data.shape[1]))

    print(theta.shape, beta.shape)
    val_acc_lst = []
    train_neg = []
    val_neg = []

    for i in range(iterations):
        neg_lld = neg_log_likelihood(data, theta=theta, beta=beta)
        val_lld = neg_log_likelihood_dict(val_data, theta, beta)
        score = evaluate(data=val_data, theta=np.ravel(theta.T), beta=np.ravel(beta))
        val_acc_lst.append(score)
        train_neg.append(neg_lld)
        val_neg.append(val_lld)
        print("NLLK: {} \t Score: {}".format(neg_lld, score))
        theta, beta = update_theta_beta(data, lr, theta, beta)

    plt.title("Negative LogLikelihood")
    plt.xlabel("Iterations")
    plt.ylabel("Negative LogLikelihood")
    plt.plot(range(iterations), train_neg, label="train loglikelihood")
    plt.plot(range(iterations), val_neg, label="Valdiation loglikelihood")
    plt.legend()
    plt.show()
    """
    ability = np.arange(np.min(theta), np.max(theta), 0.02)
    flat_beta = beta.flatten()
    question1 = sigmoid(ability - flat_beta[0])
    question2 = sigmoid(ability - flat_beta[1])
    question3 = sigmoid(ability - flat_beta[2])
    question4 = sigmoid(ability - flat_beta[3])
    question5 = sigmoid(ability - flat_beta[4])

    plt.title("Question answerability as student ability changes")
    plt.xlabel("Student ability")
    plt.ylabel("Chance to answer correct")
    plt.plot(ability, question1, label="Question 1")
    plt.plot(ability, question2, label="Question 2")
    plt.plot(ability, question3, label="Question 3")
    plt.plot(ability, question4, label="Question 4")
    plt.plot(ability, question5, label="Question 5")
    plt.legend()
    plt.show()"""
    # TODO: You may change the return values to achieve what you want.
    return theta, beta, val_acc_lst


def evaluate(data, theta, beta):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])


def main():
    train_data = load_train_csv("data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("data")
    val_data = load_valid_csv("data")
    test_data = load_public_test_csv("data")

    #####################################################################
    # TODO:                                                             #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################
    """
    for lr in [0.0001, 0.0005, 0.001, 0.005, 0.01]:
        print("Learning Rate: ", lr)
        iterations = 20
        irt(sparse_matrix.todense(), val_data, lr, iterations)
    """
    lr = 0.005
    iterations = 20
    irt(sparse_matrix.todense(), test_data, lr, iterations)


if __name__ == "__main__":
    main()
