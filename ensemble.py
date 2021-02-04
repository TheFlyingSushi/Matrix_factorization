#import os
#os.chdir("c:/users/allen/desktop/starter_code/part_a")

from sklearn.impute import KNNImputer
from random import randint
from sklearn.utils import resample
from utils import *
import item_response
import matrix_factorization
from scipy import sparse

def calculate_accuracy(data, knn_matrix, item_response_model, mat_fac_matrix):
    total_accurate = 0
    for i in range(len(data["is_correct"])):
        cur_user_id = data["user_id"][i]
        cur_question_id = data["question_id"][i]
        # cur_accuracy = 0

        # # knn
        # if ((knn_matrix[cur_question_id, cur_user_id] >= 0.5 and data["is_correct"][i]) or (knn_matrix[cur_question_id, cur_user_id] < 0.5 and not data["is_correct"][i])):
        #     cur_accuracy += 1

        # # item_response
        # if ((item_response.sigmoid(item_response_model[0][cur_user_id] - item_response_model[1][:,cur_question_id]) >= 0.5 and data["is_correct"][i]) or 
        # (item_response.sigmoid(item_response_model[0][cur_user_id] - item_response_model[1][:,cur_question_id]) < 0.5 and not data["is_correct"][i])):
        #     cur_accuracy += 1

        # # matrix_factorization
        # if ((mat_fac_matrix[cur_user_id, cur_question_id] >= 0.5 and data["is_correct"][i]) or (mat_fac_matrix[cur_user_id, cur_question_id] < 0.5 and not data["is_correct"][i])):
        #     cur_accuracy += 1

        # if (cur_accuracy > 1.5):
        #     total_accurate += 1
        prediction = knn_matrix[cur_question_id, cur_user_id] + item_response.sigmoid(item_response_model[0][cur_user_id] - item_response_model[1][:,cur_question_id]) + mat_fac_matrix[cur_user_id, cur_question_id]
        if ((prediction > 1.5 and data["is_correct"][i]) or (prediction < 1.5 and not data["is_correct"][i])):
            total_accurate += 1
    return total_accurate / len(data["is_correct"])

def generate_sample(dictionary, dimensions):
    sample = np.empty(dimensions)
    sample.fill(np.nan)
    for i in range(len(dictionary["user_id"])):
        index = randint(0, len(dictionary["user_id"])-1)
        sample[dictionary["user_id"][index],dictionary["question_id"][index]] = dictionary["is_correct"][index]
    return sample


# main
train_data = load_train_csv("../data")
sparse_matrix = load_train_sparse("../data")
val_data = load_valid_csv("../data")
test_data = load_public_test_csv("../data")

# knn
sample1 = generate_sample(train_data, sparse_matrix.shape)

k = 21
knn_model = KNNImputer(n_neighbors=k)
knn_mat = knn_model.fit_transform(sample1.T)

# item_response
sample2 = generate_sample(train_data, sparse_matrix.shape)
sample2 = sparse.csr_matrix(sample2)
theta, beta = item_response.irt(sample2.todense(), val_data, 0.005, 20)[:2]

# matrix_factorization
temp = resample(train_data["user_id"], train_data["question_id"], train_data["is_correct"])
sample3 = {"user_id": temp[0], "question_id": temp[1], "is_correct": temp[2]}
mat_fac_mat = matrix_factorization.als(sample3, 8, 0.05, 90000)


train_acc = calculate_accuracy(train_data, knn_mat, (theta, beta), mat_fac_mat)
print("Train accuracy = ", train_acc)
val_acc = calculate_accuracy(val_data, knn_mat, (theta, beta), mat_fac_mat)
print("Validation accuracy = ", val_acc)
test_acc = calculate_accuracy(test_data, knn_mat, (theta, beta), mat_fac_mat)
print("Test accuracy = ", test_acc)

# if __name__ == "__main__":
#     main()
