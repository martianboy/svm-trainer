//
//  main.cpp
//  SVMClassifier
//
//  Created by Abbas Mashayekh on 10/4/1393 AP.
//  Copyright (c) 1393 Abbas Mashayekh. All rights reserved.
//

#include <iostream>
#include <iostream>
#include <fstream>
#include <cstring>
#include "libsvm/svm.h"

#define malloc_array(type,n) (type *)malloc((n)*sizeof(type))

using namespace std;

svm_problem* read_problem(const char *);
svm_problem* create_problem(int);
svm_parameter* create_training_param();

void train_model();
void free_problem(svm_problem *problem);

int get_record_length(char *);
int get_sparse_length(char *, int);

struct svm_parameter params;

int main(int argc, const char * argv[]) {
    train_model();
    
    return 0;
}

svm_node* sparsify(initializer_list<double> dense_arr) {
    int sparse_length = 0;
    
    for (double value : dense_arr)
        if (value != 0)
            sparse_length++;
    
    int i = 0, sparse_i = 0;
    struct svm_node *sparse_arr = new struct svm_node[sparse_length + 1];
    sparse_arr[sparse_length].index = -1;
    
    for (double value : dense_arr) {
        if (value != 0) {
            sparse_arr[sparse_i].value = value;
            sparse_arr[sparse_i++].index = i;
        }
        ++i;
    }
    
    return sparse_arr;
}

ostream& operator <<(ostream& stream, svm_node* sparse_arr) {
    int last_index = 0;
    for (int sparse_i = 0; sparse_arr[sparse_i].index != -1; sparse_i++) {
        for (int j = last_index; j < sparse_arr[sparse_i].index; j++)
            stream << "0 ";
        last_index = sparse_arr[sparse_i + 1].index;
        stream << sparse_arr[sparse_i].value << " ";
    }
    
    return stream;
}

ostream& operator <<(ostream& stream, initializer_list<double> list) {
    for (double v : list) {
        if (v >= 0)
            stream << " ";
        
        stream << v << " ";
    }
    return stream;
}

void test_vector(svm_model *model, initializer_list<double> dense_arr) {
    struct svm_node *test_vector;
    test_vector = sparsify(dense_arr);
    cout << dense_arr << "-> " << svm_predict(model, test_vector) << "\n";
    free(test_vector);
}

void train_model() {
    struct svm_problem *problem = read_problem("/Users/abbas/Downloads/AND.txt");
    struct svm_parameter *params = create_training_param();
    struct svm_model *model;
    
    const char *param_check_value = svm_check_parameter(problem, params);
    
    if (param_check_value != NULL)
        cout << "Check params result: " << param_check_value << endl;
        
    
    if (param_check_value == NULL) {
        model = svm_train(problem, params);
        
        cout << "\nPredicted values:\n";
        
        test_vector(model, {-1, -1});
        test_vector(model, {-1, +1});
        test_vector(model, {+1, -1});
        test_vector(model, {+1, +1});

        svm_free_and_destroy_model(&model);
    }

    free_problem(problem);
    free(params);
}

svm_parameter* create_training_param() {
    struct svm_parameter* param = (struct svm_parameter*) malloc(sizeof(svm_parameter));

    param->svm_type = ONE_CLASS;
    param->kernel_type = LINEAR;
    param->cache_size = 100.0;
    param->eps = 1e-3;
    param->shrinking = 1;
    param->probability = 0;
    param->nu = 0.5;

    return param;
}

struct svm_problem* read_problem(const char *path) {
    ifstream ifs(path);
    if ( (ifs.rdstate() & ifstream::failbit ) != 0 ) {
        cerr << "Error opening dataset file.\n";
        return NULL;
    }
    
    ifs.seekg(0, ifs.end);
    long length = 1L + ifs.tellg();     // Add one for ending zero-byte
    cout << "File length: " << length << endl;

    ifs.seekg(0, ifs.beg);

    // Read the whole dataset into buf
    char *buf = new char[length];
    ifs.read(buf, length);
    printf("Last character: %d\n", buf[length - 1]);

    int record_length = get_record_length(buf);
    long record_count = length / (2 * record_length);
    cout << "Record count: " << record_count << endl;

    int sparse_length = (int)record_count * record_length; // get_sparse_length(buf, record_length);
    cout << "Sparse count: " << sparse_length << endl;

    struct svm_problem *problem = create_problem((int)record_count);
    struct svm_node* record_values = malloc_array(struct svm_node, sparse_length);
    problem->x[0] = record_values;
    
    int record_index = 0, sparse_value_index = 0, field_index = 0;
    for (char *c = buf; ; c++) {
        if (*c == ' ')
            continue;

//        if (*c == '1' && field_index < record_length - 1) {
//            record_values[sparse_value_index].index = field_index;
//            cout << sparse_value_index << ": Adding sparse value 1 at index " << record_values[sparse_value_index].index << endl;
//            record_values[sparse_value_index].value = 1;
//            sparse_value_index++;
//        }

        if (*c == '1' || *c == '0') {
            if (field_index == record_length - 1)
                problem->y[record_index] = (*c == '1' ? 1 : -1);
            else {
                record_values[sparse_value_index].value = (*c == '1' ? 1 : -1);
                record_values[sparse_value_index].index = field_index;
                cout << sparse_value_index << ": Adding sparse value " << record_values[sparse_value_index].value << " at index " << record_values[sparse_value_index].index << endl;
                sparse_value_index++;
            }
            field_index++;
        }
        
        if (*c == '\n' || *c == 0) {
            field_index = 0;
            record_values[sparse_value_index].value = 0;
            record_values[sparse_value_index].index = -1;
            cout << sparse_value_index << ": Adding sparse value 0 at index " << record_values[sparse_value_index].index << endl;
            sparse_value_index++;
            
            if (++record_index < record_count)
                problem->x[record_index] = &record_values[sparse_value_index];
        }
        
        if (*c == 0)
            break;
    }

    cout << "Added " << sparse_value_index << " sparse values." << endl << endl;
    cout << "Y: ";
    for (int i = 0; i < 4; i++) {
        cout << problem->y[i] << " ";
    }
    cout << endl << endl;

    cout << "Record values:" << endl;
    for (int i = 0; i < sparse_value_index; i++) {
        cout << "(" << record_values[i].index << ", " << record_values[i].value << ") ";
    }
    cout << endl << endl;

    cout << "X: ";
    for (int i = 0; i < record_count; i++) {
        for (int j = 0; problem->x[i][j].index != -1; j++) {
            cout << "(" << problem->x[i][j].index << ", " << problem->x[i][j].value << ") ";
        }
        cout << endl;
    }
    cout << endl;
    
    return problem;
}

// Gets count of all non-zero input values (skips class value)
int get_sparse_length(char *buf, int record_length) {
    int length = 0;
    int field_index = 0;
    
    for (char *c = buf; *c != 0; c++) {
        if (*c == '1' || *c == '0')
            field_index++;
        
        if (*c == '1' && field_index < record_length)
            length++;
        
        if(*c == '\n') {
            length++;
            field_index = 0;
        }
    }

    // Last line
    length++;
    
    cout << "Sparse length: " << length << endl;
    return length;
}

// Gets length of an input vector in dense mode (i.e. including zeros)
int get_record_length(char *raw_ds) {
    int length = 0;
    
    for (char *c = raw_ds; *c != '\n'; c++) {
        if(*c == '0' || *c == '1') length++;
    }

    cout << "Record length: " << length << endl;
    return length;
}

// Allocates memory for a new svm_problem structure
svm_problem* create_problem(int record_count) {
    struct svm_problem *problem = (struct svm_problem*) malloc(sizeof(svm_problem));

    problem->l = record_count;
    problem->y = malloc_array(double, problem->l);
    problem->x = malloc_array(struct svm_node *, problem->l);
    
    return problem;
}

// Frees allocated memory for a svm_problem
void free_problem(svm_problem *problem) {
    free(problem->y);
//    free(problem->x[0]);
    free(problem->x);
    free(problem);
}
