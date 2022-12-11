# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 21:25:15 2022

@author:ETI-Strategy
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys
import tensorflow as tf
from tensorflow.keras.layers import Dense,Dropout,Conv2D,Flatten,Embedding,MaxPooling2D
from tensorflow.keras import layers
from sklearn.metrics import cohen_kappa_score,balanced_accuracy_score,matthews_corrcoef,recall_score,f1_score,accuracy_score,precision_score
import warnings
import os 
from Evalute_mut import Classification_result, plot_confusion_matrix, plot_roc_curve
from sklearn.impute import SimpleImputer
import oddt
from oddt.fingerprints import InteractionFingerprint,SimpleInteractionFingerprint
from utils1 import read_ZINC_smiles, smiles_to_onehot, convert_to_graph,skip_connection,gated_skip_connection,readout,layers1,graph_convolution,morgan_fp,get_data,get_train_test,listdir,rdkit_des,get_des,TransNan
warnings.filterwarnings('ignore')

####Dataset division####
train_listname=[]
train_listname=listdir('./Training', train_listname)
train_listname.sort()
test_listname=[]
test_listname=listdir('./Testing', test_listname)
test_listname.sort()
train_fp, test_fp, train_y, test_y = [], [], [], []   
for i in range(len(train_listname)):
    x_train_fp,y_train,x_test_fp,y_test=get_train_test(train_listname[i],test_listname[i])
    # if not (((y_train == 1).all() or (y_train == 0).all()) or ((y_test == 0).all()) or (y_test == 1).all()):
    train_fp.append(x_train_fp)
    train_y.append(y_train)
    test_fp.append(x_test_fp)
    test_y.append(y_test)  

#RDKIT 2D descriptors##
test_rdkit_des=[]
train_rdkit_des=[]
scl=StandardScaler()
imputer = SimpleImputer(missing_values=np.nan,strategy="mean")
for i in range(len(train_listname)):
    test_des=get_des(test_listname[i])
    train_des=get_des(train_listname[i])
    nan_test=TransNan(test_des)
    nan_train=TransNan(train_des)
    all_test_des=scl.fit_transform(nan_test)
    all_train_des=scl.fit_transform(nan_train)
    last_test_des=imputer.fit_transform(all_test_des)
    last_train_des=imputer.fit_transform(all_train_des)
    test_rdkit_des.append(last_test_des)
    train_rdkit_des.append(last_train_des)

 
###graph ferature#####
train_graph=[]
test_graph=[]
train_atom=[]
test_atom=[]
for i in range(len(train_listname)):
    smi_train=pd.read_csv(train_listname[i],usecols=['Smiles'])
    smi_train=np.squeeze(smi_train.values)
    smi_test=pd.read_csv(test_listname[i],usecols=['Smiles'])
    smi_test=np.squeeze(smi_test.values)
    train_graph_feature,train_atom_feature =convert_to_graph(list(smi_train))
    test_graph_feature,test_atom_feature = convert_to_graph(list(smi_test))
    train_graph.append(train_graph_feature)
    test_graph.append(test_graph_feature)
    train_atom.append(train_atom_feature)
    test_atom.append(test_atom_feature)

num_atoms=50
num_features=58
X = tf.placeholder(tf.float32, shape=[None, num_atoms, num_features])
A = tf.placeholder(tf.float32, shape=[None, num_atoms, num_atoms])
h=graph_convolution(X,A,1500,act='relu',using_sc='gsc')
num_layer=2
##构建gcn层数
for i in range(num_layer):
    h = graph_convolution(h,
                          A, 
                          1024, 
                          tf.nn.relu,
                          using_sc='gsc')
h=readout(h,1024,tf.nn.sigmoid)



####构建网络        
def init_weights(shape):
    return tf.Variable(tf.random.normal(shape, stddev = 0.01))
def bias_variable(shape):
    return tf.Variable(tf.constant(0.01, shape=shape))


n_cols_fp=train_fp[0].shape[1]
n_cols_rd=train_rdkit_des[0].shape[1]
X_fp = tf.compat.v1.placeholder(tf.float32, [None,n_cols_fp])
X_rd= tf.compat.v1.placeholder(tf.float32, [None,n_cols_rd])
Y = tf.compat.v1.placeholder(tf.float32, [None, 1])
keep_prob1 = tf.compat.v1.placeholder(dtype=tf.float32)
number=len(train_y)
batch_size=32
output1=layers1(X_fp)
output2=layers1(X_rd)
conc = tf.concat([output1,output2], 1)
output = tf.compat.v1.layers.dense(inputs=conc, units=1024,activation="relu")
output = tf.compat.v1.layers.batch_normalization(output)
output = tf.compat.v1.layers.dropout(output,rate=0.3)


temp_hid=1024
w=list()
for i in range(number):
    w.append(init_weights([temp_hid, 1]))

b=list()
for i in range(number):
    b.append(bias_variable([1]))

py_x=list()
for i in range(number):
    py_x.append(tf.sigmoid(tf.matmul(output,w[i]) + b[i]))

cost=list()
for i in range(number):
    cost.append(tf.compat.v1.losses.log_loss(labels=Y, predictions=py_x[i]))

lr=1e-4
train_op=list()
for i in range(number):
    train_op.append(tf.compat.v1.train.AdamOptimizer(learning_rate = lr).minimize(cost[i]))


SAVER_DIR = "/"
saver = tf.compat.v1.train.Saver()
ckpt_path = os.path.join(SAVER_DIR, "/")
ckpt = tf.train.get_checkpoint_state(SAVER_DIR)


####################################################################
#=========================== train part ============================#
####################################################################

data=[]
data1=[]
data2=[]
data3=[]
data4=[]
data5=[]
data6=[]
data7=[]

with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    best_BA = 0
    best_idx = 0
    aucs=[]
    recalls=[]
    f1s=[]
    accs=[]
    precisions=[]
    kappas=[]
    mccs=[]
    BAs=[]
    for epoch in range(100):#epoch
        training_batch = zip(range(0, len(train_y[0]), batch_size),
                             range(batch_size, len(train_y[0])+1, batch_size))
   #for start, end in tqdm.tqdm(training_batch):
        for start, end in training_batch:
            for i in range(number):
                sess.run(train_op[i], feed_dict={A:np.array(train_atom[i])[start:end],X:train_graph[i][start:end],X_fp:train_fp[i][start:end],X_rd:train_rdkit_des[i][start:end],Y: train_y[i][start:end],keep_prob1:0.5})
                #print(i)
   #print validation loss
        print("epoch:",epoch)
        for i in range(number):
            merr = sess.run(cost[i], feed_dict={A:np.array(test_atom[i]),X:test_graph[i],X_fp:test_fp[i],X_rd:test_rdkit_des[i],Y: test_y[i],keep_prob1: 0.5})
            print(merr, end = ' ')
        print()
        
        val_aucs=[]
        val_recalls=[]
        val_f1s=[]
        val_accs=[]
        val_precisions=[]
        val_kappas=[]
        val_mccs=[]
        val_BAs = []
        pred_list=[]
        prob_list=[]
        for i in range(number):
            val_prob = sess.run(py_x[i], feed_dict={A:np.array(test_atom[i]),X:test_graph[i],X_fp:test_fp[i],X_rd:test_rdkit_des[i],keep_prob1: 0.5})
            prob_list.append(list(val_prob))
            val_auc = roc_auc_score(test_y[i], val_prob)
            val_pred = [int(round(val_prob[k][0])) for k in range(len(val_prob))]
            pred_list.append(val_pred)
            val_recall = recall_score(test_y[i], val_pred)
            val_f1 = f1_score(test_y[i], val_pred)
            val_acc = accuracy_score(test_y[i], val_pred)
            val_precision = precision_score(test_y[i], val_pred)
            val_kappa = cohen_kappa_score(test_y[i], val_pred)
            val_mcc = matthews_corrcoef(test_y[i], val_pred)
            val_BA = balanced_accuracy_score(test_y[i], val_pred)
            np.set_printoptions(precision=2)
            val_aucs.append(val_auc)
            val_recalls.append(val_recall)
            val_f1s.append(val_f1)
            val_accs.append(val_acc)
            val_precisions.append(val_precision)
            val_kappas.append(val_kappa)
            val_mccs.append(val_mcc)
            val_BAs.append(val_BA)
        # print('mean validation auc: ', end = ' ')
        # print(np.mean(val_aucs))
        # print('mean validation recall: ', end = ' ')
        # print(np.mean(val_recalls))
        # print('mean validation f1_score: ', end = ' ')
        # print(np.mean(val_f1s))
        # print('mean validation accuracy: ', end = ' ')
        # print(np.mean(val_accs))
        # print('mean validation precision: ', end = ' ')
        # print(np.mean(val_precisions))
        # print('mean validation kappa: ', end = ' ')
        # print(np.mean(val_kappas))
        # print('mean validation mcc: ', end = ' ')
        # print(np.mean(val_mccs))
        print('mean validation BA: ', end = ' ')
        print(np.mean(val_BAs))
        data.append(np.mean(val_aucs))
        data1.append(np.mean(val_recalls))
        data2.append(np.mean(val_f1s))
        data3.append(np.mean(val_accs))
        data4.append(np.mean(val_kappas))
        data5.append(np.mean(val_mccs))     
        data6.append(np.mean(val_precisions))
        data7.append(np.mean(val_BAs))

        if best_BA < np.mean(val_BAs):
           BAs=val_BAs
           aucs=val_aucs
           recalls=val_recalls
           f1s=val_f1s
           accs=val_accs
           pres=val_precisions
           kaps=val_kappas
           mcs=val_mccs
           best_BA = np.mean(val_BAs)
           best_idx = epoch
           save_path = saver.save(sess, ckpt_path, global_step = best_idx)
           print('model saved!')
           print()
           
           
print('best epoch index: '+str(best_idx))
print('best valid mean AUC:'+ str(data[best_idx]))
print('best valid mean BA : '+ str(best_BA))
print('best valid mean MCC:'+ str(data5[best_idx]))
print('best valid mean F1_score:'+ str(data2[best_idx])) 
print('best valid mean KAPPA:'+ str(data4[best_idx]))
print('best valid mean RECALL : '+ str(data1[best_idx]))
print('best valid mean ACC:'+ str(data3[best_idx]))
print('best valid mean PRE:'+ str(data6[best_idx])) 



