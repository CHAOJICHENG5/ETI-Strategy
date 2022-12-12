# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 14:27:38 2022

@author: ETI-Strategy
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys
import tensorflow as tf
from tensorflow.keras.layers import Dense,Dropout,Conv2D,Flatten,Embedding,MaxPooling2D
from tensorflow.keras import layers
from sklearn.metrics import cohen_kappa_score,balanced_accuracy_score,matthews_corrcoef,recall_score,f1_score,accuracy_score,precision_score
import warnings
from Evalute_mut import Classification_result, plot_confusion_matrix, plot_roc_curve
from sklearn.impute import SimpleImputer
from scipy.sparse import vstack as sparse_vstack
import oddt
from oddt.fingerprints import InteractionFingerprint,SimpleInteractionFingerprint
from utils import  smiles_to_onehot, convert_to_graph,skip_connection,gated_skip_connection,readout,layers1,graph_convolution,morgan_fp,get_data,get_train_test,listdir,rdkit_des,get_des,TransNan
import os
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

######docking descriptors######
imputer = SimpleImputer(strategy="median")
train_des_list=[]
train_des_list=listdir('./features/train_dock_descriptor', train_des_list)
train_des_list.sort()
test_des_list=[]
test_des_list=listdir('./features/test_dock_descriptor', test_des_list)
test_des_list.sort()
train_dock_des=[]
test_dock_des=[]
scl=StandardScaler()
for i in range(55):
    test_des=pd.read_csv(test_des_list[i],usecols=['r_i_docking_score',	'r_i_glide_ecoul','r_i_glide_eff_state_penalty','r_i_glide_einternal',	'r_i_glide_emodel','r_i_glide_energy',	'r_i_glide_erotb',	'r_i_glide_esite',	'r_i_glide_evdw','r_i_glide_gscore',	'r_i_glide_hbond',	'r_i_glide_ligand_efficiency',	'r_i_glide_ligand_efficiency_ln',	'r_i_glide_ligand_efficiency_sa',	'r_i_glide_lipo','r_i_glide_metal','r_i_glide_rewards',	'r_lp_Energy'])
    train_des=pd.read_csv(train_des_list[i],usecols=['r_i_docking_score',	'r_i_glide_ecoul','r_i_glide_eff_state_penalty','r_i_glide_einternal',	'r_i_glide_emodel','r_i_glide_energy',	'r_i_glide_erotb',	'r_i_glide_esite',	'r_i_glide_evdw','r_i_glide_gscore',	'r_i_glide_hbond',	'r_i_glide_ligand_efficiency',	'r_i_glide_ligand_efficiency_ln',	'r_i_glide_ligand_efficiency_sa',	'r_i_glide_lipo','r_i_glide_metal','r_i_glide_rewards',	'r_lp_Energy'])
    train_des=train_des.values.astype(np.float64)
    test_des=test_des.values.astype(np.float64)
    train_des=scl.fit_transform(train_des)
    test_des=scl.fit_transform(test_des)
    train_des=imputer.fit_transform(train_des)
    test_des=imputer.fit_transform(test_des)
    test_dock_des.append(np.array(test_des))
    train_dock_des.append(np.array(train_des))
    
####Binding PLIF generate######  
protein_list=[]
protein_list=listdir('./features/pocket_pdb',protein_list)
protein_list.sort()

receptors=[]
for i in range(len(protein_list)):
    protein = next(oddt.toolkit.readfile('pdb', protein_list[i]))
    protein.protein = True
    protein.addh(only_polar=True)
    receptors.append(protein)


train_list=[]
train_list=listdir('./features/train_dockpose', train_list)
train_list.sort()
test_list=[]
test_list=listdir('./features/test_dockpose', test_list)
test_list.sort()

train_ligand=[]
test_ligand=[]
for i in range(len(protein_list)):
    tr_ligand = list(oddt.toolkit.readfile('sdf', train_list[i]))
    te_ligand = list(oddt.toolkit.readfile('sdf', test_list[i]))
    train_ligand.append(tr_ligand)
    test_ligand.append(te_ligand)
    
train_ifp=[]    
for j in range(len(train_ligand)):    
    train_simpleIFP =np.array([oddt.fingerprints.InteractionFingerprint(l, receptors[j],strict = True) for l in train_ligand[j]])
    train_ifp.append(train_simpleIFP)
test_ifp=[]
for k in range(len(test_ligand)):
    test_simpleIFP=np.array([oddt.fingerprints.InteractionFingerprint(l, receptors[k],strict = True) for l in test_ligand[k]])
    test_ifp.append(test_simpleIFP)
    
train_plif=[]
test_plif=[]
max_bits=408
for i in range(len(train_ifp)):
    last_train_x=np.hstack((train_ifp[i],np.zeros((train_ifp[i].shape[0],max_bits-train_ifp[i].shape[1]))))
    last_test_x=np.hstack((test_ifp[i],np.zeros((test_ifp[i].shape[0],max_bits-test_ifp[i].shape[1]))))
    train_plif.append(last_train_x)
    test_plif.append(last_test_x)
    
### Low Variance Filter######    
# from sklearn.feature_selection import VarianceThreshold
# transfer = VarianceThreshold()
# train_x_plif=transfer.fit_transform(train_x_plif)
# test_x_plif=transfer.fit_transfrom(test_x_plif)

#### multi-task model constructing###    
def init_weights(shape):
    return tf.Variable(tf.random.normal(shape, stddev = 0.01))
def bias_variable(shape):
    return tf.Variable(tf.constant(0.01, shape=shape))


n_cols_dock=train_dock_des[0].shape[1]
n_cols_plif=train_plif[0].shape[1]
X_plif = tf.compat.v1.placeholder(tf.float32, [None,n_cols_plif])
X_dock= tf.compat.v1.placeholder(tf.float32, [None,n_cols_dock])
Y = tf.compat.v1.placeholder(tf.float32, [None, 1])
keep_prob1 = tf.compat.v1.placeholder(dtype=tf.float32)
number=len(train_y)
batch_size=32
output1=layers1(X_plif)
output2=layers1(X_dock)
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
                sess.run(train_op[i], feed_dict={X_dock:train_dock_des[i][start:end],X_plif:train_plif[i][start:end],Y: train_y[i][start:end],keep_prob1:0.5})
                #print(i)
   #print validation loss
        print("epoch:",epoch)
        for i in range(number):
            merr = sess.run(cost[i], feed_dict={X_dock:test_dock_des[i],X_plif:test_plif[i],Y: test_y[i],keep_prob1: 0.5})
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
            val_prob = sess.run(py_x[i], feed_dict={X_dock:test_dock_des[i],X_plif:test_plif[i],keep_prob1: 0.5})
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



#### single-task model constructing##
n_cols_dock = train_dock_des[0].shape[0]
n_cols_plif = train_plif[0].shape[0]
X_plif = tf.compat.v1.placeholder(tf.float32, [None,n_cols_plif])
X_dock =tf.compat.v1.placeholder(tf.float32, [None,n_cols_dock])
Y = tf.compat.v1.placeholder(tf.float32, [None, 1])
output1=layers1(X_plif)
output2=layers1(X_dock)
conc = tf.concat([output1,output2], 1)
output = tf.compat.v1.layers.dense(inputs=conc, units=1024,activation="relu")
output = tf.compat.v1.layers.batch_normalization(output)
output = tf.compat.v1.layers.dropout(output,rate=0.3)
keep_prob1 = tf.compat.v1.placeholder(dtype=tf.float32)

def init_weights(shape):
    return tf.Variable(tf.random.normal(shape, stddev = 0.01))
def bias_variable(shape):
    return tf.Variable(tf.constant(0.01, shape=shape))
temp_hid=1024
lr=1e-4
batch_size=32
w1 = init_weights([temp_hid, 1])
b1 = bias_variable([1])
py_x1 = tf.sigmoid(tf.matmul(output, w1) + b1)
cost1 = tf.compat.v1.losses.log_loss(labels=Y, predictions=py_x1)
train_op1 = tf.compat.v1.train.AdamOptimizer(learning_rate = lr).minimize(cost1)
prediction_error1 = cost1

SAVER_DIR = "/"
saver = tf.compat.v1.train.Saver()
ckpt_path = os.path.join(SAVER_DIR, "/")
ckpt = tf.train.get_checkpoint_state(SAVER_DIR)


EPIGs_Name = ['APEX1','ATM', 'AURKA', 'AURKB', 'BRD2', 'BRD4', 'BRPF1', 'CARM1', 'CDK1', 'CDK2', 'CDK5', 'CDK7', 
             'CHEK1', 'CHUK', 'CREBBP', 'DAPK3', 'DNMT1', 'DOT1L', 'EHMT2', 'EP300', 'EZH2', 'HDAC1', 'HDAC10',
             'HDAC11', 'HDAC2', 'HDAC3', 'HDAC4','HDAC5', 'HDAC6', 'HDAC7', 'HDAC8', 'HDAC9', 'JAK2', 'KAT2B', 'KDM1A', 
             'KDM4A', 'KDM4C', 'KDM4E', 'KDM5A', 'KDM6B', 'L3MBTL1', 'PARG', 'PARP1', 'PKN1', 'PRKAA1', 'PRKCB', 'PRKCD','PRKDC' ,
              'PRMT3', 'RPS6KA5', 'SIRT1', 'SIRT2', 'SIRT3', 'TOP2A', 'USP7']  

####################################################################
#=========================== train part ============================
####################################################################
all_metric_data=[]
prob_list=[]
pred_list=[]
for i in range(len(train_y)):
    SAVER_DIR = "/"
    saver = tf.compat.v1.train.Saver()
    ckpt_path = os.path.join(SAVER_DIR, EPIGs_Name[i])
    ckpt = tf.train.get_checkpoint_state(SAVER_DIR)
    data=[]
    data1=[]
    data2=[]
    data3=[]
    data4=[]
    data5=[]
    data6=[]
    data7=[]
    all_data=[]
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        best_BA = 0
        best_idx = 0
        aucs=[]
        prob_append=[]
        pred_append=[]
        for epoch in range(100):#epoch
            training_batch = zip(range(0, len(train_y[i]), batch_size),
                                 range(batch_size, len(train_y[i])+1, batch_size))
       #for start, end in tqdm.tqdm(training_batch):
            for start, end in training_batch:
                sess.run(train_op1, feed_dict={X_plif:train_plif[i][start:end],X_dock:train_dock_des[i][start:end],Y: train_y[i][start:end],keep_prob1:0.5})
       # print validation loss
            print("epoch:",epoch)
            merr = sess.run(prediction_error1, feed_dict={X_plif:test_plif[i],X_dock:test_dock_des[i],Y: test_y[i],keep_prob1: 0.5})
            print('loss:',merr, end = ' ')
            print()
            val_prob = sess.run(py_x1, feed_dict={X_plif:test_plif[i],X_dock:test_dock_des[i],keep_prob1: 0.5})
            prob_append.append(val_prob)
            val_aucs = roc_auc_score(test_y[i], val_prob)
            val_pred = [int(round(val_prob[k][0])) for k in range(len(val_prob))]
            pred_append.append(val_pred)
            val_recall = recall_score(test_y[i], val_pred)
            val_f1 = f1_score(test_y[i], val_pred)
            val_acc = accuracy_score(test_y[i], val_pred)
            val_precision = precision_score(test_y[i], val_pred)
            val_kappa = cohen_kappa_score(test_y[i], val_pred)
            val_mcc = matthews_corrcoef(test_y[i], val_pred)
            val_BA = balanced_accuracy_score(test_y[i], val_pred)
            print('test auc: ', end = ' ')
            print(round(val_aucs,5))
            print('test BA: ', end = ' ')
            print(round(val_BA,5))
            data.append(val_aucs)
            data1.append(val_recall)
            data2.append(val_f1)
            data3.append(val_acc)
            data4.append(val_kappa)
            data5.append(val_mcc)     
            data6.append(val_precision)
            data7.append(val_BA)
            if best_BA < val_BA:
                best_BA = val_BA
                recalls=val_recall
                f1s=val_f1
                accs=val_acc
                precision=val_precision
                kappa=val_kappa
                mcc=val_mcc
                AUC=val_aucs
                best_idx = epoch
                save_path = saver.save(sess, ckpt_path, global_step = best_idx)
                print('model saved!')
                print()
    
            else:
                pass
    
    print('best epoch index: '+str(best_idx))
    print('best valid auc: '+str(data[best_idx]))
    print('best valid mean BA : '+ str(best_BA))
    print('best valid mean MCC:'+ str(data5[best_idx]))
    print('best valid mean F1_score'+ str(data2[best_idx]))
    print('best valid mean KAPPA:'+ str(data4[best_idx]))
    print('best valid mean RECALL : '+ str(data1[best_idx]))
    print('best valid mean ACC:'+ str(data3[best_idx]))
    print('best valid mean PRE:'+ str(data6[best_idx]))  
