import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem.rdMolDescriptors import CalcTPSA
import tensorflow as tf
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from tensorflow.keras.layers import Dense,Dropout,Conv2D,Flatten,Embedding,MaxPooling2D
from tensorflow.keras import layers
import os

def read_data(filename):
    f = open(filename + '.smiles', 'r')
    contents = f.readlines()

    smiles = []
    labels = []
    for i in contents:
        smi = i.split()[0]
        label = int(i.split()[2].strip())

        smiles.append(smi)
        labels.append(label)

    num_total = len(smiles)
    rand_int = np.random.randint(num_total, size=(num_total,))
    
    return np.asarray(smiles)[rand_int], np.asarray(labels)[rand_int]

def smiles_to_onehot(smi_list):
    def smiles_to_vector(smiles, vocab, max_length):
        while len(smiles)<max_length:
            smiles +=" "
        return [vocab.index(str(x)) for x in smiles]

    vocab = np.load('./vocab.npy')
    smi_total = []
    for smi in smi_list:
        smi_onehot = smiles_to_vector(smi, list(vocab), 120)
        smi_total.append(smi_onehot)
    return np.asarray(smi_total)

def convert_to_graph(smiles_list):
    adj = []
    adj_norm = []
    features = []
    maxNumAtoms = 100
    for i in smiles_list:
        # Mol
        iMol = Chem.MolFromSmiles(i.strip())
        #Adj
        iAdjTmp = Chem.rdmolops.GetAdjacencyMatrix(iMol)
        # Feature
        if( iAdjTmp.shape[0] <= maxNumAtoms):
            # Feature-preprocessing
            iFeature = np.zeros((maxNumAtoms, 58))
            iFeatureTmp = []
            for atom in iMol.GetAtoms():
                iFeatureTmp.append( atom_feature(atom) ) ### atom features only
            iFeature[0:len(iFeatureTmp), 0:58] = iFeatureTmp ### 0 padding for feature-set
            features.append(iFeature)

            # Adj-preprocessing
            iAdj = np.zeros((maxNumAtoms, maxNumAtoms))
            iAdj[0:len(iFeatureTmp), 0:len(iFeatureTmp)] = iAdjTmp + np.eye(len(iFeatureTmp))
            adj.append(np.asarray(iAdj))
    features = np.asarray(features)

    return features,adj
    
def atom_feature(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                      ['C', 'N', 'O', 'S', 'F', 'H', 'Si', 'P', 'Cl', 'Br',
                                       'Li', 'Na', 'K', 'Mg', 'Ca', 'Fe', 'As', 'Al', 'I', 'B',
                                       'V', 'Tl', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn',
                                       'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'Mn', 'Cr', 'Pt', 'Hg', 'Pb']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5]) +
                    [atom.GetIsAromatic()])    # (40, 6, 5, 6, 1)

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    #print list((map(lambda s: x == s, allowable_set)))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def skip_connection(input_X, new_X, act):
    # Skip-connection, H^(l+1)_sc = H^(l) + H^(l+1)
    inp_dim = int(input_X.get_shape()[2])
    out_dim = int(new_X.get_shape()[2])

    if(inp_dim != out_dim):
        output_X = act(new_X + tf.layers.dense(input_X, units=out_dim, use_bias=False))

    else:
        output_X = act(new_X + input_X)

    return output_X        

def gated_skip_connection(input_X, new_X, act):
    # Skip-connection, H^(l+1)_gsc = z*H^(l) + (1-z)*H^(l+1)
    inp_dim = int(input_X.get_shape()[2])
    out_dim = int(new_X.get_shape()[2])

    def get_gate_coefficient(input_X, new_X, out_dim):
        X1 = tf.layers.dense(input_X, units=out_dim, use_bias=True)
        X2 = tf.layers.dense(new_X, units=out_dim, use_bias=True)
        gate_coefficient = tf.nn.sigmoid(X1 + X2)

        return gate_coefficient

    if(inp_dim != out_dim):
        input_X = tf.layers.dense(input_X, units=out_dim, use_bias=False)

    gate_coefficient = get_gate_coefficient(input_X, new_X, out_dim)
    output_X = tf.multiply(new_X, gate_coefficient) + tf.multiply(input_X, 1.0-gate_coefficient)

    return output_X

def readout(input_X, hidden_dim, act):
    # Readout, Z = sum_{v in G} NN(H^(L)_v)
    output_Z = tf.layers.dense(input_X, 
                               units=hidden_dim, 
                               use_bias=True,
                               activation=None,
                               kernel_initializer=tf.contrib.layers.xavier_initializer())
    output_Z = tf.reduce_sum(output_Z, axis=1)
    output = act(output_Z)

    return output_Z


def layers1(X_des):
    model1 =tf.keras.models.Sequential([
            layers.Dense(1024, use_bias=False,activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(512, use_bias=False,activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(256, use_bias=False,activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.3)
            ])
    return model1(X_des)

def layers2(X_fp):
    model2 =tf.keras.models.Sequential([
            layers.Dense(1024, use_bias=False,activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(512, use_bias=False,activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(256, use_bias=False,activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.5)
            ])
    return model2(X_fp)

def layers3(X_fp):
    model3=tf.keras.Sequential([
    layers.Conv2D(16,1, (1, 1),use_bias=True, padding="valid", activation="relu"),
    layers.MaxPooling2D(pool_size=(6, 6), strides=(1, 1), padding='same'),
    layers.Flatten(),
    layers.Dense(1024,activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(512, use_bias=False,activation="relu"),
    ])
    return model3(X_fp)

def graph_convolution(input_X, input_A, hidden_dim, act, using_sc):
    # Graph Convolution, H^(l+1) = A{H^(l)W^(l)+b^(l))
    output_X = tf.layers.dense(input_X,
                               units=hidden_dim, 
                               use_bias=True,
                               activation=None,
                               kernel_initializer=tf.contrib.layers.xavier_initializer())
    output_X = tf.matmul(input_A, output_X)

    if( using_sc == 'sc' ):
        output_X = skip_connection(input_X, output_X, act)

    elif( using_sc == 'gsc' ):
        output_X = gated_skip_connection(input_X, output_X, act)

    elif( using_sc == 'no' ):
        output_X = act(output_X)

    else:
        output_X = gated_skip_connection(input_X, output_X)

    return output_X

def morgan_fp(smiles):
	mol = Chem.MolFromSmiles(smiles)
	fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
	npfp = np.array(list(fp.ToBitString())).astype('int8')
	return npfp

def get_data(filepath):
    smi=pd.read_csv(filepath, usecols=["Smiles"])
    smi=np.squeeze(smi.values)
    x_fp=[morgan_fp(s) for s in smi]
    x_fp=np.array(x_fp)
    y=pd.read_csv(filepath, usecols=['label']).values
    return x_fp,y

def get_train_test(file_train,file_test):
    x_train_des,y_train=get_data(file_train)
    x_test_des,y_test=get_data(file_test) 
    return x_train_des,y_train,x_test_des,y_test

def listdir(path, list_name): 
    for file in os.listdir(path):  
        file_path = os.path.join(path, file)  
        if os.path.isdir(file_path):  
            listdir(file_path, list_name)  
        else:  
            list_name.append(file_path)
    return list_name

def rdkit_des(smiles):
    mol = Chem.MolFromSmiles(smiles)
    descs = [desc_name[0] for desc_name in Descriptors._descList]
    calc = MoleculeDescriptors.MolecularDescriptorCalculator(descs)
    des = calc.CalcDescriptors(mol)
    twod=np.array(des)
    return twod
def get_des(filepath):
    smi=pd.read_csv(filepath, usecols=["Smiles"])
    smi=np.squeeze(smi.values)
    x_fp=[rdkit_des(s) for s in smi]
    x_fp=np.array(x_fp)
    y=pd.read_csv(filepath, usecols=['label']).values
    return x_fp

def TransNan(data_set):
    for i in range(data_set.shape[1]):
        mean_des = np.nanmean(data_set[:,i])
        for j in range(data_set.shape[0]):
            if np.isnan(data_set[j,i]):
                data_set[j,i] = mean_des
    return data_set