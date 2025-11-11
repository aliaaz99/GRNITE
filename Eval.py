import os
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(rc={"lines.linewidth": 2}, palette  = "deep", style = "ticks")
from sklearn.metrics import precision_recall_curve, roc_curve, auc, precision_score, recall_score, f1_score, confusion_matrix, balanced_accuracy_score, matthews_corrcoef
from itertools import product, permutations, combinations, combinations_with_replacement
from tqdm import tqdm

import networkx as nx
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from networkx.convert_matrix import from_pandas_adjacency

import pdb


def PRROC(ref_path, input_path, output_path, method, directed = True, selfEdges = False, plotFlag = False, restrict=False):
    '''
    Computes areas under the precision-recall and ROC curves
    for a given dataset for each algorithm.
    
    :param directed:   A flag to indicate whether to treat predictionsas directed edges (directed = True) or undirected edges (directed = False).
    :type directed: bool
        
    :param selfEdges:   A flag to indicate whether to includeself-edges (selfEdges = True) or exclude self-edges (selfEdges = False) from evaluation.
    :type selfEdges: bool
    
    :param plotFlag:   A flag to indicate whether or not to save PR and ROC plots.
    :type plotFlag: bool
        
    :returns:
            - AUPRC: A dictionary containing AUPRC values for each algorithm
            - AUROC: A dictionary containing AUROC values for each algorithm
    '''
    
    # Read file for trueEdges
    trueEdgesDF = pd.read_csv(ref_path,
                                sep = ',', 
                                header = 0, index_col = None)
    
    trueEdgesDF['Gene1'] = trueEdgesDF['Gene1'].str.upper()
    trueEdgesDF['Gene2'] = trueEdgesDF['Gene2'].str.upper()
    true_genes = list(set(trueEdgesDF['Gene1'].tolist()+trueEdgesDF['Gene2'].tolist()))
    
    # set-up outDir that stores output directory name
    outDir = output_path
    
    if directed:
        predDF = pd.read_csv(input_path, sep = ',', header =  0, index_col = None)
        predDF['Gene1'] = predDF['Gene1'].str.upper()
        predDF['Gene2'] = predDF['Gene2'].str.upper()
        pred_genes = list(set(predDF['Gene1'].tolist()+predDF['Gene2'].tolist()))

        if restrict:
            if len(pred_genes) < len(true_genes):
                trueEdgesDF = trueEdgesDF[trueEdgesDF['Gene1'].isin(pred_genes) & trueEdgesDF['Gene2'].isin(pred_genes)]


        cm, precision_val , recall_val, f1_val, precision, recall, FPR, TPR, AUPRC, AUROC, ball_acc, mcc, gms, jc, num_edges_possible, num_edges_true, num_edges_pred = computeScores(trueEdgesDF, predDF, method, directed = True, selfEdges = selfEdges)


        PRName = '/PRplot'
        ROCName = '/ROCplot'
    else:
        predDF = pd.read_csv(input_path, sep = ',', header =  0, index_col = None)
        predDF['Gene1'] = predDF['Gene1'].str.upper()
        predDF['Gene2'] = predDF['Gene2'].str.upper()
        pred_genes = list(set(predDF['Gene1'].tolist()+predDF['Gene2'].tolist()))

        if restrict:
            if len(pred_genes) < len(true_genes):
                trueEdgesDF = trueEdgesDF[trueEdgesDF['Gene1'].isin(pred_genes) & trueEdgesDF['Gene2'].isin(pred_genes)]


        cm, precision_val , recall_val, f1_val, precision, recall, FPR, TPR, AUPRC, AUROC, ball_acc, mcc, gms, jc, num_edges_possible, num_edges_true, num_edges_pred = computeScores(trueEdgesDF, predDF, method, directed = False, selfEdges = selfEdges)
    
        PRName = '/uPRplot'
        ROCName = '/uROCplot'
    if (plotFlag):
         ## Make PR curves
        legendList = []
        print("Recall: ", recall)
        print("Precision: ", precision)
        # sns.lineplot(recall,precision, ci=None)
        plt.plot(recall,precision, linewidth = 2)
        legendList.append(method + ' (AUPRC = ' + str("%.2f" % (AUPRC))+')')
        plt.xlim(0,1)    
        plt.ylim(0,1)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend(legendList) 
        # plt.savefig(outDir+PRName+'_'+method+'.pdf')
        plt.savefig(outDir+method+'_pr.png')
        plt.clf()

        ## Make ROC curves
        legendList = []
        # sns.lineplot(FPR,TPR, ci=None)
        plt.plot(FPR,TPR, linewidth = 2)
        legendList.append(method + ' (AUROC = ' + str("%.2f" % (AUROC))+')')

        plt.plot([0, 1], [0, 1], linewidth = 1.5, color = 'k', linestyle = '--')

        plt.xlim(0,1)    
        plt.ylim(0,1)
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.legend(legendList) 
        # plt.savefig(outDir+ROCName+'_'+method+'.pdf')
        plt.savefig(outDir+method+'_roc.png')
        plt.clf()

    return cm, precision_val, recall_val, f1_val, AUPRC, AUROC, ball_acc, jc, num_edges_possible, num_edges_true, num_edges_pred



def computeScores(trueEdgesDF, predEdgeDF, method, directed = True, selfEdges = True):
    '''        
    Computes precision-recall and ROC curves
    using scikit-learn for a given set of predictions in the 
    form of a DataFrame.
    

    :param trueEdgesDF:   A pandas dataframe containing the true classes.The indices of this dataframe are all possible edgesin a graph formed using the genes in the given dataset. This dataframe only has one column to indicate the classlabel of an edge. If an edge is present in the reference network, it gets a class label of 1, else 0.
    :type trueEdgesDF: DataFrame
        
    :param predEdgeDF:   A pandas dataframe containing the edge ranks from the prediced network. The indices of this dataframe are all possible edges.This dataframe only has one column to indicate the edge weightsin the predicted network. Higher the weight, higher the edge confidence.
    :type predEdgeDF: DataFrame
    
    :param directed:   A flag to indicate whether to treat predictionsas directed edges (directed = True) or undirected edges (directed = False).
    :type directed: bool
    :param selfEdges:   A flag to indicate whether to includeself-edges (selfEdges = True) or exclude self-edges (selfEdges = False) from evaluation.
    :type selfEdges: bool
        
    :returns:
            - prec: A list of precision values (for PR plot)
            - recall: A list of precision values (for PR plot)
            - fpr: A list of false positive rates (for ROC plot)
            - tpr: A list of true positive rates (for ROC plot)
            - AUPRC: Area under the precision-recall curve
            - AUROC: Area under the ROC curve
    '''

    if directed:        
        # Initialize dictionaries with all 
        # possible edges
        if selfEdges:
            possibleEdges = list(product(np.unique(trueEdgesDF.loc[:,['Gene1','Gene2']]),
                                         repeat = 2))
        else:
            possibleEdges = list(permutations(np.unique(trueEdgesDF.loc[:,['Gene1','Gene2']]),
                                         r = 2))
        
        TrueEdgeDict = {'|'.join(p):0 for p in possibleEdges}
        PredEdgeDict = {'|'.join(p):0 for p in possibleEdges}
        
        # Compute TrueEdgeDict Dictionary
        # 1 if edge is present in the ground-truth
        # 0 if edge is not present in the ground-truth
        #for key in tqdm(TrueEdgeDict.keys()):
        #    if len(trueEdgesDF.loc[(trueEdgesDF['Gene1'] == key.split('|')[0]) &
        #           (trueEdgesDF['Gene2'] == key.split('|')[1])])>0:
        #            TrueEdgeDict[key] = 1
        for i,row in tqdm(trueEdgesDF.iterrows(), total=len(trueEdgesDF), desc="Processing Ground truth"):
            if row['Gene1']+'|'+row['Gene2'] in TrueEdgeDict.keys():
                TrueEdgeDict[row['Gene1']+'|'+row['Gene2']] = 1
            else:
                print('ERROR: Added new key to all possible edges')
                
        #for key in tqdm(PredEdgeDict.keys()):
        #    subDF = predEdgeDF.loc[(predEdgeDF['Gene1'] == key.split('|')[0]) &
        #                       (predEdgeDF['Gene2'] == key.split('|')[1])]
        #    if len(subDF)>0:
        #        PredEdgeDict[key] = np.abs(subDF.weight.values[0])
        for i,row in tqdm(predEdgeDF.iterrows(), total=len(predEdgeDF), desc="Processing Predictions"):
            if row['Gene1']+'|'+row['Gene2'] in PredEdgeDict.keys():
                if 'our' in method:
                    if row['Gene2'] in trueEdgesDF['Gene1'].tolist():
                        PredEdgeDict[row['Gene1']+'|'+row['Gene2']] = 1
                        PredEdgeDict[row['Gene2']+'|'+row['Gene1']] = 1
                else:
                    PredEdgeDict[row['Gene1']+'|'+row['Gene2']] = 1

    # if undirected
    else:
        # Initialize dictionaries with all 
        # possible edges
        if selfEdges:
            possibleEdges = list(combinations_with_replacement(np.unique(trueEdgesDF.loc[:,['Gene1','Gene2']]),
                                                               r = 2))
        else:
            possibleEdges = list(combinations(np.unique(trueEdgesDF.loc[:,['Gene1','Gene2']]),
                                                               r = 2))
        TrueEdgeDict = {'|'.join(p):0 for p in possibleEdges}
        PredEdgeDict = {'|'.join(p):0 for p in possibleEdges}

        # Compute TrueEdgeDict Dictionary
        # 1 if edge is present in the ground-truth
        # 0 if edge is not present in the ground-truth

        for key in tqdm(TrueEdgeDict.keys()):
            if len(trueEdgesDF.loc[((trueEdgesDF['Gene1'] == key.split('|')[0]) &
                           (trueEdgesDF['Gene2'] == key.split('|')[1])) |
                              ((trueEdgesDF['Gene2'] == key.split('|')[0]) &
                           (trueEdgesDF['Gene1'] == key.split('|')[1]))]) > 0:
                TrueEdgeDict[key] = 1  

        # Compute PredEdgeDict Dictionary
        # from predEdgeDF

        for key in tqdm(PredEdgeDict.keys()):
            subDF = predEdgeDF.loc[((predEdgeDF['Gene1'] == key.split('|')[0]) &
                               (predEdgeDF['Gene2'] == key.split('|')[1])) |
                              ((predEdgeDF['Gene2'] == key.split('|')[0]) &
                               (predEdgeDF['Gene1'] == key.split('|')[1]))]
            if len(subDF)>0:
                PredEdgeDict[key] = max(np.abs(subDF.weight.values))

                
                
    # Combine into one dataframe
    # to pass it to sklearn
    outDF = pd.DataFrame([TrueEdgeDict,PredEdgeDict]).T
    outDF.columns = ['TrueEdges','PredEdges']
    
    precision_val = precision_score(y_true=outDF['TrueEdges'], y_pred=outDF['PredEdges'], pos_label=1)
    recall_val = recall_score(y_true=outDF['TrueEdges'], y_pred=outDF['PredEdges'], pos_label=1)
    f1_val = f1_score(y_true=outDF['TrueEdges'], y_pred=outDF['PredEdges'], pos_label=1, average='macro')
    
    fpr, tpr, thresholds = roc_curve(y_true=outDF['TrueEdges'],
                                     y_score=outDF['PredEdges'], pos_label=1)

    prec, recall, thresholds = precision_recall_curve(y_true=outDF['TrueEdges'],
                                                      probas_pred=outDF['PredEdges'], pos_label=1)
    cm = confusion_matrix(y_true=outDF['TrueEdges'], y_pred=outDF['PredEdges'])

    ball_acc = balanced_accuracy_score(y_true=outDF['TrueEdges'], y_pred=outDF['PredEdges'])

    mcc = matthews_corrcoef(y_true=outDF['TrueEdges'], y_pred=outDF['PredEdges'])

    # gms = geometric_mean_score(y_true=outDF['TrueEdges'], y_pred=outDF['PredEdges'])
    gms = ((cm[1,1]/(cm[1,1]+cm[1,0]) * cm[0,0]/(cm[0,0]+cm[0,1])))**0.5

    TP = cm[1,1]
    FP = cm[0,1]
    FN = cm[1,0]
    denom = TP + FP + FN
    if denom == 0:
        jaccard_val = 0.0
    else:
        jaccard_val = TP / denom

    num_edges_possible = outDF.shape[0]
    num_edges_true = outDF['TrueEdges'].sum()
    num_edges_pred = outDF['PredEdges'].sum()

    return cm, precision_val, recall_val, f1_val, prec, recall, fpr, tpr, auc(recall, prec), auc(fpr, tpr), ball_acc, mcc, gms, jaccard_val, num_edges_possible, num_edges_true, num_edges_pred


def Jaccard(ref_path, input_path, output_path, method):
    """
    A function to compute median pairwirse Jaccard similarity index
    of predicted top-k edges for a given set of datasets (obtained from
    the same reference network). Here k is the number of edges in the
    reference network (excluding self loops). 
    
    
    :param evalObject: An object of class :class:`BLEval.BLEval`.
    :type evalObject: :obj:`BLEval`
      
      
    :param algorithmName: Name of the algorithm for which the Spearman correlation is computed.
    :type algorithmName: str
      
      
    :returns:
        - median: Median of Jaccard correlation values
        - mad: Median Absolute Deviation of  the Spearman correlation values
    """

    rankDict = {}
    trueEdgesDF = pd.read_csv(ref_path, sep = ',', header = 0, index_col = None)

    possibleEdges = list(permutations(np.unique(trueEdgesDF.loc[:,['Gene1','Gene2']]),
                                    r = 2))

    TrueEdgeDict = {'|'.join(p):0 for p in possibleEdges}
    PredEdgeDict = {'|'.join(p):0 for p in possibleEdges}

    # Compute TrueEdgeDict Dictionary
    # 1 if edge is present in the ground-truth
    # 0 if edge is not present in the ground-truth
    numEdges = 0
    #for key in TrueEdgeDict.keys():
    #    if len(trueEdgesDF.loc[(trueEdgesDF['Gene1'] == key.split('|')[0]) &
    #            (trueEdgesDF['Gene2'] == key.split('|')[1])])>0:
    #            TrueEdgeDict[key] = 1
    #            numEdges += 1
    for i,row in tqdm(trueEdgesDF.iterrows(), total=len(trueEdgesDF), desc="Processing Ground truth"):
        if row['Gene1']+'|'+row['Gene2'] in TrueEdgeDict.keys():
            TrueEdgeDict[row['Gene1']+'|'+row['Gene2']] = 1
            numEdges+=1
        else:
            print('ERROR: Added new key to all possible edges')

    #algos = evalObject.input_settings.algorithms
    predDF = pd.read_csv(input_path, sep=",", header=0, index_col=None)
    predDF = predDF.loc[(predDF['Gene1'] != predDF['Gene2'])]
    predDF.drop_duplicates(keep = 'first', inplace=True)
    predDF.reset_index(drop = True,  inplace= True)
    # check if ranked edges list is empty
    # if so, it is just set to an empty set

    if not predDF.shape[0] == 0:
        # we want to ensure that we do not include
        # edges without any edge weight
        # so check if the non-zero minimum is
        # greater than the edge weight of the top-kth
        # node, else use the non-zero minimum value.
        predDF.weight = predDF.weight.round(6)
        predDF.weight = predDF.weight.abs()

        # Use num True edges or the number of
        # edges in the dataframe, which ever is lower
        maxk = min(predDF.shape[0], numEdges)
        edgeWeightTopk = predDF.iloc[maxk-1].weight

        nonZeroMin = np.nanmin(predDF.weight.replace(0, np.nan).values)
        bestVal = max(nonZeroMin, edgeWeightTopk)

        newDF = predDF.loc[(predDF['weight'] >= bestVal)]
        rankDict[method] = set(newDF['Gene1'] + "|" + newDF['Gene2'])
    else:
        rankDict[method] = set([])

    Jdf = computePairwiseJacc(rankDict)
    df = Jdf.where(np.triu(np.ones(Jdf.shape),  k = 1).astype(np.bool_))
    df = df.stack().reset_index()
    df.columns = ['Row','Column','Value']
    return(df.Value.median(),df.Value.mad())


def computePairwiseJacc(inDict):
    """
    A helper function to compute all pairwise Jaccard similarity indices
    of predicted top-k edges for a given set of datasets (obtained from
    the same reference network). Here k is the number of edges in the
    reference network (excluding self loops). 
    
    :param inDict:  A dictionary contaninig top-k predicted edges  for each dataset. Here, keys are the dataset name and the values are the set of top-k edges.
    :type inDict: dict
    :returns:
        A dataframe containing pairwise Jaccard similarity index values
    """
    jaccDF = {key:{key1:{} for key1 in inDict.keys()} for key in inDict.keys()}
    for key_i in tqdm(inDict.keys(), desc="Computing pairwise jaccard"):
        for key_j in inDict.keys():
            num = len(inDict[key_i].intersection(inDict[key_j]))
            den = len(inDict[key_i].union(inDict[key_j]))
            if den != 0:
                jaccDF[key_i][key_j] = num/den
            else:
                jaccDF[key_i][key_j] = 0
    return pd.DataFrame(jaccDF)


def EarlyPrec(ref_path, input_path, method):
    temp = pd.read_csv(input_path, sep = ',',header = 0, index_col = None)
    trueDF = pd.read_csv(ref_path, sep = ',',header = 0, index_col = None)
    if 'our' in method:
        predDF = []
        for i,row in tqdm(temp.iterrows(), total=len(temp), desc="Processing baseline"):
            if row['Gene2'] in trueDF['Gene1'].tolist():
                predDF.append({'Gene1': row['Gene2'], 'Gene2': row['Gene1'], 'weight': row['weight']})
            predDF.append({'Gene1': row['Gene1'], 'Gene2': row['Gene2'], 'weight': row['weight']})
        predDF = pd.DataFrame(predDF)
    else:
        predDF = temp[['Gene1', 'Gene2', 'weight']].copy()
    
    #predDF.columns = ['Gene1','Gene2','weight']

    predDF['Gene1'] = predDF['Gene1'].str.upper()
    predDF['Gene2'] = predDF['Gene2'].str.upper()
    predDF.sort_values(by='weight', ascending=False, inplace=True)

    trueEdgesDF = trueDF.loc[(trueDF['Gene1'] != trueDF['Gene2'])]
    trueEdgesDF.drop_duplicates(keep = 'first', inplace=True)
    trueEdgesDF.reset_index(drop=True, inplace=True)
    
    unique_genes = pd.concat([predDF['Gene2'], predDF['Gene1']]).unique()

    netDF = trueEdgesDF.iloc[:, :2].copy()
    netDF.columns = ['Gene1','Gene2']
    netDF['Gene1'] = netDF['Gene1'].str.upper()
    netDF['Gene2'] = netDF['Gene2'].str.upper()
    netDF = netDF[(netDF.Gene1.isin(unique_genes)) & (netDF.Gene2.isin(unique_genes))]
    # Remove self-loops.
    netDF = netDF[netDF.Gene1 != netDF.Gene2]
    # Remove duplicates (there are some repeated lines in the ground-truth networks!!!). 
    netDF.drop_duplicates(keep = 'first', inplace=True)
    trueEdgesDF = netDF.copy()
    unique_gene1 = netDF['Gene1'].unique()
    unique_gene2_combined = pd.concat([netDF['Gene2'], netDF['Gene1']]).unique()
    predEdgeDF = predDF[
        predDF['Gene1'].isin(unique_gene1) &
        predDF['Gene2'].isin(unique_gene2_combined)
    ].copy()
    
    # predEdgeDF['Edges'] = predEdgeDF['Gene1'] + "|" + predEdgeDF['Gene2']
    predEdgeDF.loc[:, 'Edges'] = predEdgeDF['Gene1'] + "|" + predEdgeDF['Gene2']
    # limit the predicted edges to the genes that are in the ground truth
    Eprec = {}
    # Consider only edges going out of TFs

    trueEdgesDF = trueEdgesDF.loc[(trueEdgesDF['Gene1'] != trueEdgesDF['Gene2'])]
    trueEdgesDF.drop_duplicates(keep='first', inplace=True)
    trueEdgesDF.reset_index(drop=True, inplace=True)

    uniqueNodes = np.unique(trueEdgesDF.loc[:, ['Gene1', 'Gene2']])
    possibleEdges_TF = set(product(set(trueEdgesDF.Gene1), set(uniqueNodes)))

    # Get a list of all possible interactions
    possibleEdges_noSelf = set(permutations(uniqueNodes, r=2))

    # Find intersection of above lists to ignore self edges
    possibleEdges = possibleEdges_TF.intersection(possibleEdges_noSelf)

    TrueEdgeDict = {'|'.join(p): 0 for p in possibleEdges}

    trueEdges = trueEdgesDF['Gene1'] + "|" + trueEdgesDF['Gene2']
    trueEdges = trueEdges[trueEdges.isin(TrueEdgeDict)]
    numEdges = len(trueEdges)

    predDF_new = predEdgeDF[predEdgeDF.loc[:, 'Edges'].isin(TrueEdgeDict)]

    # Use num True edges or the number of
    # edges in the dataframe, which ever is lower
    maxk = min(predDF_new.shape[0], numEdges)
    edgeWeightTopk = predDF_new.iloc[maxk-1].weight

    nonZeroMin = np.nanmin(predDF_new.weight.replace(0, np.nan).values)
    bestVal = max(nonZeroMin, edgeWeightTopk)

    newDF = predDF_new.loc[(predDF_new['weight'] >= bestVal)]
    rankDict = set(newDF['Gene1'] + "|" + newDF['Gene2'])

    # Erec = {}
    intersectionSet = rankDict.intersection(trueEdges)
    Eprec = len(intersectionSet)/len(rankDict)
    randomEprc = len(trueEdges) / len(possibleEdges)
    EPR = Eprec/randomEprc
    # Erec = len(intersectionSet)/len(trueEdges)

    return EPR


# Add Other datasets here
sample_names = [
                "GroundGAN/PBMC-ALL-Human/", 
                "TF500/hESC/",
                 ]

# Add other methods here
method_names = [
                "step1_grnite",
                "scenic-network", "scenic-network_grnite",
                "grnboost", "grnboost_grnite",
                ]



# Final results stored per sample
all_results = {}

for sample in sample_names:
    # Load reference edges
    sample_name = sample.split('/')[-2]
    print("=============================")
    print(f"Processing Data: {sample_name}")
    method_metrics = []

    for method in method_names:
        print("****")
        print(f"Method: {method}")
        try:
            cm, precision, recall, f1, AUPRC, AUROC, ball_acc, jc, possible_edges, true_edges, pred_edges = PRROC("Data/"+sample+sample_name+"-ref_present.csv", "Data/"+sample+sample_name+"-"+method+".csv", "Data/"+sample+sample_name, sample_name+'-'+method, 
                                                            directed = True, selfEdges = False, plotFlag = False, restrict=False)

        except Exception as e:
            print(f"Error in PRROC for method {method} on sample {sample_name}: {e}")
            precision, recall, f1, AUPRC, AUROC, ball_acc, jc = 0, 0, 0, 0, 0, 0, 0
            possible_edges, true_edges, pred_edges = 0, 0, 0

        

        method_metrics.append([method, ball_acc, jc, precision, recall, f1, AUPRC, AUROC,  possible_edges, true_edges, pred_edges])
        print("Confusion Matrix:")
        print(cm)
                


    # Create DataFrame with results
    df_metrics = pd.DataFrame(method_metrics, columns=["Method","Balanced_Acc", "JC", "Precision", "Recall", "F1", "AUPRC", "AUROC", "Possible_Edges", "True_Edges", "Predicted_Edges"])
    all_results[sample] = df_metrics

writer = pd.ExcelWriter('Example_eval.xlsx')
for key in all_results.keys():
    sheet_name = key.split('/')[-2]
    all_results[key].to_excel(writer, sheet_name=sheet_name, index=False)
writer.close()
