
# coding: utf-8

# In[9]:

import sys
import os
from pathlib import Path
os.chdir(Path(os.getcwd()).resolve().parents[1])

import setup
from methods import runExperiments
#from methods import static_labelpropagation
from methods import amanda_dynamic
from methods import amanda_fixed
from methods import incremental_LN
from methods import deslizante_LN
from methods import compose_gmm_version


class Experiment():
     def __init__(self, method, K=None, excludingPercentage=None, densityFunction=None, clfName=None):
        self.method = method
        self.clfName = clfName
        self.densityFunction=densityFunction
        self.excludingPercentage = excludingPercentage
        self.K_variation = K


#def loadLevelResults(path, sep, key, steps):
#    originalAccs, F1s, time = setup.loadKeystroke(path, sep)
#    predictions = F1s[key]
#    predictions = [ predictions[i::steps] for i in range(steps) ]
#    
#    return predictions, originalAccs[key], time[key]
#

def main():
    experiments = {}
    is_windows = sys.platform.startswith('win')
    sep = '\\'

    if is_windows == False:
        sep = '/'

    path = os.getcwd()+sep+'data'+sep
    
    # SETTINGS
    sslClassifier = 'IF' # lp = label propagation, rf = random forests, cl = cluster and label, knn = k-nn, svm = svm
   #在这里要把算法换掉
    steps = 10
    poolSize = None
    isBatchMode = True # False = Stream
    isBinaryClassification = True
    isImbalanced = True
    externalResults = []
    # Load dataset
    dataValues, dataLabels, description = setup.loadiot(path, sep)
    
    # Only 5% of initial labeled data - Extreme verification latency scenario
    labeledData = int(0.1*len(dataLabels))

    # Static SSL
#    experiments[0] = Experiment(static_labelpropagation, 11, clfName=sslClassifier)
    
    # Sliding SSL
#    experiments[1] = Experiment(deslizante_LN, 11, clfName=sslClassifier)
    
    #Incremental SSL
#    experiments[2] = Experiment(incremental_LN, 11, clfName=sslClassifier)
    
    # Proposed Method 1 (AMANDA - Fixed core extraction percentage)
    experiments[3] = Experiment(amanda_fixed,11, 0.8, "kde", sslClassifier)

    # Proposed method 2 (AMANDA - Dynamic core extraction percentage)
#    experiments[4] = Experiment(amanda_dynamic, 11, None, "kde", sslClassifier)
    
    # Run experiments
    runExperiments.run(dataValues, dataLabels, description, isBinaryClassification, isImbalanced, 
                       experiments, steps, labeledData, isBatchMode, poolSize, externalResults)
    

if __name__ == "__main__":
    main()


# In[ ]:



