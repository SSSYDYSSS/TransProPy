def print_results(fr, fre1, frequency, len_FName, FName, Fauc):
    print('Ranked features (start from higher rank): ', fr)
    print('Features and its frequency: ', fre1)
    print('Sorted features with frequency higher than 1: ', frequency)
    print('The count of AUC values greater than 0.5: ', len_FName)
    print('The list of feature names after ranking (AUC > 0.5): ', FName)
    print('The list of AUC values corresponding to the ranked feature names: ', Fauc)
