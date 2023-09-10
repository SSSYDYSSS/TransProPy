import pandas as pd

def filter_samples(threshold, data_path='../data/gene_tpm.csv'):
    """
    Remove samples with high zero expression.
    -----------------------------------------
    Parameters
    data_path: string
        For example: '../data/gene_tpm.csv'
        Please note: The input data matrix should have genes as rows and samples as columns.
    threshold: float
        For example: 0.9
        The set threshold indicates the proportion of non-zero value samples to all samples in each feature.
    --------------------------------------------------------------------------------------------------------
    Return
        X: pandas.core.frame.DataFrame
    -----------------------------------
    """
    data = pd.read_csv(data_path, index_col=0, header=0)
    # 计算每一行的非零值数量
    non_zero_counts = data.astype(bool).sum(axis=1)
    # 设置阈值，表示大多数基因表达为0的比例
    # threshold = 0.9
    # 根据阈值过滤行
    X = data[non_zero_counts / data.shape[1] > threshold]
    # 输出结果
    return X
