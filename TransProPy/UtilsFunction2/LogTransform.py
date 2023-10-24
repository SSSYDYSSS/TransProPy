import numpy as np

def log_transform(data):
    """
    Evaluate and potentially apply log2 transformation to data.
    -This function checks data against a set of criteria to determine if a log2 transformation is needed, applying the transformation if necessary.
    -----------------------------------------------------------------------------------------------------------------------------------------------
    Parameters:
    -data (np.ndarray): A numerical numpy array.
    ------------------------------------------
    Returns:
    -np.ndarray: The original data or the data transformed with log2.
    -----------------------------------------------------------------
    """
    # Calculate quantiles
    qx = np.quantile(data, [0., 0.25, 0.5, 0.75, 0.99, 1.0])

    # Define conditions for log transformation
    LogC = (qx[4] > 100) or \
           (qx[5] - qx[0] > 50 and qx[1] > 0) or \
           (qx[1] > 0 and qx[1] < 1 and qx[3] > 1 and qx[3] < 2)

    # Apply log transformation based on conditions
    if LogC:
        data[data <= 0] = np.NaN  # Use NaN for non-applicable data
        result = np.log2(data)
        print("log2 transform finished")
    else:
        result = data
        print("log2 transform not needed")

    return result