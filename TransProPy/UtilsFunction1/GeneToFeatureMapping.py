def gene_map_feature(gene_names, ranked_features):
    """
    gene map feature.
    ------------------------
    Parameters
    gene_names: list
        For example: ['GeneA', 'GeneB', 'GeneC', 'GeneD', 'GeneE']
        containing strings
    ranked_features: list
        For example: [2, 0, 1]
        containing integers
    -----------------------
    Return
        gene_to_feature_mapping: dictionary
        gene_to_feature_mapping is a Python dictionary type. It is used to map gene names to their corresponding feature (or ranked feature) names.
    -----------------------------------------------------------------------------------------------------------------------------------------------
    """
    gene_to_feature_mapping = {}

    for feature_index in ranked_features:
        if 0 <= feature_index < len(gene_names):
            gene_name = gene_names[feature_index]
            gene_to_feature_mapping[gene_name] = feature_index
        else:
            print(f"Invalid feature index: {feature_index}")

    return gene_to_feature_mapping