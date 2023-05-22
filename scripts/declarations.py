class ExperimentOutputRow(object):
    def __init__(self,**kwargs):
        self.dataset_id                  = kwargs.get("dataset_id","")
        self.algorithm                   = kwargs.get("algorithm","SKMEANS")
        self.round                       = kwargs.get("round",1)
        self.row                         = kwargs.get("rows",0)
        self.columns                     = kwargs.get("columns",0)
        self.k                           = kwargs.get("k",0)
        self.n_iterations                = kwargs.get("n_iterations",0)
        self.cipher_time                 = kwargs.get("cipher_time",0.0)
        # self.udm_calculation             = kwargs.get("udm_calculation",0.0)
        # self.udm_encryption              = kwargs.get("udm_encryption",0.0)
        self.udm_time                    = kwargs.get("udm_time",0.0)
        # self.clustering_time             = kwargs.get("clustering_time",0.0)
        self.service_time                = kwargs.get("service_time",0.0)
        self.response_time               = kwargs.get("response_time",0.0)
        self.silhouette_coefficient      = kwargs.get("silhouette_coefficient",0.0)
        self.davies_bouldin_index        = kwargs.get("davies_bouldin_index",0.0)
        self.calinski_harabaz_index      = kwargs.get("calinski_harabaz_index",0.0)
        self.dunn_index                  = kwargs.get("dunn_index",0.0)
        self.adjusted_mutual_information = kwargs.get("adjusted_mutual_information",0.0)
        self.fowlkes_mallows_index       = kwargs.get("fowlkes_mallows_index",0.0)
        self.adjusted_rand_index         = kwargs.get("adjusted_rand_index",0.0)
        self.jaccard_index               = kwargs.get("jaccard_index",0.0)
        self.url                         = kwargs.get("url","")
        self.experiment_index            = kwargs.get("experiment_index",0)
        self.metadata                    = {}

    def __str__(self):
        filtered_values = list( filter(lambda x: not x[0] in ["metadata","url"] ,self.__dict__.items() ))
        filtered_values = list(map(lambda x: str(x[1]), filtered_values))
        # return "<SHANEL>"
        return ",".join(filtered_values)
            