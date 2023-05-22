# ORDER MATTERS! THIS SHOULD EXECUTE FIRST.
import os
import sys
from pathlib import Path
path_root      = Path(__file__).parent.absolute()
(path_root, _) = os.path.split(path_root)
sys.path.append(str(path_root))
# ______________________________________________________

import pandas as pd
import numpy as  np
from security.cryptosystem.liu import Liu

import scripts.routines as Routines
import scripts.declarations as Declarations
from logger.Logger import create_logger

from concurrent.futures import ThreadPoolExecutor,wait

import multiprocessing

# EXPERIMENT_ITERATIONS    = kwargs.get("experiment_iterations",2)
# SINK_PATH                = kwargs.get("sink_path","/test/sink")
# SOURCE_PATH              = kwargs.get("source_path","/test/source")
# TRACE_FILENAME           = kwargs.get("trace_filename","trace1")
# TRACE_EXT                = kwargs.get("trace_ext","csv")
# DATASET_DESCRIPTION_FILENAME = kwargs.get("dataset_description_filename","datasets_desc")

BATCH_ID                     = os.environ.get("BATCH_ID",1)
EXPERIMENT_ITERATIONS        = os.environ.get("EXPERIMENT_ITERATIONS",31)
SINK_PATH                    = os.environ.get("SINK_PATH","/test/sink")
SOURCE_PATH                  = os.environ.get("SOURCE_PATH","/test/source")
TRACE_FILENAME               = os.environ.get("TRACE_FILENAME","trace_batch{}".format(BATCH_ID))
TRACE_EXTENSION              = os.environ.get("TRACE_EXTENSION","csv")
M                            = os.environ.get("M",3)
MAX_WORKERS                  = os.environ.get("MAX_WORKERS",1)
DATASET_DESCRIPTION_FILENAME = os.environ.get("DATASET_DESCRIPTION_FILENAME","datasets_desc")
TRACE_PATH                   = "{}/{}.{}".format(SOURCE_PATH,TRACE_FILENAME,TRACE_EXTENSION)
L                            = create_logger(
    name         = "experiment-logger",
    LOG_PATH     = SINK_PATH,
    LOG_FILENAME = "experiments_batch{}".format(BATCH_ID),
    add_error_log = False
)

def experiment(*args,**kwargs):
    try:
        eor = Declarations.ExperimentOutputRow(**kwargs)
        eor = Routines.clustering(**kwargs,experiment_output_row = eor)
        eor = Routines.validation_indexes(**kwargs, experiment_output_row = eor)
        L.info(eor)
        return
    except Exception as e:
        print(e)
        raise e

def run_experiments(**kwargs):
    TRACE                    = pd.read_csv(TRACE_PATH)
    LIU                      = Liu()
    DATASET_DESCRIPTION_PATH = "{}/{}.{}".format(SOURCE_PATH,DATASET_DESCRIPTION_FILENAME,"csv")
    DATASET_DESCRIPTION_DF   = pd.read_csv(DATASET_DESCRIPTION_PATH)
    # ________________________________________________________________________________________
    
    futures = []
    try:
        with ThreadPoolExecutor(max_workers = MAX_WORKERS) as executor:
            for index,row in TRACE.iterrows():
                dataset_id             = row["DATASET_ID"]
                _round                 = row["BATCH"]
                
                allowed_algorithms     = list( 
                    map(
                        bool,
                        map(int,[row["KMEANS"],row["SKMEANS"],row["DBSKMEANS"]])
                    )
                )
                allowed_algorithms_inv = np.invert(np.array([allowed_algorithms])) 
                ALGORITHMS = ["KMEANS","SKMEANS","DBSKMEANS"]
                FILTERED_ALGORITHMS = np.ma.masked_array(ALGORITHMS,mask = allowed_algorithms_inv).compressed()
                # 
                L.debug("TRACE index={} dataset_id={} round={} algorithms={}".format(index,dataset_id,_round,FILTERED_ALGORITHMS))
                # _____________________________________________________________________
                fullname               = "{}.{}".format(dataset_id,TRACE_EXTENSION)
                BATCH_FOLDER           = "batch{}".format(BATCH_ID)
                dataset_df_path        = "{}/datasets/{}/{}".format(SOURCE_PATH,BATCH_FOLDER,fullname)
                counter_df_path        = "{}/datasets/{}/{}.{}".format(SOURCE_PATH,BATCH_FOLDER,dataset_id+"_counter",TRACE_EXTENSION)
                target_df_path         = "{}/datasets/{}/{}.{}".format(SOURCE_PATH,BATCH_FOLDER,dataset_id+"_target",TRACE_EXTENSION)
                dataset_df             = pd.read_csv(dataset_df_path)
                rows                   = dataset_df.shape[0]
                columns                = dataset_df.shape[1]
                counter_df             = pd.read_csv(counter_df_path)
                target_df              = pd.read_csv(target_df_path)
                target                 = target_df.values.flatten()
                k                      = counter_df.shape[0]
                plaintext_matrix       = dataset_df.values.tolist()
                url                    = DATASET_DESCRIPTION_DF[DATASET_DESCRIPTION_DF["DATASET_ID"]==dataset_id]["URL"].values[0]    
                

                for experiment_index in range(EXPERIMENT_ITERATIONS):
                    
                    L.debug("CLUSTERING_INIT experiment_index = {} algorithms={} rows={} cols={}".format(experiment_index,FILTERED_ALGORITHMS,rows,columns ))
                    # print("EXPERIMENT[{}]".format(experiment_index))
                    future = None
                    if(allowed_algorithms[0]):
                        future = executor.submit(
                            experiment,
                            dataset_id       = dataset_id,
                            k                = k,
                            url              = url ,
                            rows             = rows,
                            columns          = columns,
                            algorithm        = "KMEANS",
                            plaintext_matrix = plaintext_matrix,
                            target           = target,
                            round            = _round,
                            logger           = L,
                            experiment_index = experiment_index
                        )
                        
                    if(allowed_algorithms[1]):
                        future = executor.submit(
                            experiment,
                            dataset_id = dataset_id,
                            k          = k,
                            url        = url ,
                            rows       = rows,
                            columns    = columns,
                            algorithm  = "SKMEANS",
                            plaintext_matrix      = plaintext_matrix,
                            target                = target,
                            LIU                   = LIU,
                            round                 = _round,
                            logger                = L ,
                            experiment_index = experiment_index
                        
                        )
                    futures.append(future)
                    # print("_"*100)
                # wait(futures)
                print("_"*40)
        wait(futures)
    except Exception as e:
        L.erorr("dataset_id={} description = {}".format(dataset_id,e) )


if __name__  == "__main__":
    
    run_experiments(
        experiment_iterations        = EXPERIMENT_ITERATIONS,
        sink_path                    = SINK_PATH,
        source_path                  = SOURCE_PATH,
        trace_filename               = TRACE_FILENAME, 
        trace_ext                    = TRACE_EXTENSION,
        m                            = M,
        MAX_WORKERS                  = MAX_WORKERS,# Number of threads to perform parallel tasks
        dataset_description_filename = DATASET_DESCRIPTION_FILENAME
    )