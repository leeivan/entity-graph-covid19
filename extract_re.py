import sqlite3
import pandas as pd
import numpy as np
import multiprocessing
from multiprocessing_logging import install_mp_handler
import logging
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import transformers
# setting number of GPU
GPU_N = 2

type_res = ['GAD', 'euadr']

# extracting entities from sections
def extract_re(work_data):

    n = 0 #defining log number
    device = work_data[0]
    sections = work_data[1]
    df_re = pd.DataFrame()
    logging.debug("processing data (%s) on GPU_(%s)" %  (len(sections), device))
    #defining pipeline
    model = AutoModelForSequenceClassification.from_pretrained(fine_tuned_model_dir)
    tokenizer = AutoTokenizer.from_pretrained(fine_tuned_model_dir)
    re_model = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer, device=device)

    for index, row in sections.iterrows():
        n = n + 1
        id = row['Id']
        sequence = row['Text']
        item = re_model(sequence[:512])
        if n % 1000 == 0:
            logging.debug("currently finished senction id is %s-%s on GPU_(%s)" % (id, n, device))
        if item[0]['label'] == 'LABEL_1':
            re = []
            re.append(id)
            df = pd.DataFrame(re, columns=['section_id'])
            df_re = df_re.append(df)
    return df_re

if __name__ == '__main__':

    transformers.logging.set_verbosity_debug()
    logging.basicConfig(format='%(levelname)s: %(asctime)s - %(process)s - %(message)s',
                        filename='1_re.log', filemode='w', level=logging.DEBUG)
    install_mp_handler()
    logging.debug('reading data from sqlite file')
    db = sqlite3.connect("/nfs/workspaces/datasets/covid-19/articles.sqlite")
    sections = pd.read_sql_query("SELECT * FROM sections WHERE Name='ABSTRACT'", db)
    logging.debug("the total sentences are (%s)" % (len(sections)))
    df_split = np.array_split(sections,GPU_N)
    for type_ in type_res:
        logging.debug("starting to extract type %s relations" % (type_))
        fine_tuned_model_dir = "/nfs/storages/bio_corpus/re_all_in_one/" + type_ + "/re_outputs"
        result_dir = "/nfs/storages/bio_corpus/re_all_in_one/" + type_ + "/all_relations.csv"
        work = []
        for n in range(0, GPU_N):
            work.append([n, df_split[n]])
        p = multiprocessing.Pool(GPU_N)
        all_re = pd.concat(p.map(extract_re, work))
        all_re.to_csv(result_dir, index=False)
        p.close()
        p.join()
