import sqlite3
import pandas as pd
import numpy as np
import multiprocessing
from multiprocessing_logging import install_mp_handler
import logging
from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer
import transformers
# setting number of GPU
GPU_N = 2

type_entities = ['BC4CHEMD', 'BC5CDR-chem', 'BC5CDR-disease', 'JNLPBA', 'NCBI-disease', 'linnaeus', 's800', 'BC2GM']

# extracting entities from sections
def extract_entity(work_data):

    n = 0 #defining log number
    device = work_data[0]
    sections = work_data[1]
    df_entity = pd.DataFrame()
    logging.debug("processing data (%s) on GPU_(%s)" %  (len(sections), device))
    #defining pipeline

    model = AutoModelForTokenClassification.from_pretrained(fine_tuned_model_dir)
    tokenizer = AutoTokenizer.from_pretrained(fine_tuned_model_dir)
    ner_model = pipeline('ner', model=model, tokenizer=tokenizer, device=device, grouped_entities=True)

    for index, row in sections.iterrows():
        n = n + 1
        id = row['Id']
        sequence = row['Text']
        items = ner_model(sequence)
        if n % 1000 == 0:
            logging.debug("currently finished senction id is %s-%s on GPU_(%s)" % (id, n, device))
        if len(items) > 0:
            entities = []
            for item in items:
                word = item['word']
                entity_group = item['entity_group']
                if entity_group == 'B':
                    if word.startswith('##'):
                        word = word.replace('##','')
                if entity_group == 'I':
                    if word.startswith('##'):
                        if len(entities) == 0:
                            word = word.replace('##','')
                        else:
                            word = entities[len(entities)-1] + word.replace('##','')
                            entities.pop()
                    else:
                        if len(entities) > 0:
                            word = entities[len(entities)-1] +' ' + word
                            entities.pop()
                entities.append(word)
            df = pd.DataFrame (entities,columns=['entity'])
            df['n'] = df.index + 1
            df['section_id'] = id
            df_entity = df_entity.append(df)
    return df_entity

if __name__ == '__main__':

    transformers.logging.set_verbosity_debug()
    logging.basicConfig(format='%(levelname)s: %(asctime)s - %(process)s - %(message)s',
                        filename='1_ner.log', filemode='w', level=logging.DEBUG)
    install_mp_handler()
    logging.debug('reading data from sqlite file')
    db = sqlite3.connect("/nfs/workspaces/datasets/covid-19/articles.sqlite")
    sections = pd.read_sql_query("SELECT * FROM sections WHERE Name='ABSTRACT'", db)
    logging.debug("the total sentences are (%s)" % (len(sections)))
    df_split = np.array_split(sections,GPU_N)
    for type_ in type_entities:
        logging.debug("starting to extract type %s entity" % (type_))
        fine_tuned_model_dir = "/nfs/storages/bio_corpus/ner/" + type_ + "/ner_outputs"
        rusult_dir = "/nfs/storages/bio_corpus/ner/" + type_ + "/all_entity.csv"
        work = []
        for n in range(0, GPU_N):
            work.append([n, df_split[n]])
        p = multiprocessing.Pool(GPU_N)
        all_entity = pd.concat(p.map(extract_entity, work))
        all_entity.to_csv(rusult_dir, index = False)
        p.close()
        p.join()


