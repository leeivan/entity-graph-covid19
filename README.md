# entity-graph-covid19
This project's target is extracting entities and relations among them, then building a graph. 
We used COVID-19 medical literatures, which is from the [COVID-19 Open Research Dataset Challenge (CORD-19)](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge).
## methodology
During process of extracting entities and relations, our implementation took advantage of pre-trained model [BioBERT](https://github.com/dmis-lab/biobert)，which a biomedical language representation model designed for biomedical text mining tasks such as biomedical named entity recognition, relation extraction, question answering, etc. We fininshed fine-tuning for 8 datasets(BC2GM, BC4CHEMD, BC5CDR-chem, BC5CDR-disease, JNLPBA, NCBI-disease, linnaeus, s800) on biomedical named entity recognition, and for 2 datasets(GAD, euadr) on biomedical relation extraction. Our code recalled the [Huggingface Tranformers API](https://github.com/huggingface/transformers) to excute and finish those series of process.
After, we used [Spark GraphX](https://github.com/apache/spark) to load the extracting results and build COVID-19 entity graph.

