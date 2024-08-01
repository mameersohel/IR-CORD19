#initiate pyterrier to start
import pyterrier as pt
if not pt.started():
    pt.init()

#load dataset
dataset = pt.datasets.get_dataset("irds:beir/trec-covid")

import os
pt_index_path = './indices/cord19'

if not os.path.exists(pt_index_path + "/data.properties"):
    indexer = pt.index.IterDictIndexer(pt_index_path, blocks=True)
    index_ref = indexer.index(dataset.get_corpus_iter(),
                              fields=['title', 'text'],
                              meta=('docno',))

else:
    index_ref = pt.IndexRef.of(pt_index_path + "/data.properties")

index = pt.IndexFactory.of(index_ref)
#print stats of data
print(index.getCollectionStatistics())

#bm25 model
bm25 = pt.BatchRetrieve(index_ref, wmodel="BM25")
topics = dataset.get_topics('query')

#get topics
res = bm25.transform(topics)
print(res)

#get qrels for each topic
qrels = dataset.get_qrels()

# Re-ranking model - random forests from scikit-learn, a pointwise regression tree technique
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

#training, testing, validation split of data
tr_topics, test_topics = train_test_split(topics, test_size=15)
train_topics, valid_topics = train_test_split(tr_topics, test_size=5)

#metadata keys that are present ('doc_id', 'text', 'title', 'url', 'pubmed_id')
#below uses BM25 for a query, title, url, and coordinate match
ltr_feats1 = (bm25) >> pt.text.get_text(dataset, ["title", "url"]) >> (
    pt.transformer.IdentityTransformer()
    ** # score of text for query 'coronavirus covid'
    (pt.apply.query(lambda row: 'coronavirus covid') >> bm25)
    ** # score of title
    (pt.text.scorer(body_attr="title", takes='docs', wmodel='BM25'))
    ** # has url
    (pt.apply.doc_score(lambda row: int(row["url"] is not None and len(row["url"]) > 0) ))
    ** # abstract coordinate match
    pt.BatchRetrieve(index, wmodel="CoordinateMatch")
)

#feature sets
fnames=["BM25", 'coronavirus covid', 'title', "url", "CoordinateMatch"]

#call pointwise regression tree technique
rf = RandomForestRegressor(n_estimators=400, verbose=1, random_state=42, n_jobs=2)

#send to pipe and fit
rf_pipe = ltr_feats1 >> pt.ltr.apply_learned_model(rf)
rf_pipe.fit(train_topics, dataset.get_qrels())

#evaluate experiment
eval_metrics=['P_10', 'ndcg_cut_10', 'map']
exp_res = pt.Experiment(
    [bm25, rf_pipe],
    topics,
    qrels,
    names=["BM25", "BM25 + RF(7f)"],
    eval_metrics=eval_metrics,
)

#print results
print(exp_res)
