import os
import sys
import ast
import unicodedata
import pandas as pd
from tqdm import tqdm
from datasets import Dataset

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)

path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, path)

from app.rag.rag_engine import rag_chain_with_source, ensemble_retriever


synth_data = pd.read_csv('data/evaluation/synthetic_questions.csv')

answers = []
contexts = []

for query in tqdm(synth_data.question.tolist(), desc='Generation answers'):
    answers.append(rag_chain_with_source.invoke(query)['answer'])
    contexts.append([unicodedata.normalize('NFKD', docs.page_content) for docs in ensemble_retriever.get_relevant_documents(query)])

ground_truth = list(map(ast.literal_eval, synth_data.ground_truth.tolist()))

data = {
    "question": synth_data.question.tolist(),
    "answer": answers,
    "contexts": contexts,
    "ground_truths": ground_truth
}

dataset = Dataset.from_dict(data)

result = evaluate(
    dataset = dataset, 
    metrics=[
        context_precision,
        context_recall,
        faithfulness,
        answer_relevancy,
    ],
)

result_df = result.to_pandas()

result_df.to_excel('data/evaluation/results.xlsx', index=False)
