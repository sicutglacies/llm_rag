from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from ragas.llms import LangchainLLM
from generator import RussianTestGenerator

import dotenv
dotenv.load_dotenv()

loader = TextLoader('data/docs/bank_name_docs.md')
docs = loader.load()


# Add custom llms and embeddings
generator_llm = LangchainLLM(llm=ChatOpenAI(model="gpt-3.5-turbo"))
critic_llm = LangchainLLM(llm=ChatOpenAI(model="gpt-4"))
embeddings_model = OpenAIEmbeddings()

# Change resulting question type distribution
testset_distribution = {
    "simple": 0.25,
    "reasoning": 0.25,
    "multi_context": 0.25,
    "conditional": 0.25,
}

test_generator = RussianTestGenerator(
    generator_llm=generator_llm,
    critic_llm=critic_llm,
    embeddings_model=embeddings_model,
    testset_distribution=testset_distribution,
)

testset = test_generator.generate(docs, test_size=15)


test_df = testset.to_pandas()
test_df.to_csv('data/evaluation/synthetic_questions.csv', index=False)
