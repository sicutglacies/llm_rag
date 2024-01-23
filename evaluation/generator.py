import typing as t

from ragas.utils import load_as_json
from ragas.testset.utils import load_as_score

from ragas.testset import TestsetGenerator
from langchain.prompts import ChatPromptTemplate
from prompts import (
    ANSWER_FORMULATE,
    COMPRESS_QUESTION,
    CONDITIONAL_QUESTION,
    CONTEXT_FORMULATE,
    CONVERSATION_QUESTION,
    FILTER_QUESTION,
    MULTICONTEXT_QUESTION,
    REASONING_QUESTION,
    SCORE_CONTEXT,
    SEED_QUESTION,
)


class RussianTestGenerator(TestsetGenerator):
    def _filter_context(self, context: str) -> bool:
        """
        context: str
            The input context

        Checks if the context is has information worthy of framing a question
        """
        human_prompt = SCORE_CONTEXT.format(context=context)
        prompt = ChatPromptTemplate.from_messages([human_prompt])
        results = self.critic_llm.generate(prompts=[prompt])
        output = results.generations[0][0].text.strip()
        score = load_as_score(output)
        return score >= self.threshold
    
    def _seed_question(self, context: str) -> str:
        human_prompt = SEED_QUESTION.format(context=context)
        prompt = ChatPromptTemplate.from_messages([human_prompt])
        results = self.generator_llm.generate(prompts=[prompt])
        return results.generations[0][0].text.strip()

    def _filter_question(self, question: str) -> bool:
        human_prompt = FILTER_QUESTION.format(question=question)
        prompt = ChatPromptTemplate.from_messages([human_prompt])

        results = self.critic_llm.generate(prompts=[prompt])
        results = results.generations[0][0].text.strip()
        json_results = load_as_json(results)
        return json_results.get("verdict") != "No"

    def _reasoning_question(self, question: str, context: str) -> str:
        return self._qc_template(REASONING_QUESTION, question, context)

    def _condition_question(self, question: str, context: str) -> str:
        return self._qc_template(CONDITIONAL_QUESTION, question, context)

    def _multicontext_question(
        self, question: str, context1: str, context2: str
    ) -> str:
        human_prompt = MULTICONTEXT_QUESTION.format(
            question=question, context1=context1, context2=context2
        )
        prompt = ChatPromptTemplate.from_messages([human_prompt])
        results = self.generator_llm.generate(prompts=[prompt])
        return results.generations[0][0].text.strip()

    def _compress_question(self, question: str) -> str:
        return self._question_transformation(COMPRESS_QUESTION, question=question)

    def _conversational_question(self, question: str) -> str:
        return self._question_transformation(CONVERSATION_QUESTION, question=question)
    
    def _generate_answer(self, question: str, context: t.List[str]) -> t.List[str]:
        return [
            self._qc_template(ANSWER_FORMULATE, qstn, context[i])
            for i, qstn in enumerate(question.split("\n"))
        ]

    def _generate_context(self, question: str, text_chunk: str) -> t.List[str]:
        return [
            self._qc_template(CONTEXT_FORMULATE, qstn, text_chunk)
            for qstn in question.split("\n")
        ]
