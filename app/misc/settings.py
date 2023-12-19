from typing import Final
from langchain.chains.query_constructor.base import AttributeInfo


class Settings:
    HEADERS_TO_SPLIT: Final = [
        ("#", "Заголовок темы"),
        ("##", "Подзаголовок темы"),
        ("###", "Заголовок абзаца")
    ]

    METADATA_INFO: Final = [
        AttributeInfo(
            name="Заголовок",
            description="Часть документа, откуда был взят текст",
            type="string or list[string]",
        ),
    ]

    CONTENT_DESCRIPTION: Final = "Описание банковских продуктов"

    PROMT_TEMPLATE: Final = """
    Вы помощник по продуктам банка Tinkoff и отвечаете на вопросы клиентов.
    Используйте фрагменты полученного контекста, чтобы ответить на вопрос.
    Если вы не знаете ответа, то скажите, что не знаете, не придумывайте ответ.
    Используйте максимум три предложения и будьте краткими.\n
    Вопрос: {question} \n
    Контекст: {context} \n
    Ответ:
    """

    BM25_K: Final = 2
    MMR_K: Final = 2
    MMR_FETCH_K: Final = 5
