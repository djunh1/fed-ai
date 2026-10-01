from abc import ABC, abstractmethod

from llm_engineering.domain.base import NoSQLBaseDocument


class BaseApi(ABC):
    model: type[NoSQLBaseDocument]
