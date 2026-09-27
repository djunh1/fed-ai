from abc import ABC, abstractmethod

from llm_engineering.domain.base import NoSQLBaseDocument


class BaseApi(ABC):
    model: type[NoSQLBaseDocument]

    @abstractmethod
    def register_api(self, link: str, **kwargs) -> None: ...

    @abstractmethod
    def fetch_api_data(self, link: str, **kwargs) -> None: ...
