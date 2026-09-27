import re
from urllib.parse import urlparse

from .base import BaseApi
from .fed import FraserFedApi


class ApiDispatcher:
    def __init__(self) -> None:
        self._crawlers = {}

    @classmethod
    def build(cls) -> "ApiDispatcher":
        dispatcher = cls()

        return dispatcher

    def register_fraser(self) -> "ApiDispatcher":
        self.register("https://fraser.stlouisfed.org", FraserFedApi)
        return self
