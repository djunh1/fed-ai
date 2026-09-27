from .base import BaseApi


class FraserFedApi(BaseApi):
    def __init__(self) -> None:
        super().__init__()

    def register_api(self, options) -> None:
        pass

    def fetch_api_data(self, options) -> None:
        pass


class FedGovApi(BaseApi):
    pass
