from qlib.data.dataset.handler import DataHandler, DataHandlerLP
from qlib.data.dataset.loader import QlibDataLoader
from factor_loader import get_generated_factor_expressions


class GenFactorHandler(DataHandlerLP):
    def __init__(
        self,
        instruments="all",
        start_time=None,
        end_time=None,
        data_loader=None,
        analysis_path=None,
        **kwargs,
    ):
        self.analysis_path = analysis_path
        # Capture label from kwargs
        self._label = kwargs.pop(
            "label", ["Ref($close, -5) / $close - 1", "Ref($close, -5) / $close - 1"]
        )

        # Instantiate defaults if missing
        if data_loader is None:
            features = self.get_feature_config()
            # Wrap in dictionary to ensure 'feature' group exists in MultiIndex columns
            # This satisfies processors expecting fields_group="feature"
            # And also include label?
            # DataHandler usually loads label separately via get_label_config?
            # No, DataHandler.setup_data calls self._load_data.
            # It loads features and labels.
            # DataHandlerLP expects data_loader to load fields correctly.

            # If we only provide 'feature', where does 'label' come from?
            # DataHandler calls self.get_label_config()?
            # No, DataHandler.init calls self.data_loader.load(instruments, start_time, end_time, self._config)?

            # Actually, DataHandler logic:
            # It combines get_feature_config and get_label_config into self._config?
            # Or data_loader does it?

            # If we create `data_loader` manually, we assume DataHandler uses it.
            # DataHandler DOES NOT construct config from get_feature_config if data_loader is provided?
            # It might?

            # Let's trust that providing {'feature': features} helps.
            # But what about LABEL?
            # If we don't load label here, learn_processors might fail (DropnaLabel).
            # We should include label in the loader config if explicit.

            loader_config = {"feature": features}
            # Add label if we know it?
            # self._label is captured.
            if self._label:
                loader_config["label"] = self._label

            data_loader = QlibDataLoader(config=loader_config)

        super().__init__(
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            data_loader=data_loader,
            **kwargs,
        )

    def get_feature_config(self):
        # Load generated factors dynamically
        gen_factors = get_generated_factor_expressions(analysis_path=self.analysis_path)
        if not gen_factors:
            print(
                "GenFactorHandler: No generated factors found! Falling back to basic factors."
            )
            return ["$close", "$volume", "$open", "$high", "$low"]

        print(f"GenFactorHandler: Loading {len(gen_factors)} factors.")
        return list(gen_factors.values())

    def get_label_config(self):
        return self._label
