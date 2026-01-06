from qlib.data.dataset.handler import DataHandler, DataHandlerLP
from qlib.data.dataset.loader import QlibDataLoader
from qlib.contrib.data.handler import Alpha158
from factor_loader import get_generated_factor_expressions
import json
import os


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
            loader_config = {"feature": features}
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


class CombinedFactorHandler(Alpha158):
    def __init__(self, analysis_path=None, **kwargs):
        self.analysis_path = analysis_path
        super().__init__(**kwargs)

    def get_feature_config(self):
        # Get Alpha158 factors (tuple of exprs, names)
        alpha158_factors = super().get_feature_config()
        if isinstance(alpha158_factors, tuple):
            alpha_exprs, alpha_names = alpha158_factors
        else:
            # Fallback if it returns list
            alpha_exprs, alpha_names = alpha158_factors, alpha158_factors

        # Get generated factors
        gen_factors_dict = get_generated_factor_expressions(
            analysis_path=self.analysis_path
        )
        gen_exprs = list(gen_factors_dict.values()) if gen_factors_dict else []
        gen_names = list(gen_factors_dict.keys()) if gen_factors_dict else []

        print(
            f"CombinedFactorHandler: Loading {len(alpha_exprs)} Alpha158 factors and {len(gen_exprs)} generated factors."
        )

        # Combine and Deduplicate
        final_exprs = list(alpha_exprs)
        final_names = list(alpha_names)
        existing_names = set(final_names)

        duplicates = []
        for i, name in enumerate(gen_names):
            if name in existing_names:
                duplicates.append(name)
            else:
                final_exprs.append(gen_exprs[i])
                final_names.append(name)
                existing_names.add(name)

        if duplicates:
            print(
                f"CombinedFactorHandler: Skipped {len(duplicates)} duplicate factors: {duplicates}"
            )

        return (final_exprs, final_names)


class LowCorrFactorHandler(GenFactorHandler):
    def __init__(self, low_corr_path="low_corr_factors.json", **kwargs):
        self.low_corr_path = low_corr_path
        super().__init__(**kwargs)

    def get_feature_config(self):
        # 1. Get RD-Agent Factors
        rd_factors_dict = get_generated_factor_expressions(
            analysis_path=self.analysis_path
        )
        rd_exprs = list(rd_factors_dict.values()) if rd_factors_dict else []
        rd_names = list(rd_factors_dict.keys()) if rd_factors_dict else []

        # 2. Get Low-Corr Alpha158 Factors
        low_corr_factors = {}
        if os.path.exists(self.low_corr_path):
            with open(self.low_corr_path, "r") as f:
                low_corr_factors = json.load(f)
        else:
            print(f"LowCorrFactorHandler: {self.low_corr_path} not found!")

        lc_names = list(low_corr_factors.keys())
        lc_exprs = list(low_corr_factors.values())

        print(
            f"LowCorrFactorHandler: Loading {len(rd_exprs)} RD-Agent factors and {len(lc_exprs)} Low-Corr Alpha158 factors."
        )

        return (rd_exprs + lc_exprs, rd_names + lc_names)
