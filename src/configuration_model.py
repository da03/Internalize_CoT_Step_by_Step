from transformers import PretrainedConfig

class ImplicitModelConfig(PretrainedConfig):
    def __init__(
        self,
        base_model='gpt2',
        **kwargs,
    ):
        self.base_model = base_model
        self.tokenizer_name = base_model
        super().__init__(**kwargs)

