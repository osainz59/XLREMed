from transformers import AutoTokenizer
import logging

logger = logging.getLogger(__name__)


class Tokenizer(AutoTokenizer):

    def __init__(self, *args, **kwargs):
        super(Tokenizer, self).__init__(*args, **kwargs)

    @classmethod
    def from_pretrained(cls, *inputs, **kwargs):
        if len(inputs):
            pretrained_model_name = inputs[0]
        obj = super().from_pretrained(*inputs, **kwargs)
        obj.pretrained_model_name_or_path = pretrained_model_name

        def add_tokens(self, new_tokens):
            if not 'roberta' in self.pretrained_model_name_or_path:
                if not new_tokens:
                    return 0

                if not isinstance(new_tokens, list):
                    new_tokens = [new_tokens]

                to_add_tokens = []
                for token in new_tokens:
                    assert isinstance(token, str)
                    if self.init_kwargs.get("do_lower_case", False) and token not in self.all_special_tokens:
                        token = token.lower()
                    if (
                        token != self.unk_token
                        and self.convert_tokens_to_ids(token) == self.convert_tokens_to_ids(self.unk_token)
                        and token not in to_add_tokens
                    ):
                        to_add_tokens.append(token)
                        logger.info("Adding %s to the vocabulary", token)

                added_tok_encoder = dict((tok, len(self) + i) for i, tok in enumerate(to_add_tokens))
                added_tok_decoder = {v: k for k, v in added_tok_encoder.items()}
                self.added_tokens_encoder.update(added_tok_encoder)
                self.unique_added_tokens_encoder = set(self.added_tokens_encoder.keys()).union(set(self.all_special_tokens))
                self.added_tokens_decoder.update(added_tok_decoder)

                return len(to_add_tokens)
            else:
                if not new_tokens:
                    return 0

                if not isinstance(new_tokens, list):
                    new_tokens = [new_tokens]

                to_add_tokens = []
                for token in new_tokens:
                    assert isinstance(token, str)
                    if self.init_kwargs.get("do_lower_case", False) and token not in self.all_special_tokens:
                        token = token.lower()
                    if (
                        token != self.unk_token and token not in to_add_tokens
                    ):
                        to_add_tokens.append(token)
                        logger.info("Adding %s to the vocabulary", token)

                added_tok_encoder = dict((tok, len(self) + i) for i, tok in enumerate(to_add_tokens))
                added_tok_decoder = {v: k for k, v in added_tok_encoder.items()}
                self.added_tokens_encoder.update(added_tok_encoder)
                self.unique_added_tokens_encoder = set(self.added_tokens_encoder.keys()).union(set(self.all_special_tokens))
                self.added_tokens_decoder.update(added_tok_decoder)

                return len(to_add_tokens)

        obj.add_tokens = type(obj.add_tokens)(add_tokens, obj)

        return obj
