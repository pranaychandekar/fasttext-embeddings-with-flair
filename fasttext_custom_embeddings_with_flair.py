"""
Flair framework has recently made an enhancement to load custom embeddings in Flair through 'WordEmbeddings' class.
However, this has a limitation. Internally, Flair uses gensim to load these custom embeddings. So, if we wish to
import our custom embeddings in gensim then we have to convert it to gensim format and then use it.
While doing so we lose the ability of fasttext to approximate vector for out of vocabulary words using the sub-word
information. As a solution to this problem, I present you this script.

The embedding object defined using this script can be used in the same way as any other Flair embedding objects.
We can use the 'embed' function to add embeddings to tokens in the 'Sentence' object. We can use this along with other
embeddings in 'StackedEmbeddings' and 'DocumentEmbeddings'.

Please install packages from requirements.txt to use this script. You can then import this script to your project
where you wish to use custom fasttext embeddings with Flair.
"""

import numpy as np
import gensim
import torch
from typing import List

from flair.embeddings import TokenEmbeddings
from flair.file_utils import cached_path
from flair.data import Sentence

from pathlib import Path
import fasttext as ft


class FastTextEmbeddings(TokenEmbeddings):
    """FastText Embeddings to use with Flair framework"""

    def __init__(self, embeddings: str = None, use_local: bool = True, use_gensim: bool = False, field: str = None):
        """
        Initializes fasttext word embeddings. Constructor downloads required embedding file and stores in cache
        if use_local is False.

        :param embeddings: path to your embeddings '.bin' file
        :param use_local: set this to False if you are using embeddings from a remote source
        :param use_gensim: set this to true if your fasttext embedding is trained with fasttext version below 0.9.1
        """

        if embeddings is None:
            raise ValueError(f'The given embeddings "{embeddings}" is not available or is not a valid path.')

        cache_dir = Path("embeddings")
        if use_local:
            if not Path(embeddings).exists():
                raise ValueError(f'The given embeddings "{embeddings}" is not available or is not a valid path.')
        else:
            embeddings = cached_path(f"{embeddings}", cache_dir=cache_dir)

        self.embeddings = embeddings

        self.name: str = str(embeddings)
        self.static_embeddings = True

        self.use_gensim = use_gensim
        self.field = field

        if use_gensim:
            self.precomputed_word_embeddings = gensim.models.FastText.load_fasttext_format(str(embeddings))
            self.__embedding_length: int = self.precomputed_word_embeddings.vector_size
        else:
            self.precomputed_word_embeddings = ft.load_model(str(embeddings))
            self.__embedding_length: int = self.precomputed_word_embeddings.get_dimension()

        super().__init__()

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length

    def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:
        for i, sentence in enumerate(sentences):
            for token, token_idx in zip(sentence.tokens, range(len(sentence.tokens))):
                if "field" not in self.__dict__ or self.field is None:
                    word = token.text
                else:
                    word = token.get_tag(self.field).value

                try:
                    word_embedding = self.precomputed_word_embeddings[word]
                except:
                    word_embedding = np.zeros(self.embedding_length, dtype="float")

                word_embedding = torch.FloatTensor(word_embedding)
                token.set_embedding(self.name, word_embedding)

        return sentences

    def __str__(self):
        return self.name

    def extra_repr(self):
        return f"'{self.embeddings}'"
