"""
Pluggable text encoders for Hopfield Memory.

Provides an abstract Encoder interface and four implementations at
different quality/dependency trade-off points. HopfieldMemory accepts
any Encoder instance via its constructor.

Use `auto_encoder()` to get the best available encoder for the current
environment without manual dependency checking.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import hashlib
import numpy as np
import warnings


def _stable_seed(text: str) -> int:
    """Derive a deterministic 64-bit seed from a string, independent of PYTHONHASHSEED."""
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little")


class Encoder(ABC):
    """Abstract base class for text-to-vector encoders."""

    @property
    @abstractmethod
    def dim(self) -> int:
        """Dimensionality of output vectors."""
        ...

    @abstractmethod
    def encode(self, text: str) -> np.ndarray:
        """Encode a single text string into a vector.

        Returns a unit-norm vector for non-empty text. Empty text returns
        the zero vector (used by the sentinel pattern).
        """
        ...

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """Encode multiple texts. Returns (len(texts), dim) array.

        Default implementation loops; subclasses may override for efficiency.
        """
        return np.vstack([self.encode(t) for t in texts])


class RandomIndexEncoder(Encoder):
    """Bag-of-words with deterministic random word vectors.

    Zero external dependencies. Low semantic quality -- words must match
    exactly for any similarity signal. Useful as a fallback and for testing.
    """

    def __init__(self, dim: int = 512):
        self._dim = dim
        self._cache: Dict[str, np.ndarray] = {}

    @property
    def dim(self) -> int:
        return self._dim

    def _word_vector(self, word: str) -> np.ndarray:
        if word not in self._cache:
            rng = np.random.default_rng(_stable_seed(word))
            self._cache[word] = rng.standard_normal(self._dim)
        return self._cache[word]

    def encode(self, text: str) -> np.ndarray:
        tokens = text.lower().replace(",", " ").replace(".", " ").split()
        if not tokens:
            return np.zeros(self._dim)
        vec = np.zeros(self._dim)
        for t in tokens:
            vec += self._word_vector(t)
        norm = np.linalg.norm(vec)
        if norm > 1e-12:
            vec /= norm
        return vec


class TFIDFEncoder(Encoder):
    """TF-IDF encoder backed by scikit-learn.

    Medium quality. Captures term importance across a corpus but has no
    semantic understanding (synonyms get zero similarity). Requires
    fitting on a corpus before encoding queries.

    If unfitted, auto-fits on the first batch of texts seen via store().
    """

    def __init__(self, dim: int = 512):
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.decomposition import TruncatedSVD
        except ImportError:
            raise ImportError("TFIDFEncoder requires scikit-learn: pip install scikit-learn")

        self._dim = dim
        self._vectorizer = TfidfVectorizer(max_features=5000)
        self._svd = TruncatedSVD(n_components=dim)
        self._fitted = False
        self._corpus: List[str] = []

    @property
    def dim(self) -> int:
        return self._dim

    def fit(self, texts: List[str]) -> None:
        """Fit the TF-IDF + SVD pipeline on a corpus."""
        from sklearn.pipeline import make_pipeline
        self._corpus = list(texts)
        if len(self._corpus) < self._dim:
            effective_dim = max(2, len(self._corpus) - 1)
            self._svd.n_components = effective_dim
            self._dim = effective_dim

        tfidf_matrix = self._vectorizer.fit_transform(self._corpus)
        self._svd.fit(tfidf_matrix)
        self._fitted = True

    def _ensure_fitted(self, text: str) -> None:
        if not self._fitted:
            self._corpus.append(text)
            if len(self._corpus) >= 5:
                self.fit(self._corpus)

    def encode(self, text: str) -> np.ndarray:
        self._ensure_fitted(text)
        if not self._fitted:
            rng = np.random.default_rng(_stable_seed(text))
            vec = rng.standard_normal(self._dim)
            vec /= np.linalg.norm(vec)
            return vec

        tfidf_vec = self._vectorizer.transform([text])
        svd_vec = self._svd.transform(tfidf_vec)[0]
        norm = np.linalg.norm(svd_vec)
        if norm > 1e-12:
            svd_vec /= norm
        return svd_vec


class SentenceTransformerEncoder(Encoder):
    """Encoder using sentence-transformers (all-MiniLM-L6-v2 by default).

    High quality semantic embeddings. "dog" and "canine" will have high
    cosine similarity. Requires ~80MB model download on first use.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "SentenceTransformerEncoder requires sentence-transformers: "
                "pip install sentence-transformers"
            )

        self._model = SentenceTransformer(model_name)
        self._dim = self._model.get_sentence_embedding_dimension()

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, text: str) -> np.ndarray:
        vec = self._model.encode(text, normalize_embeddings=True)
        return np.asarray(vec, dtype=np.float64)

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        vecs = self._model.encode(texts, normalize_embeddings=True, batch_size=64)
        return np.asarray(vecs, dtype=np.float64)


class OpenAIEncoder(Encoder):
    """Encoder using OpenAI's text-embedding API.

    Highest quality. Requires an API key and network access.
    Costs money per call.
    """

    def __init__(self, model: str = "text-embedding-3-small", api_key: Optional[str] = None, dim: Optional[int] = None):
        try:
            import openai
        except ImportError:
            raise ImportError("OpenAIEncoder requires the openai SDK: pip install openai")

        import os as _os
        key = api_key or _os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError("OpenAI API key required (pass api_key= or set OPENAI_API_KEY)")

        self._client = openai.OpenAI(api_key=key)
        self._model = model
        if dim is not None:
            self._dim = dim
        else:
            _known = {
                "text-embedding-3-small": 1536,
                "text-embedding-3-large": 3072,
                "text-embedding-ada-002": 1536,
            }
            self._dim = _known.get(model, 1536 if "small" in model else 3072)

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, text: str) -> np.ndarray:
        resp = self._client.embeddings.create(input=[text], model=self._model)
        vec = np.array(resp.data[0].embedding, dtype=np.float64)
        norm = np.linalg.norm(vec)
        if norm > 1e-12:
            vec /= norm
        return vec

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        resp = self._client.embeddings.create(input=texts, model=self._model)
        vecs = np.array([d.embedding for d in resp.data], dtype=np.float64)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        return vecs / norms


def auto_encoder(preferred_dim: Optional[int] = None) -> Encoder:
    """Return the best available encoder for the current environment.

    Tries in order: SentenceTransformer > TFIDF > RandomIndex.
    """
    try:
        enc = SentenceTransformerEncoder()
        warnings.warn(f"Using SentenceTransformerEncoder (dim={enc.dim})")
        return enc
    except ImportError:
        pass

    try:
        dim = preferred_dim or 512
        enc = TFIDFEncoder(dim=dim)
        warnings.warn(f"Using TFIDFEncoder (dim={dim}); install sentence-transformers for better quality")
        return enc
    except ImportError:
        pass

    dim = preferred_dim or 512
    warnings.warn(f"Using RandomIndexEncoder (dim={dim}); install sentence-transformers for better quality")
    return RandomIndexEncoder(dim=dim)
