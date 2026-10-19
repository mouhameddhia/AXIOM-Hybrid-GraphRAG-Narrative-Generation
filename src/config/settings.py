"""
Central, immutable configuration for the AXIOM hybrid retrieval stack.

Every tunable in the retrieval layer -- fan-out widths, the RRF smoothing
constant, per-stream fusion weights, traversal depth, rerank thresholds -- is
declared here and nowhere else. Modules never call ``os.getenv`` directly and
never embed a literal threshold.

Two properties matter for testability:

``from_env`` over import-time defaults
    Values bind when :func:`get_settings` is first called, not when this module
    is imported. A test can therefore build a ``Settings`` from a plain dict and
    inject it, with no monkeypatching of ``os.environ``.

Frozen dataclasses
    Configuration cannot drift at runtime, so a trace captured alongside a
    result always describes the configuration that produced it.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import ClassVar

from dotenv import load_dotenv

from src.config.env import (
    ConfigError,
    EnvSource,
    env_bool,
    env_float,
    env_frozenset,
    env_int,
    env_path,
    env_str,
    env_str_float_map,
    env_tuple,
    process_env,
)

__all__ = [
    "ConfigError",
    "CorpusConfig",
    "EmbeddingConfig",
    "FusionConfig",
    "GraphTraversalConfig",
    "LLMConfig",
    "LexicalConfig",
    "NarrativeConfig",
    "ReliabilityConfig",
    "Neo4jConfig",
    "RerankerConfig",
    "RetrievalConfig",
    "Settings",
    "TextToCypherConfig",
    "get_settings",
]

#: Project root, resolved from this file's location so that relative data paths
#: work regardless of the caller's working directory.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]


#: Sentinel for BM25_STOPWORDS meaning "filter nothing".
_STOPWORDS_DISABLED = "-"


def _stopwords_from_env(env: EnvSource) -> frozenset[str]:
    """
    Read the stopword list, defaulting to the bundled bilingual set.

    Imported lazily to keep this module free of a dependency on the retrieval
    package, which would otherwise make the import graph circular.
    """
    from src.retrieval.stopwords import DEFAULT_STOPWORDS

    configured = env_frozenset(env, "BM25_STOPWORDS", DEFAULT_STOPWORDS)
    if configured == {_STOPWORDS_DISABLED}:
        return frozenset()
    return frozenset(word.casefold() for word in configured)


def _resolve_under_root(path: Path) -> Path:
    """
    Anchor a relative configured path to :data:`PROJECT_ROOT`.

    Absolute paths pass through untouched. ``resolve()`` also collapses any
    ``..`` segments, so a path pointing outside the project (the mentor-supplied
    export lives in a sibling directory) normalises to a stable absolute path.
    """
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


@dataclass(frozen=True, slots=True)
class Neo4jConfig:
    """Connection parameters for the Neo4j graph database."""

    uri: str
    user: str
    password: str
    database: str
    connection_timeout_s: float

    @classmethod
    def from_env(cls, env: EnvSource) -> "Neo4jConfig":
        return cls(
            uri=env_str(env, "NEO4J_URI", "bolt://localhost:7687"),
            user=env_str(env, "NEO4J_USER", "neo4j"),
            password=env_str(env, "NEO4J_PASSWORD", ""),
            database=env_str(env, "NEO4J_DATABASE", "neo4j"),
            connection_timeout_s=env_float(
                env, "NEO4J_CONNECTION_TIMEOUT_S", 10.0, minimum=0.1
            ),
        )


@dataclass(frozen=True, slots=True)
class CorpusConfig:
    """
    How graph records are turned into retrievable, citable text.

    The retrieval corpus is derived from the knowledge graph at runtime, so no
    label, property name, or identifier field is hard-coded anywhere in the
    codebase. These knobs describe the *derivation strategy* instead:

    ``id_key_suffix``
        Marks a property as an identifier candidate (``customer_id``,
        ``order_line_id``, ...). The serializer picks the identifier whose name
        best matches the node's label -- see
        :class:`src.retrieval.corpus.RecordSerializer`.
    ``name_key_suffixes``
        Fallback naming properties for label types that carry no identifier,
        such as ``City.city_name``.
    ``excluded_property_keys``
        Properties omitted from indexed text: internal keys and any embedded
        vectors, which would otherwise pollute the lexical index.
    """

    source: str
    json_path: Path
    nodes_key: str
    relationships_key: str
    id_key_suffix: str
    name_key_suffixes: tuple[str, ...]
    excluded_property_keys: frozenset[str]
    include_labels: frozenset[str]
    exclude_labels: frozenset[str]
    max_text_chars: int
    float_precision: int

    #: Source identifiers understood by :func:`src.graph.sources.build_graph_source`.
    #: ClassVar, not a field -- these are constants, not per-instance configuration.
    JSON: ClassVar[str] = "json"
    NEO4J: ClassVar[str] = "neo4j"

    @classmethod
    def from_env(cls, env: EnvSource) -> "CorpusConfig":
        source = env_str(env, "CORPUS_SOURCE", cls.JSON).lower()
        if source not in {cls.JSON, cls.NEO4J}:
            raise ConfigError(
                "CORPUS_SOURCE must be one of "
                f"{sorted({cls.JSON, cls.NEO4J})}, got {source!r}"
            )
        return cls(
            source=source,
            # Relative paths resolve against the project root, not the caller's
            # working directory, so the configured location means the same
            # thing whether the entry point is run from the project root, an
            # IDE, or a test runner.
            json_path=_resolve_under_root(
                env_path(
                    env,
                    "CORPUS_JSON_PATH",
                    # The PoC export is the source of truth for this project.
                    # data/knowledge_graph_ai_sample.json is an earlier,
                    # smaller extract whose schema has since drifted from it
                    # (no applies_to_id on PriceComponent), so it is not the
                    # default and is not what the tests run against.
                    PROJECT_ROOT / "axiom_pricing_module_Knowledge-graph_PoC.json",
                )
            ),
            nodes_key=env_str(env, "CORPUS_JSON_NODES_KEY", "nodes"),
            relationships_key=env_str(
                env, "CORPUS_JSON_RELATIONSHIPS_KEY", "relationships"
            ),
            id_key_suffix=env_str(env, "CORPUS_ID_KEY_SUFFIX", "_id"),
            name_key_suffixes=env_tuple(
                env, "CORPUS_NAME_KEY_SUFFIXES", ("_name", "name")
            ),
            excluded_property_keys=env_frozenset(
                env, "CORPUS_EXCLUDED_PROPERTY_KEYS", frozenset({"_eid", "embedding"})
            ),
            include_labels=env_frozenset(env, "CORPUS_INCLUDE_LABELS", frozenset()),
            exclude_labels=env_frozenset(env, "CORPUS_EXCLUDE_LABELS", frozenset()),
            max_text_chars=env_int(env, "CORPUS_MAX_TEXT_CHARS", 1024, minimum=32),
            float_precision=env_int(
                env, "CORPUS_FLOAT_PRECISION", 4, minimum=0, maximum=12
            ),
        )


@dataclass(frozen=True, slots=True)
class LexicalConfig:
    """BM25 parameters. Exposed because pricing text is short and ID-dense."""

    k1: float
    b: float
    epsilon: float
    token_pattern: str
    min_token_length: int
    stopwords: frozenset[str]

    @classmethod
    def from_env(cls, env: EnvSource) -> "LexicalConfig":
        return cls(
            k1=env_float(env, "BM25_K1", 1.5, minimum=0.0),
            b=env_float(env, "BM25_B", 0.75, minimum=0.0, maximum=1.0),
            epsilon=env_float(env, "BM25_EPSILON", 0.25, minimum=0.0),
            # Unicode-aware: keeps Arabic script and alphanumeric identifiers
            # such as SKU0001 as single tokens instead of splitting them apart.
            token_pattern=env_str(env, "BM25_TOKEN_PATTERN", r"[^\W_]+"),
            min_token_length=env_int(env, "BM25_MIN_TOKEN_LENGTH", 1, minimum=1),
            # Essential here, not cosmetic: this corpus is key-value records, so
            # English function words are RARE in it and BM25 scores them as
            # highly discriminative. See src/retrieval/stopwords.py.
            # Set BM25_STOPWORDS to a comma-separated list to override, or to a
            # single "-" to disable filtering entirely.
            stopwords=_stopwords_from_env(env),
        )


@dataclass(frozen=True, slots=True)
class EmbeddingConfig:
    """Configuration for the dense (semantic) retrieval encoder."""

    model_name: str
    embedding_dim: int
    batch_size: int
    normalize: bool
    device: str | None

    @classmethod
    def from_env(cls, env: EnvSource) -> "EmbeddingConfig":
        device = env_str(env, "EMBEDDING_DEVICE", "")
        return cls(
            model_name=env_str(
                env, "EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
            ),
            embedding_dim=env_int(env, "EMBEDDING_DIM", 384, minimum=1),
            batch_size=env_int(env, "EMBEDDING_BATCH_SIZE", 64, minimum=1),
            normalize=env_bool(env, "EMBEDDING_NORMALIZE", True),
            device=device or None,  # None => let sentence-transformers choose.
        )


@dataclass(frozen=True, slots=True)
class GraphTraversalConfig:
    """
    Parameters for the graph-traversal modality.

    ``hop_decay`` is what makes traversal results *rankable*. Graph traversal
    has no natural relevance score, so proximity stands in for one: a record
    reached in one hop scores ``1.0``, two hops ``hop_decay``, and so on.
    Without it every traversal hit ties, and RRF -- which is purely rank-based
    -- would consume an arbitrary ordering.
    """

    max_hops: int
    hop_decay: float
    max_neighbors: int
    directed: bool
    relationship_types: frozenset[str]
    anchor_pattern: str
    max_anchors: int

    @classmethod
    def from_env(cls, env: EnvSource) -> "GraphTraversalConfig":
        return cls(
            max_hops=env_int(env, "GRAPH_MAX_HOPS", 2, minimum=1, maximum=6),
            hop_decay=env_float(env, "GRAPH_HOP_DECAY", 0.5, minimum=0.0, maximum=1.0),
            max_neighbors=env_int(env, "GRAPH_MAX_NEIGHBORS", 200, minimum=1),
            directed=env_bool(env, "GRAPH_DIRECTED", False),
            # Empty => traverse every relationship type present in the graph.
            relationship_types=env_frozenset(
                env, "GRAPH_RELATIONSHIP_TYPES", frozenset()
            ),
            # Identifier shapes such as CUST0015, SKU0001, OPP00003, AG001.
            anchor_pattern=env_str(
                env, "GRAPH_ANCHOR_PATTERN", r"\b[A-Za-z][A-Za-z_]*\d{2,}\b"
            ),
            max_anchors=env_int(env, "GRAPH_MAX_ANCHORS", 4, minimum=1),
        )


@dataclass(frozen=True, slots=True)
class FusionConfig:
    """
    Reciprocal Rank Fusion parameters.

    ``weights`` is an open map keyed by *stream name*. The fuser accepts any
    number of streams, so adding a fourth modality later needs a configuration
    entry, not a code change. Streams absent from the map use ``default_weight``.
    """

    rrf_k: int
    weights: Mapping[str, float]
    default_weight: float
    max_candidates: int

    @classmethod
    def from_env(cls, env: EnvSource) -> "FusionConfig":
        return cls(
            # RRF smoothing constant. Higher values flatten the contribution
            # curve, so agreement across streams matters more than depth within
            # any single stream. 60 is the value from Cormack et al. (2009).
            rrf_k=env_int(env, "RRF_K", 60, minimum=1),
            weights=MappingProxyType(
                env_str_float_map(env, "FUSION_STREAM_WEIGHTS", {}, minimum=0.0)
            ),
            default_weight=env_float(env, "FUSION_DEFAULT_WEIGHT", 1.0, minimum=0.0),
            max_candidates=env_int(env, "FUSION_MAX_CANDIDATES", 100, minimum=1),
        )

    def weight_for(self, stream_name: str) -> float:
        """Return the configured weight for ``stream_name``, or the default."""
        return self.weights.get(stream_name, self.default_weight)


@dataclass(frozen=True, slots=True)
class RerankerConfig:
    """
    Cross-encoder reranking parameters.

    ``score_transform`` matters for governance. ``ms-marco-MiniLM`` emits an
    unbounded logit, against which no fixed ``min_score`` is meaningful.
    Applying a sigmoid maps it to ``(0, 1)`` so that ``min_score`` becomes a
    stable, auditable relevance floor rather than a magic number.
    """

    enabled: bool
    model_name: str
    top_n: int
    batch_size: int
    min_score: float
    score_transform: str
    min_results: int
    fail_open: bool

    #: ClassVar, not a field -- see the note on CorpusConfig.
    IDENTITY: ClassVar[str] = "identity"
    SIGMOID: ClassVar[str] = "sigmoid"

    @classmethod
    def from_env(cls, env: EnvSource) -> "RerankerConfig":
        transform = env_str(env, "RERANK_SCORE_TRANSFORM", cls.SIGMOID).lower()
        if transform not in {cls.IDENTITY, cls.SIGMOID}:
            raise ConfigError(
                "RERANK_SCORE_TRANSFORM must be one of "
                f"{sorted({cls.IDENTITY, cls.SIGMOID})}, got {transform!r}"
            )
        return cls(
            enabled=env_bool(env, "RERANK_ENABLED", True),
            model_name=env_str(
                env, "RERANKER_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2"
            ),
            # Rerank depth: the cross-encoder costs one transformer pass per
            # candidate, so it only ever sees the fused head, never the corpus.
            top_n=env_int(env, "RERANK_TOP_N", 20, minimum=1),
            batch_size=env_int(env, "RERANK_BATCH_SIZE", 32, minimum=1),
            min_score=env_float(env, "RERANK_MIN_SCORE", 0.0, minimum=0.0),
            score_transform=transform,
            # Guarantees a non-empty result even when everything scores below
            # the floor. Set to 0 to let the threshold produce an empty answer:
            # the retrieval-side precondition for an honest abstention.
            min_results=env_int(env, "RERANK_MIN_RESULTS", 1, minimum=0),
            # When True, a model load or inference failure degrades to fused
            # order instead of failing the request.
            fail_open=env_bool(env, "RERANK_FAIL_OPEN", True),
        )


@dataclass(frozen=True, slots=True)
class RetrievalConfig:
    """Fan-out widths shared by every modality."""

    default_top_k: int
    candidate_multiplier: int
    max_candidate_top_k: int

    @classmethod
    def from_env(cls, env: EnvSource) -> "RetrievalConfig":
        return cls(
            default_top_k=env_int(env, "DEFAULT_TOP_K", 10, minimum=1),
            # Each stream retrieves deeper than the final answer needs, so RRF
            # has room to promote records that several streams agree on but
            # none ranked inside its own top-k.
            candidate_multiplier=env_int(env, "CANDIDATE_MULTIPLIER", 4, minimum=1),
            max_candidate_top_k=env_int(env, "MAX_CANDIDATE_TOP_K", 200, minimum=1),
        )

    def candidate_top_k(self, top_k: int) -> int:
        """Per-stream fan-out width for a request asking for ``top_k`` results."""
        return min(top_k * self.candidate_multiplier, self.max_candidate_top_k)


@dataclass(frozen=True, slots=True)
class LLMConfig:
    """LLM parameters for text-to-Cypher and grounded generation."""

    provider: str
    model_name: str
    base_url: str
    api_key: str
    temperature: float
    request_timeout_s: float
    max_retries: int

    @classmethod
    def from_env(cls, env: EnvSource) -> "LLMConfig":
        return cls(
            provider=env_str(env, "LLM_PROVIDER", "ollama"),
            model_name=env_str(env, "LLM_MODEL_NAME", "qwen2.5:7b"),
            base_url=env_str(env, "OLLAMA_BASE_URL", "http://localhost:11434"),
            api_key=env_str(env, "LLM_API_KEY", ""),
            temperature=env_float(env, "LLM_TEMPERATURE", 0.0, minimum=0.0, maximum=2.0),
            request_timeout_s=env_float(env, "LLM_REQUEST_TIMEOUT_S", 60.0, minimum=1.0),
            max_retries=env_int(env, "LLM_MAX_RETRIES", 3, minimum=0, maximum=10),
        )


@dataclass(frozen=True, slots=True)
class TextToCypherConfig:
    """
    Parameters for the natural-language-to-Cypher engine.

    ``min_schema_overlap`` and ``max_schema_labels`` control schema pruning:
    the LLM is shown only labels whose name or properties share vocabulary
    with the question, capped at a maximum count so the prompt stays bounded
    even on a large schema. Both operate on the *live* schema derived from the
    graph at startup -- nothing here names an AXIOM label.
    """

    enabled: bool
    max_retries: int
    min_schema_overlap: int
    max_schema_labels: int
    max_few_shot_examples: int
    max_result_rows: int
    retry_temperature_step: float

    @classmethod
    def from_env(cls, env: EnvSource) -> "TextToCypherConfig":
        return cls(
            enabled=env_bool(env, "TEXT_TO_CYPHER_ENABLED", True),
            # Self-heal attempts after the first. The exact server error text
            # is what gets fed back each time -- see Neo4jQueryError.
            max_retries=env_int(env, "TEXT_TO_CYPHER_MAX_RETRIES", 3, minimum=0, maximum=10),
            min_schema_overlap=env_int(
                env, "TEXT_TO_CYPHER_MIN_SCHEMA_OVERLAP", 1, minimum=0
            ),
            max_schema_labels=env_int(
                env, "TEXT_TO_CYPHER_MAX_SCHEMA_LABELS", 12, minimum=1
            ),
            max_few_shot_examples=env_int(
                env, "TEXT_TO_CYPHER_MAX_FEW_SHOT_EXAMPLES", 3, minimum=0
            ),
            # A safety net, not a correctness feature: aggregation queries
            # return one row, but an unaggregated MATCH could return the whole
            # corpus. Rows beyond this are dropped from the result, not from
            # what the database computed -- an aggregate value is unaffected.
            max_result_rows=env_int(
                env, "TEXT_TO_CYPHER_MAX_RESULT_ROWS", 200, minimum=1
            ),
            # Added to the base temperature on each retry. At temperature 0 a
            # model re-reads a corrected prompt and greedily regenerates the
            # same rejected plan, so a self-heal loop that does not vary
            # sampling cannot escape its first answer. Set to 0.0 to keep
            # every attempt fully deterministic.
            retry_temperature_step=env_float(
                env, "TEXT_TO_CYPHER_RETRY_TEMPERATURE_STEP", 0.3, minimum=0.0, maximum=1.0
            ),
        )

    def temperature_for(self, attempt: int, base: float) -> float:
        """
        Sampling temperature for a given attempt (1 = the first try).

        Capped at 1.0: beyond that the model stops producing well-formed JSON
        more often than it produces a better plan.
        """
        return min(base + self.retry_temperature_step * (attempt - 1), 1.0)


@dataclass(frozen=True, slots=True)
class NarrativeConfig:
    """
    Grounded narrative generation (D4).

    The tolerances are the only judgement calls here. They exist so a
    narrative rounding a figure for readability is not treated as a
    fabrication, and they are deliberately tight: every unit of slack is
    room a model could use to state a number nobody computed.
    """

    enabled: bool
    max_retries: int
    max_words: int
    absolute_tolerance: float
    relative_tolerance: float
    min_citation_coverage: float
    retry_temperature_step: float

    @classmethod
    def from_env(cls, env: EnvSource) -> "NarrativeConfig":
        return cls(
            enabled=env_bool(env, "NARRATIVE_ENABLED", True),
            # Corrective attempts after the first. The exact verification
            # violations are fed back each time; past this, the generator
            # abstains rather than serving an unverified explanation.
            max_retries=env_int(env, "NARRATIVE_MAX_RETRIES", 2, minimum=0, maximum=5),
            max_words=env_int(env, "NARRATIVE_MAX_WORDS", 130, minimum=20, maximum=600),
            # Currency minor units: a narrative may write 89,489.98 for a
            # computed 89489.98 without that counting as a new number.
            absolute_tolerance=env_float(
                env, "NARRATIVE_ABSOLUTE_TOLERANCE", 0.01, minimum=0.0, maximum=1.0
            ),
            # Proportional slack for a rounded figure. At 0.005, a stated
            # 13.2% clears a computed 13.2% but a stated 15% does not.
            relative_tolerance=env_float(
                env, "NARRATIVE_RELATIVE_TOLERANCE", 0.005, minimum=0.0, maximum=0.05
            ),
            # "Every narrative claim cites a specific record" (acceptance
            # criteria, Grounding & citation). At 1.0 every sentence stating a
            # figure must name the record it rests on; lowering this admits
            # sentences a reader cannot check.
            min_citation_coverage=env_float(
                env, "NARRATIVE_MIN_CITATION_COVERAGE", 1.0, minimum=0.0, maximum=1.0
            ),
            retry_temperature_step=env_float(
                env, "NARRATIVE_RETRY_TEMPERATURE_STEP", 0.2, minimum=0.0, maximum=1.0
            ),
        )

    def temperature_for(self, attempt: int, base: float) -> float:
        """Sampling temperature for a given attempt (1 = the first try)."""
        return min(base + self.retry_temperature_step * (attempt - 1), 1.0)


@dataclass(frozen=True, slots=True)
class ReliabilityConfig:
    """
    The reliability layer (D5): sampling, agreement, and corrective retrieval.

    The defaults here were set from measurement rather than taste; the D5
    section of the deliverable document records the runs behind each one.
    """

    enabled: bool
    #: How many narratives to sample per explanation. The plan specifies 3.
    samples: int
    #: How many must survive verification and attribution for any answer to
    #: be served.
    min_verified: int
    #: Consensus-core fraction below which samples are judged to disagree.
    min_agreement: float
    #: Sampling temperature. Must be above zero: at 0 a model regenerates the
    #: same text k times, and k identical samples measure nothing while
    #: costing k times as much.
    sample_temperature: float
    #: Whether a misattributed sample is disqualified from voting. Always
    #: measured; turning this off yields the D4 baseline for comparison.
    enforce_attribution: bool
    #: Whether a weak-evidence pack triggers corrective re-retrieval.
    corrective_enabled: bool
    #: Citable records per figure below which evidence counts as thin. This
    #: is the trigger corrective retrieval can actually move; per-figure
    #: provenance (citation density) is upstream and reported separately.
    min_support_ratio: float
    corrective_top_k: int
    corrective_max_added: int

    @classmethod
    def from_env(cls, env: EnvSource) -> "ReliabilityConfig":
        return cls(
            enabled=env_bool(env, "RELIABILITY_ENABLED", True),
            samples=env_int(env, "RELIABILITY_SAMPLES", 3, minimum=1, maximum=9),
            min_verified=env_int(env, "RELIABILITY_MIN_VERIFIED", 2, minimum=1),
            min_agreement=env_float(
                env, "RELIABILITY_MIN_AGREEMENT", 0.6, minimum=0.0, maximum=1.0
            ),
            sample_temperature=env_float(
                env, "RELIABILITY_SAMPLE_TEMPERATURE", 0.7, minimum=0.05, maximum=1.5
            ),
            enforce_attribution=env_bool(
                env, "RELIABILITY_ENFORCE_ATTRIBUTION", True
            ),
            corrective_enabled=env_bool(env, "RELIABILITY_CORRECTIVE_ENABLED", True),
            min_support_ratio=env_float(
                env, "RELIABILITY_MIN_SUPPORT_RATIO", 0.75, minimum=0.0, maximum=5.0
            ),
            corrective_top_k=env_int(
                env, "RELIABILITY_CORRECTIVE_TOP_K", 6, minimum=1, maximum=50
            ),
            corrective_max_added=env_int(
                env, "RELIABILITY_CORRECTIVE_MAX_ADDED", 4, minimum=0, maximum=20
            ),
        )

    def __post_init__(self) -> None:
        if self.min_verified > self.samples:
            raise ConfigError(
                f"RELIABILITY_MIN_VERIFIED ({self.min_verified}) cannot exceed "
                f"RELIABILITY_SAMPLES ({self.samples}): the quorum would never "
                f"be reachable and every explanation would abstain."
            )


@dataclass(frozen=True, slots=True)
class Settings:
    """Aggregate of every configuration section."""

    neo4j: Neo4jConfig
    corpus: CorpusConfig
    lexical: LexicalConfig
    embedding: EmbeddingConfig
    graph: GraphTraversalConfig
    fusion: FusionConfig
    reranker: RerankerConfig
    retrieval: RetrievalConfig
    llm: LLMConfig
    text_to_cypher: TextToCypherConfig
    narrative: NarrativeConfig
    reliability: ReliabilityConfig

    @classmethod
    def from_env(cls, env: EnvSource | None = None) -> "Settings":
        """
        Build a full configuration from ``env`` (defaults to the process
        environment). Raises :class:`ConfigError` on the first invalid value.
        """
        source = process_env() if env is None else env
        return cls(
            neo4j=Neo4jConfig.from_env(source),
            corpus=CorpusConfig.from_env(source),
            lexical=LexicalConfig.from_env(source),
            embedding=EmbeddingConfig.from_env(source),
            graph=GraphTraversalConfig.from_env(source),
            fusion=FusionConfig.from_env(source),
            reranker=RerankerConfig.from_env(source),
            retrieval=RetrievalConfig.from_env(source),
            llm=LLMConfig.from_env(source),
            text_to_cypher=TextToCypherConfig.from_env(source),
            narrative=NarrativeConfig.from_env(source),
            reliability=ReliabilityConfig.from_env(source),
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return the process-wide settings, loading ``.env`` on first access.

    Cached so repeated calls are free. Tests needing a different configuration
    should construct ``Settings.from_env({...})`` and inject it rather than
    clearing this cache.
    """
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    return Settings.from_env()
