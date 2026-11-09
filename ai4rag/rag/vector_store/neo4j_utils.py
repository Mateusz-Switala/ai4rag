# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
"""Private adapters and helpers used by `ai4rag.rag.vector_store.neo4j`."""

import asyncio
import hashlib
import json
from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from threading import Lock
from time import perf_counter
from typing import Any, Iterator

import neo4j
from neo4j_graphrag.components.kg_writer import KGWriterModel, Neo4jWriter
from neo4j_graphrag.components.text_splitters.base import TextSplitter
from neo4j_graphrag.components.types import LexicalGraphConfig, Neo4jGraph, Neo4jRelationship, TextChunk, TextChunks
from neo4j_graphrag.embeddings.base import Embedder as _NeoEmbedder
from neo4j_graphrag.generation.prompts import ERExtractionTemplate
from neo4j_graphrag.llm.base import LLMInterfaceV2 as _NeoLLMInterface
from neo4j_graphrag.llm.types import LLMResponse
from neo4j_graphrag.neo4j_queries import db_cleaning_query
from neo4j_graphrag.types import LLMMessage

from ai4rag.rag.embedding.base_model import BaseEmbeddingModel
from ai4rag.rag.foundation_models.base_model import BaseFoundationModel
from ai4rag.rag.vector_store.utils import validate_search_params


class Neo4jGraphSchema:
    """Names used by ai4rag's persistent Neo4j graph schema."""

    DOCUMENT_LABEL = "Document"
    CHUNK_LABEL = "Chunk"
    ENTITY_LABEL = "__Entity__"
    KG_BUILDER_LABEL = "__KGBuilder__"

    CONTAINS_RELATIONSHIP = "CONTAINS"
    NEXT_CHUNK_RELATIONSHIP = "NEXT_CHUNK"
    FROM_CHUNK_RELATIONSHIP = "FROM_CHUNK"

    ID = "id"
    TEXT = "text"
    EMBEDDING = "embedding"
    DOCUMENT_ID = "document_id"
    SEQUENCE_NUMBER = "sequence_number"
    METADATA = "metadata"
    COLLECTION = "collection"
    SOURCE = "source"
    KG_COLLECTIONS = "ai4rag_kg_collections"
    ENTITY_KEY = "ai4rag_entity_key"


_SCHEMA = Neo4jGraphSchema


_CONSTRAINED_KG_ENTITIES = ("Person", "Organization", "Place", "Concept", "Event", "Product", "Technology")
_CONSTRAINED_KG_RELATIONS = ("RELATED_TO", "PART_OF", "LOCATED_IN", "BELONGS_TO", "CREATED_BY", "MENTIONS")
DEFAULT_KG_EXTRACTION_MAX_COMPLETION_TOKENS = 1024
# Entity extraction is generation-heavy. Two chunks per request amortises
# request overhead while retaining chunk-level provenance.
DEFAULT_KG_EXTRACTION_BATCH_SIZE = 2
KG_MAX_ENTITIES_PER_CHUNK = 6
KG_MAX_RELATIONSHIPS_PER_CHUNK = 6
# MaaS' OpenAI-compatible endpoint accepts JSON-object mode, but the deployed
# Qwen model does not reliably honour strict JSON Schema mode. The detailed
# prompt and local validation retain the application contract without repeatedly
# retrying an unsupported response-format capability.
KG_BATCH_RESPONSE_FORMAT = {"type": "json_object"}
KG_BATCH_JSON_SCHEMA_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "kg_batch",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "chunks": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "chunk_id": {"type": "string"},
                            "nodes": {"type": "array"},
                            "relationships": {"type": "array"},
                        },
                        "required": ["chunk_id", "nodes", "relationships"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["chunks"],
            "additionalProperties": False,
        },
    },
}

# GraphRAG's lexical nodes are temporary: _CanonicalKGWriter discards them in
# favor of the canonical nodes already stored by add_documents. Keep their
# labels distinct from entity types chosen by free extraction (e.g. Document).
_KG_LEXICAL_GRAPH_CONFIG = LexicalGraphConfig(
    document_node_label="__AI4RAG_KG_SOURCE_DOCUMENT__",
    chunk_node_label="__AI4RAG_KG_SOURCE_CHUNK__",
)


class _PreChunkedTextSplitter(TextSplitter):
    """Preserve the canonical chunk passed to the KG pipeline as one text chunk."""

    def iter_chunks(self, text: str) -> Iterator[TextChunk]:
        """Yield the canonical text unchanged as one chunk."""
        yield TextChunk(text=text, index=0)

    async def run(self, text: str) -> TextChunks:
        return TextChunks(chunks=[TextChunk(text=text, index=0)])


def _collection_vector_index_name(collection_name: str) -> str:
    """Return the vector-index name dedicated to *collection_name*."""
    return f"{collection_name}__embedding"


def _validate_kg_extraction_config(config: dict[str, Any] | None) -> dict[str, Any]:
    """Validate KG extraction settings and apply the constrained default."""
    if config is None:
        return {
            "mode": "constrained",
            "max_completion_tokens": DEFAULT_KG_EXTRACTION_MAX_COMPLETION_TOKENS,
            "batch_size": DEFAULT_KG_EXTRACTION_BATCH_SIZE,
            "response_format": "json_object",
            "max_skipped_chunks": 0,
            "partial_failure_policy": "degrade",
        }
    if not isinstance(config, dict):
        raise TypeError("kg_extraction_config must be a dictionary or None.")

    mode = config.get("mode", "constrained")
    if mode not in {"constrained", "free"}:
        raise ValueError("kg_extraction_config.mode must be 'constrained' or 'free'.")
    max_completion_tokens = config.get("max_completion_tokens", DEFAULT_KG_EXTRACTION_MAX_COMPLETION_TOKENS)
    if (
        isinstance(max_completion_tokens, bool)
        or not isinstance(max_completion_tokens, int)
        or max_completion_tokens < 1
    ):
        raise ValueError("kg_extraction_config.max_completion_tokens must be a positive integer.")
    batch_size = config.get("batch_size", DEFAULT_KG_EXTRACTION_BATCH_SIZE)
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("kg_extraction_config.batch_size must be a positive integer.")
    response_format = config.get("response_format", "json_object")
    if response_format not in {"json_object", "json_schema"}:
        raise ValueError("kg_extraction_config.response_format must be 'json_object' or 'json_schema'.")
    max_skipped_chunks = config.get("max_skipped_chunks", 0)
    if isinstance(max_skipped_chunks, bool) or not isinstance(max_skipped_chunks, int) or max_skipped_chunks < 0:
        raise ValueError("kg_extraction_config.max_skipped_chunks must be a non-negative integer.")
    partial_failure_policy = config.get("partial_failure_policy", "degrade")
    if partial_failure_policy not in {"degrade", "raise"}:
        raise ValueError("kg_extraction_config.partial_failure_policy must be 'degrade' or 'raise'.")
    common = {
        "mode": mode,
        "max_completion_tokens": max_completion_tokens,
        "batch_size": batch_size,
        "response_format": response_format,
        "max_skipped_chunks": max_skipped_chunks,
        "partial_failure_policy": partial_failure_policy,
    }
    if mode == "constrained":
        return common

    limits = {
        "max_entities_per_chunk": config.get("max_entities_per_chunk"),
        "max_relationships_per_chunk": config.get("max_relationships_per_chunk"),
    }
    for name, value in limits.items():
        if not isinstance(value, int) or value < 1:
            raise ValueError(f"kg_extraction_config.{name} must be a positive integer for free extraction.")
    return {**common, **limits}


def _kg_pipeline_extraction_options(config: dict[str, Any]) -> dict[str, Any]:
    """Return ``SimpleKGPipeline`` options for the requested extraction mode."""
    if config["mode"] == "constrained":
        return {"entities": _CONSTRAINED_KG_ENTITIES, "relations": _CONSTRAINED_KG_RELATIONS}

    entity_limit = config["max_entities_per_chunk"]
    relation_limit = config["max_relationships_per_chunk"]
    template = ERExtractionTemplate.DEFAULT_TEMPLATE.replace(
        "Extract the entities (nodes) and specify their type from the following text.",
        "Extract the entities (nodes) and specify their type from the following text. "
        f"Extract at most {entity_limit} entities.",
    ).replace(
        "Also extract the relationships between these nodes.",
        f"Also extract at most {relation_limit} relationships between these nodes.",
    )
    # "FREE" bypasses automatic schema extraction. Omitting the schema would
    # ask the model to generate constraints, which can be invalid or conflicting.
    # The model still determines entity and relationship types from each chunk.
    return {"schema": "FREE", "prompt_template": ERExtractionTemplate(template=template)}


class _EmbedderAdapter(_NeoEmbedder):  # type: ignore[misc]
    """Wraps :class:`BaseEmbeddingModel` to satisfy ``neo4j_graphrag``'s embedder interface."""

    def __init__(self, model: BaseEmbeddingModel, cached_embeddings: Mapping[str, list[float]] | None = None) -> None:
        super().__init__()
        self._model = model
        self._cached_embeddings = cached_embeddings or {}
        self._model_calls = 0
        self._cache_hits = 0
        self._elapsed_seconds = 0.0
        self._metrics_lock = Lock()

    def embed_query(self, text: str) -> list[float]:
        cached = self._cached_embeddings.get(text)
        if cached is not None:
            with self._metrics_lock:
                self._cache_hits += 1
            return cached

        started_at = perf_counter()
        result = self._model.embed_query(text)
        with self._metrics_lock:
            self._model_calls += 1
            self._elapsed_seconds += perf_counter() - started_at
        return result

    @property
    def metrics(self) -> tuple[int, int, float]:
        """Return model calls, cache hits, and cumulative model-call duration."""
        with self._metrics_lock:
            return self._model_calls, self._cache_hits, self._elapsed_seconds


class _LLMAdapter(_NeoLLMInterface):  # type: ignore[misc]
    """Wraps :class:`BaseFoundationModel` to satisfy ``neo4j_graphrag``'s LLM interface.

    Implements GraphRAG's V2 list-of-messages contract and bridges the sync
    :meth:`BaseFoundationModel.chat` via a worker thread. Response JSON is
    normalised for common, schema-compatible model variants before parsing.
    """

    def __init__(
        self,
        model: BaseFoundationModel,
        max_completion_tokens: int | None = None,
        response_format: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(model_name=model.model_id)
        self._model = model
        self._max_completion_tokens = max_completion_tokens
        self._response_format = dict(response_format) if response_format is not None else None
        self._calls = 0
        self._elapsed_seconds = 0.0
        self._format_repairs = 0
        self._metrics_lock = Lock()

    def invoke(
        self,
        input: list[LLMMessage],  # pylint: disable=redefined-builtin
        *,
        response_format: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Invoke the wrapped model using GraphRAG V2's message contract."""
        return self._invoke(
            input,
            max_completion_tokens=kwargs.pop("max_completion_tokens", self._max_completion_tokens),
            response_format=response_format,
            **kwargs,
        )

    def _invoke(
        self,
        input: list[LLMMessage],  # pylint: disable=redefined-builtin
        max_completion_tokens: int | None = None,
        response_format: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Make one request using the supplied completion limit."""
        model_kwargs = dict(kwargs)
        if max_completion_tokens is not None:
            model_kwargs["max_completion_tokens"] = max_completion_tokens
        effective_response_format = response_format if response_format is not None else self._response_format
        if effective_response_format is not None:
            model_kwargs["response_format"] = effective_response_format
        started_at = perf_counter()
        response = self._model.chat(input, **model_kwargs)[0]
        with self._metrics_lock:
            self._calls += 1
            self._elapsed_seconds += perf_counter() - started_at
        content = (response["content"] if isinstance(response, Mapping) else response.message.content) or ""
        normalised_content = _normalise_kg_response(content)
        if normalised_content != content:
            with self._metrics_lock:
                self._format_repairs += 1
        return LLMResponse(content=normalised_content)

    @property
    def metrics(self) -> tuple[int, float, int]:
        """Return model calls, cumulative duration, and repaired response count."""
        with self._metrics_lock:
            return self._calls, self._elapsed_seconds, self._format_repairs

    async def ainvoke(
        self,
        input: list[LLMMessage],  # pylint: disable=redefined-builtin
        *,
        response_format: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Asynchronously invoke the wrapped model using GraphRAG V2's contract."""
        return await asyncio.to_thread(self.invoke, input, response_format=response_format, **kwargs)

    async def ainvoke_with_completion_limit(
        self, input: str, max_completion_tokens: int  # pylint: disable=redefined-builtin
    ) -> LLMResponse:
        """Invoke the model with a request-specific completion-token limit."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._invoke_with_completion_limit, input, max_completion_tokens)

    def _invoke_with_completion_limit(
        self, input: str, max_completion_tokens: int  # pylint: disable=redefined-builtin
    ) -> LLMResponse:
        """Invoke with a request-specific output limit while retaining adapter metrics."""
        return self._invoke([{"role": "user", "content": input}], max_completion_tokens=max_completion_tokens)


# Keep malformed payloads untouched so GraphRAG can apply its configured error policy.
# pylint: disable-next=too-many-return-statements
def _normalise_kg_response(content: str) -> str:
    """Convert common schema-compatible KG response variants to GraphRAG's JSON object.

    The extractor expects an object containing ``nodes`` and ``relationships``.
    Some OpenAI-compatible models instead return a fenced object, a singleton
    list containing that object, or use ``entities``/``relations`` aliases.
    Keep unparseable content unchanged so GraphRAG retains its normal error
    policy, and never log model output because it contains source content.
    """
    stripped = content.strip()
    recovered_embedded_object = False
    if stripped.startswith("```") and stripped.endswith("```"):
        stripped = stripped.split("\n", maxsplit=1)[-1].rsplit("```", maxsplit=1)[0].strip()
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        # Some MaaS deployments add a short preamble despite JSON-object mode.
        # Recover only an already-valid, balanced object; never synthesize data.
        decoder = json.JSONDecoder()
        parsed = None
        for offset, character in enumerate(stripped):
            if character != "{":
                continue
            try:
                candidate, _ = decoder.raw_decode(stripped[offset:])
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, dict):
                parsed = candidate
                recovered_embedded_object = True
                break
        if parsed is None:
            return content

    if isinstance(parsed, list):
        if len(parsed) == 1 and isinstance(parsed[0], dict):
            parsed = parsed[0]
        else:
            parsed = {"nodes": parsed, "relationships": []}
    if not isinstance(parsed, dict):
        return content
    if "chunks" in parsed:
        if isinstance(parsed["chunks"], list):
            return (
                content
                if not recovered_embedded_object and content == stripped and set(parsed) == {"chunks"}
                else json.dumps({"chunks": parsed["chunks"]})
            )
        return content

    nodes = parsed.get("nodes", parsed.get("entities", []))
    relationships = parsed.get("relationships", parsed.get("relations", []))
    if not isinstance(nodes, list) or not isinstance(relationships, list):
        return content
    if not recovered_embedded_object and content == stripped and set(parsed) == {"nodes", "relationships"}:
        return content
    return json.dumps({"nodes": nodes, "relationships": relationships})


class _CollectionKGWriter(Neo4jWriter):
    """Record ownership on precisely the relationships written by the pipeline."""

    def __init__(self, driver: neo4j.Driver, database: str, collection_name: str) -> None:
        super().__init__(driver=driver, neo4j_database=database)
        self._collection_name = collection_name
        self._setup_complete = False
        self._setup_lock = Lock()
        self._write_elapsed_seconds = 0.0
        self._write_lock = Lock()

    def _db_setup(self) -> None:
        # SimpleKGPipeline invokes its writer once per canonical chunk. The
        # temporary-node index is database-wide, so creating it more than once
        # produces needless schema calls and noisy notifications.
        with self._setup_lock:
            if self._setup_complete:
                return
            self.driver.execute_query(
                "CREATE INDEX __entity__tmp_internal_id IF NOT EXISTS FOR (n:__KGBuilder__) ON (n.__tmp_internal_id)",
                database_=self.neo4j_database,
            )
            self._setup_complete = True

    def ensure_setup(self) -> None:
        """Create the shared temporary-node index before concurrent writes begin."""
        self._db_setup()

    def _db_cleaning(self) -> None:
        query = db_cleaning_query(
            support_variable_scope_clause=self.is_version_5_23_or_above,
            batch_size=self.batch_size,
        )
        with self.driver.session(database=self.neo4j_database) as session:
            session.run(query)

    async def run(
        self,
        graph: Neo4jGraph,
        lexical_graph_config: LexicalGraphConfig = LexicalGraphConfig(),
    ) -> KGWriterModel:
        started_at = perf_counter()
        try:
            return await super().run(graph, lexical_graph_config)
        finally:
            with self._write_lock:
                self._write_elapsed_seconds += perf_counter() - started_at

    @property
    def write_elapsed_seconds(self) -> float:
        """Return cumulative time spent writing KG graphs."""
        with self._write_lock:
            return self._write_elapsed_seconds

    def _upsert_relationships(self, rels: list[Neo4jRelationship]) -> None:
        # Match each endpoint in a separate clause.  The upstream query uses
        # two disconnected patterns in one MATCH, which emits a Cartesian
        # product warning even though both endpoints are keyed by one row.
        # Keeping `row` between clauses preserves its one-to-one association.
        subquery_open = "CALL (rel, row) {" if self.is_version_5_23_or_above else "CALL {"
        query = (
            "UNWIND $rows AS row "
            "MATCH (start:__KGBuilder__ {__tmp_internal_id: row.start_node_id}) "
            "WITH start, row "
            "MATCH (end:__KGBuilder__ {__tmp_internal_id: row.end_node_id}) "
            "WITH start, end, row "
            "CALL apoc.merge.relationship(start, row.type, {}, row.properties, end, row.properties) YIELD rel "
            "WITH rel, row "
            f"{subquery_open} WITH rel, row "
            "WITH rel, row WHERE row.embedding_properties IS NOT NULL "
            "UNWIND keys(row.embedding_properties) AS emb "
            "CALL db.create.setRelationshipVectorProperty(rel, emb, row.embedding_properties[emb]) } "
            "SET rel.ai4rag_kg_collections = CASE "
            "WHEN $col IN COALESCE(rel.ai4rag_kg_collections, []) "
            "THEN rel.ai4rag_kg_collections "
            "ELSE COALESCE(rel.ai4rag_kg_collections, []) + $col END "
            "RETURN elementId(rel)"
        )
        self.driver.execute_query(
            query,
            parameters_={"rows": self._relationships_to_rows(rels), "col": self._collection_name},
            database_=self.neo4j_database,
        )


class _CanonicalKGWriter(_CollectionKGWriter):
    """Write extracted entities onto chunks already stored by ``add_documents``."""

    def __init__(self, driver: neo4j.Driver, database: str, collection_name: str) -> None:
        super().__init__(driver, database, collection_name)
        self._chunk_id: ContextVar[str | None] = ContextVar("canonical_kg_chunk_id", default=None)

    @contextmanager
    def for_chunk(self, chunk_id: str) -> Iterator[None]:
        """Bind the canonical ID to one asynchronous KG pipeline run."""
        token = self._chunk_id.set(chunk_id)
        try:
            yield
        finally:
            self._chunk_id.reset(token)

    def write_graphs(self, graphs: list[tuple[str, Neo4jGraph]]) -> None:
        """Persist extracted graphs with collection-scoped entity identities.

        The upstream writer creates temporary nodes for every extraction and
        resolves duplicates in a later APOC pass.  That makes indexing more
        expensive as the graph grows.  Instead, entity nodes are merged by a
        stable key composed from collection, type, and normalized name.  Nodes
        without a name remain local to their source chunk to avoid an unsafe
        merge.  All rows are written in bounded transactions.
        """
        node_rows: list[dict[str, Any]] = []
        relationship_rows: list[dict[str, Any]] = []
        link_rows: list[dict[str, str]] = []

        for chunk_id, graph in graphs:
            entities = [
                node for node in graph.nodes if node.label not in _KG_LEXICAL_GRAPH_CONFIG.lexical_graph_node_labels
            ]
            key_by_id = {node.id: self._entity_key(chunk_id, node.id, node.label, node.properties) for node in entities}
            for node in entities:
                node_rows.append(
                    {
                        "key": key_by_id[node.id],
                        "labels": [node.label, _SCHEMA.ENTITY_LABEL],
                        "properties": node.properties,
                    }
                )
                link_rows.append({"key": key_by_id[node.id], "chunk_id": chunk_id})
            relationship_rows.extend(
                {
                    "start_key": key_by_id[relationship.start_node_id],
                    "end_key": key_by_id[relationship.end_node_id],
                    "type": relationship.type,
                    "properties": relationship.properties,
                }
                for relationship in graph.relationships
                if relationship.start_node_id in key_by_id and relationship.end_node_id in key_by_id
            )

        started_at = perf_counter()
        for offset in range(0, len(node_rows), self.batch_size):
            self._write_entity_batch(node_rows[offset : offset + self.batch_size])
        for offset in range(0, len(relationship_rows), self.batch_size):
            self._write_relationship_batch(relationship_rows[offset : offset + self.batch_size])
        for offset in range(0, len(link_rows), self.batch_size):
            self._write_link_batch(link_rows[offset : offset + self.batch_size])
        with self._write_lock:
            self._write_elapsed_seconds += perf_counter() - started_at

    def _entity_key(self, chunk_id: str, node_id: str, label: str, properties: dict[str, Any]) -> str:
        name = properties.get("name")
        identity = (
            f"{self._collection_name}\x00{label.casefold()}\x00{' '.join(name.split()).casefold()}"
            if isinstance(name, str) and name.strip()
            else f"{self._collection_name}\x00{chunk_id}\x00{node_id}"
        )
        return hashlib.sha256(identity.encode()).hexdigest()

    def _write_entity_batch(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        with self.driver.session(database=self.neo4j_database) as session:
            session.execute_write(self._write_entity_batch_tx, rows, self._collection_name)

    @staticmethod
    def _write_entity_batch_tx(tx: neo4j.Transaction, rows: list[dict[str, Any]], collection_name: str) -> None:
        tx.run(
            "UNWIND $rows AS row "
            "CALL apoc.merge.node(row.labels, {ai4rag_entity_key: row.key}, {}) YIELD node "
            "SET node += row.properties, node.ai4rag_entity_key = row.key "
            "SET node.ai4rag_kg_collections = CASE "
            "WHEN $col IN COALESCE(node.ai4rag_kg_collections, []) THEN node.ai4rag_kg_collections "
            "ELSE COALESCE(node.ai4rag_kg_collections, []) + $col END",
            rows=rows,
            col=collection_name,
        ).consume()

    def _write_relationship_batch(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        with self.driver.session(database=self.neo4j_database) as session:
            session.execute_write(self._write_relationship_batch_tx, rows, self._collection_name)

    @staticmethod
    def _write_relationship_batch_tx(tx: neo4j.Transaction, rows: list[dict[str, Any]], collection_name: str) -> None:
        tx.run(
            "UNWIND $rows AS row "
            "MATCH (start:__Entity__ {ai4rag_entity_key: row.start_key}) "
            "WITH start, row MATCH (end:__Entity__ {ai4rag_entity_key: row.end_key}) "
            "CALL apoc.merge.relationship(start, row.type, {}, row.properties, end, row.properties) YIELD rel "
            "SET rel.ai4rag_kg_collections = CASE "
            "WHEN $col IN COALESCE(rel.ai4rag_kg_collections, []) THEN rel.ai4rag_kg_collections "
            "ELSE COALESCE(rel.ai4rag_kg_collections, []) + $col END",
            rows=rows,
            col=collection_name,
        ).consume()

    def _write_link_batch(self, rows: list[dict[str, str]]) -> None:
        if not rows:
            return
        with self.driver.session(database=self.neo4j_database) as session:
            session.execute_write(self._write_link_batch_tx, rows, self._collection_name)

    @staticmethod
    def _write_link_batch_tx(tx: neo4j.Transaction, rows: list[dict[str, str]], collection_name: str) -> None:
        tx.run(
            "UNWIND $rows AS row "
            "MATCH (e:__Entity__ {ai4rag_entity_key: row.key}) "
            f"MATCH (c:`{collection_name}`:{_SCHEMA.CHUNK_LABEL} "
            f"{{{_SCHEMA.COLLECTION}: $col, {_SCHEMA.ID}: row.chunk_id}}) "
            f"MERGE (e)-[:{_SCHEMA.FROM_CHUNK_RELATIONSHIP}]->(c)",
            rows=rows,
            col=collection_name,
        ).consume()

    async def run(
        self,
        graph: Neo4jGraph | dict[str, Any],
        lexical_graph_config: LexicalGraphConfig = LexicalGraphConfig(),
    ) -> KGWriterModel:
        # The pipeline serializes component results before passing them to the
        # next component, so pruner.graph arrives as a dictionary.
        if isinstance(graph, dict):
            graph = Neo4jGraph.model_validate(graph)
        documents = [node for node in graph.nodes if node.label == lexical_graph_config.document_node_label]
        entity_nodes = [
            node for node in graph.nodes if node.label not in lexical_graph_config.lexical_graph_node_labels
        ]
        chunk_id = self._chunk_id.get()
        if len(documents) > 1:
            raise ValueError("KG extraction produced multiple document nodes for one canonical chunk.")
        if documents:
            document_chunk_id = documents[0].properties.get("ai4rag_chunk_id")
            if chunk_id is not None and document_chunk_id is not None and document_chunk_id != chunk_id:
                raise ValueError("KG extraction document metadata does not match the canonical chunk ID.")
            chunk_id = chunk_id or document_chunk_id
        if not chunk_id:
            raise ValueError("KG extraction requires the canonical chunk ID.")

        entity_ids = {node.id for node in entity_nodes}
        # This pipeline processes one canonical chunk per run. GraphRAG may omit
        # lexical nodes and FROM_CHUNK edges, but all extracted entities still
        # belong to that chunk.
        entity_graph = Neo4jGraph(
            nodes=entity_nodes,
            relationships=[
                rel for rel in graph.relationships if rel.start_node_id in entity_ids and rel.end_node_id in entity_ids
            ],
        )
        self.write_graphs([(chunk_id, entity_graph)])
        return KGWriterModel(status="SUCCESS", metadata={"files": [], "statistics": {}})


def _build_graph_retrieval_query(
    include_entity_neighbors: bool,
    entity_neighbor_limit: int,
    entity_pivot_limit: int = 3,
    entity_relationship_hops: int = 1,
    relationship_neighbor_limit: int = 5,
) -> str:
    """Build the Cypher retrieval query for :class:`VectorCypherRetriever`.

    The query receives seed ``Chunk`` nodes from vector search, ranks their
    graph neighbors by query similarity and graph support, then returns
    distinct chunks as individual results. A seed with no graph neighbors is
    returned as a fallback. Candidate chunks are scored using their own
    embeddings rather than inheriting a seed's ANN score.
    """
    # $col is passed via query_params in _search_graph to scope graph expansion
    # to one collection. The seed index is already collection-specific.
    collection_filter = f"WHERE node.{_SCHEMA.COLLECTION} = $col "

    if include_entity_neighbors and entity_neighbor_limit:
        entity_block = (
            "CALL (node) { "
            f"OPTIONAL MATCH (entity:{_SCHEMA.ENTITY_LABEL})-[:{_SCHEMA.FROM_CHUNK_RELATIONSHIP}]->(node) "
            f"OPTIONAL MATCH (entity)-[:{_SCHEMA.FROM_CHUNK_RELATIONSHIP}]->(ent_nb:{_SCHEMA.CHUNK_LABEL}) "
            f"WHERE ent_nb <> node AND ent_nb.{_SCHEMA.COLLECTION} = $col "
            "WITH ent_nb, count(DISTINCT entity) AS overlap "
            f"WHERE ent_nb IS NOT NULL AND ent_nb.{_SCHEMA.EMBEDDING} IS NOT NULL "
            f"WITH ent_nb, overlap, vector.similarity.cosine(ent_nb.{_SCHEMA.EMBEDDING}, $query_vector) AS similarity "
            f"ORDER BY similarity DESC, overlap DESC, ent_nb.{_SCHEMA.ID} ASC "
            f"LIMIT {entity_neighbor_limit} "
            "RETURN collect({chunk: ent_nb, strength: toFloat(overlap)}) AS ent_hits } "
        )
    else:
        entity_block = "WITH node, [] AS ent_hits "

    if entity_pivot_limit and entity_relationship_hops and relationship_neighbor_limit:
        relationship_block = (
            "CALL (node) { "
            f"MATCH (pivot:{_SCHEMA.ENTITY_LABEL})-[:{_SCHEMA.FROM_CHUNK_RELATIONSHIP}]->(node) "
            f"OPTIONAL MATCH (pivot)-[:{_SCHEMA.FROM_CHUNK_RELATIONSHIP}]->(pivot_chunk:{_SCHEMA.CHUNK_LABEL}) "
            f"WHERE pivot_chunk.{_SCHEMA.COLLECTION} = $col "
            "WITH node, pivot, count(DISTINCT pivot_chunk) AS degree "
            f"ORDER BY degree DESC, pivot.name ASC, pivot.{_SCHEMA.ID} ASC "
            f"LIMIT {entity_pivot_limit} "
            f"MATCH path = (pivot)-[*1..{entity_relationship_hops}]-(related:{_SCHEMA.ENTITY_LABEL}) "
            f"WHERE ALL(rel IN relationships(path) WHERE type(rel) <> '{_SCHEMA.FROM_CHUNK_RELATIONSHIP}' "
            f"AND $col IN COALESCE(rel.{_SCHEMA.KG_COLLECTIONS}, [])) "
            f"MATCH (related)-[:{_SCHEMA.FROM_CHUNK_RELATIONSHIP}]->(rel_nb:{_SCHEMA.CHUNK_LABEL}) "
            f"WHERE rel_nb <> node AND rel_nb.{_SCHEMA.COLLECTION} = $col "
            f"AND rel_nb.{_SCHEMA.EMBEDDING} IS NOT NULL "
            "WITH rel_nb, count(DISTINCT path) AS path_count, min(length(path)) AS hops "
            "WITH rel_nb, path_count, hops, "
            f"vector.similarity.cosine(rel_nb.{_SCHEMA.EMBEDDING}, $query_vector) AS similarity "
            f"ORDER BY similarity DESC, path_count DESC, hops ASC, rel_nb.{_SCHEMA.ID} ASC "
            f"LIMIT {relationship_neighbor_limit} "
            "RETURN collect({chunk: rel_nb, strength: toFloat(path_count) / toFloat(hops)}) AS rel_hits } "
        )
    else:
        relationship_block = "WITH node, ent_hits, [] AS rel_hits "

    # Give independent query similarity most of the weight; bound accumulated
    # graph support so high-degree entities cannot dominate the local route.
    return (
        collection_filter
        + entity_block
        + relationship_block
        + "WITH node, ent_hits + rel_hits AS graph_hits "
        + "WITH CASE WHEN size(graph_hits) = 0 "
        + "THEN [{chunk: node, strength: 0.0}] ELSE graph_hits END AS hits "
        + "UNWIND hits AS hit "
        + "WITH hit.chunk AS candidate, sum(hit.strength) AS graph_strength "
        + "WITH candidate, graph_strength, "
        + f"coalesce(vector.similarity.cosine(candidate.{_SCHEMA.EMBEDDING}, $query_vector), 0.0) AS semantic_score "
        + "WITH candidate, graph_strength, "
        + "0.8 * semantic_score + 0.2 * graph_strength / (1.0 + graph_strength) AS score "
        + f"ORDER BY score DESC, graph_strength DESC, candidate.{_SCHEMA.ID} ASC LIMIT $top_k "
        + f"RETURN candidate.{_SCHEMA.TEXT} AS text, score, "
        + f"candidate.{_SCHEMA.DOCUMENT_ID} AS document_id, candidate.{_SCHEMA.METADATA} AS metadata"
    )


def _validate_neo4j_search_params(
    search_mode: str,
    ranker_strategy: str | None = None,
    ranker_k: int | None = None,
    ranker_alpha: float | None = None,
    store_class: type | None = None,
    **kwargs: Any,
) -> None:
    # pylint: disable=duplicate-code
    validate_search_params(
        search_mode,
        ranker_strategy,
        ranker_k,
        ranker_alpha,
        supported_modes=("graph",),
        store_class=store_class,
    )
    # pylint: enable=duplicate-code

    if search_mode == "graph":
        graph_hops = kwargs.get("graph_hops", 0)
        entity_neighbor_limit = kwargs.get("entity_neighbor_limit", 5)
        entity_pivot_limit = kwargs.get("entity_pivot_limit", 3)
        entity_relationship_hops = kwargs.get("entity_relationship_hops", 1)
        relationship_neighbor_limit = kwargs.get("relationship_neighbor_limit", 5)
        if not isinstance(graph_hops, int) or graph_hops != 0:
            raise ValueError(
                f"graph_hops must be 0 because Neo4j graph search does not expand NEXT_CHUNK neighbors, "
                f"got {graph_hops!r}."
            )
        if not isinstance(entity_neighbor_limit, int) or entity_neighbor_limit < 0:
            raise ValueError(f"entity_neighbor_limit must be a non-negative integer, got {entity_neighbor_limit!r}.")
        for name, value in {
            "entity_pivot_limit": entity_pivot_limit,
            "entity_relationship_hops": entity_relationship_hops,
            "relationship_neighbor_limit": relationship_neighbor_limit,
        }.items():
            if not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer, got {value!r}.")
