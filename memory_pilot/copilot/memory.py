import numpy as np
import time
from functools import wraps
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    HnswAlgorithmConfiguration,
    SearchableField,
    SearchFieldDataType,
    SearchIndex,
    SemanticConfiguration,
    SemanticField,
    SemanticPrioritizedFields,
    SemanticSearch,
    SimpleField,
    VectorSearch,
    SearchField,
    VectorSearchProfile,
)
from azure.search.documents.models import VectorizedQuery
from dataclasses import dataclass


@dataclass
class MemoryCfg:
    endpoint: str
    credential: str
    index_name: str


def retry(max_retries, base_delay=1, backoff_factor=2):
    def decorator_retry(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            delay = base_delay
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    print(f"Exception caught: {e}")
                    retries += 1
                    if retries >= max_retries:
                        raise
                    print(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                    delay *= backoff_factor

        return wrapper

    return decorator_retry


class CopilotIndex:

    def __init__(
        self,
        index_name,
        vector_field_name="document_vector",
        semantic_config_name="rdw-semantic-config",
        vector_config_name="rdw-vector-config",
    ):
        self.index_name = index_name
        self.vector_field_name = vector_field_name

        self.vector_config_name = vector_config_name
        self.semantic_config_name = semantic_config_name

    @property
    def fields(self):
        fields = [
            SimpleField(name="id", type=SearchFieldDataType.String, key=True),
            SearchableField(
                name="title", type=SearchFieldDataType.String, sortable=True
            ),
            SearchableField(
                name="body",
                type=SearchFieldDataType.String,
                analyzer_name="en.lucene",
            ),
            SearchableField(
                name="category",
                type=SearchFieldDataType.String,
                facetable=True,
                filterable=True,
                sortable=True,
            ),
            SearchField(
                name=self.vector_field_name,
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=1536,
                vector_search_profile_name=self.vector_config_name,
            ),
        ]
        return fields

    def vector_search(self):
        return VectorSearch(
            profiles=[
                VectorSearchProfile(
                    name=self.vector_config_name, algorithm_configuration_name="hnsw"
                )
            ],
            algorithms=[HnswAlgorithmConfiguration(name="hnsw")],
        )

    def semantic_search(self):
        semantic_config = SemanticConfiguration(
            name=self.semantic_config_name,
            prioritized_fields=SemanticPrioritizedFields(
                title_field=SemanticField(field_name="title"),
                content_fields=[SemanticField(field_name="body")],
                keywords_fields=[SemanticField(field_name="category")],
            ),
        )
        return SemanticSearch(configurations=[semantic_config])

    def new(self):
        return SearchIndex(
            name=self.index_name,
            fields=self.fields,
            vector_search=self.vector_search(),
            semantic_search=self.semantic_search(),
        )


class CopilotMemory:

    def __init__(self, kernel, cfg: MemoryCfg):
        self.kernel = kernel
        credential = AzureKeyCredential(cfg.credential)
        self.copilot_index = CopilotIndex(index_name=cfg.index_name)
        self.search_index_client = SearchIndexClient(
            endpoint=cfg.endpoint, credential=credential
        )
        self.search_client = SearchClient(
            endpoint=cfg.endpoint,
            index_name=self.copilot_index.index_name,
            credential=credential,
        )

    def create_index(self):
        self.search_index_client.create_index(self.copilot_index.new())

    def delete_index(self):
        self.search_index_client.delete_index(self.copilot_index.new())

    async def upload_documents(self, documents):
        documents_to_add = []
        for document in documents:
            document = await self.update_document_with_embedding(document, "body")
            documents_to_add.append(document)
        self.search_client.upload_documents(documents_to_add)

    @retry(5)
    async def generate_embedding(self, texts: list[str]) -> np.ndarray:
        vectors = await self.kernel.services[
            "text-embedding-ada-002"
        ].generate_embeddings(texts)
        return vectors[0]

    async def update_document_with_embedding(self, document, field_name):
        vector = await self.generate_embedding([document[field_name]])
        document[self.copilot_index.vector_field_name] = vector.tolist()
        return document

    async def search(self, query, top_k=5):
        vector = await self.generate_embedding(query)
        vector_queries = [
            VectorizedQuery(
                vector=vector,
                k_nearest_neighbors=top_k,
                fields=self.copilot_index.vector_field_name,
            )
        ]
        results = self.search_client.search(
            query_type="semantic",
            search_text=query,
            semantic_configuration_name=self.copilot_index.semantic_config_name,
            vector_queries=vector_queries,
            select="title,body,category",
            query_caption="extractive",
            top=top_k,
        )
        return results

    async def search_vector_only(self, query, top_k=5):
        vector = await self.generate_embedding(query)
        vector_queries = [
            VectorizedQuery(
                vector=vector,
                k_nearest_neighbors=top_k,
                fields=self.copilot_index.vector_field_name,
            )
        ]
        results = self.search_client.search(
            search_text=None,
            vector_queries=vector_queries,
            select="title,body,category",
            top=top_k,
        )
        return results
