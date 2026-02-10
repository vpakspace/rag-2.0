"""Unit tests for VectorStore (mock Neo4j driver)."""

from unittest.mock import MagicMock, patch

import pytest

from core.models import Chunk
from storage.vector_store import VectorStore, INDEX_NAME, NODE_LABEL


@pytest.fixture
def mock_driver():
    driver = MagicMock()
    session = MagicMock()
    driver.session.return_value.__enter__ = MagicMock(return_value=session)
    driver.session.return_value.__exit__ = MagicMock(return_value=False)
    return driver, session


@pytest.fixture
def store(mock_driver):
    driver, _ = mock_driver
    return VectorStore(driver=driver)


class TestVectorStoreInit:
    def test_init_index(self, store, mock_driver):
        _, session = mock_driver
        store.init_index()
        session.run.assert_called_once()
        call_args = session.run.call_args
        assert "CREATE VECTOR INDEX" in call_args[0][0]
        assert INDEX_NAME in call_args[0][0]

    def test_custom_driver(self):
        driver = MagicMock()
        vs = VectorStore(driver=driver)
        assert vs._driver is driver

    def test_default_driver(self):
        with patch.dict("sys.modules", {"neo4j": MagicMock()}):
            import importlib
            import storage.vector_store as vs_mod

            importlib.reload(vs_mod)
            store = vs_mod.VectorStore()
            assert store._driver is not None


class TestAddChunks:
    def test_add_empty_list(self, store):
        result = store.add_chunks([])
        assert result == 0

    def test_add_single_chunk(self, store, mock_driver):
        _, session = mock_driver
        chunk = Chunk(
            id="c1",
            content="Test content",
            context="Test context",
            embedding=[0.1, 0.2, 0.3],
            metadata={"source": "test"},
        )
        result = store.add_chunks([chunk])
        assert result == 1
        session.run.assert_called_once()
        call_kwargs = session.run.call_args[1]
        assert call_kwargs["id"] == "c1"
        assert call_kwargs["content"] == "Test content"
        assert call_kwargs["context"] == "Test context"

    def test_add_chunk_auto_id(self, store, mock_driver):
        _, session = mock_driver
        chunk = Chunk(content="Auto ID content", embedding=[0.1])
        store.add_chunks([chunk])
        call_kwargs = session.run.call_args[1]
        assert len(call_kwargs["id"]) == 32  # md5 hex

    def test_add_multiple_chunks(self, store, mock_driver):
        _, session = mock_driver
        chunks = [
            Chunk(id=f"c{i}", content=f"Content {i}", embedding=[0.1])
            for i in range(5)
        ]
        result = store.add_chunks(chunks)
        assert result == 5
        assert session.run.call_count == 5

    def test_enriched_content_stored(self, store, mock_driver):
        _, session = mock_driver
        chunk = Chunk(
            id="c1",
            content="Main text",
            context="This chunk is about X",
            embedding=[0.1],
        )
        store.add_chunks([chunk])
        call_kwargs = session.run.call_args[1]
        assert call_kwargs["enriched_content"] == "This chunk is about X\n\nMain text"


class TestSearch:
    def test_search_returns_results(self, store, mock_driver):
        _, session = mock_driver
        mock_record = {
            "id": "c1",
            "content": "Found content",
            "context": "Found context",
            "metadata": "{}",
            "score": 0.95,
        }
        session.run.return_value = [mock_record]

        results = store.search([0.1, 0.2, 0.3], top_k=5)
        assert len(results) == 1
        assert results[0].chunk.id == "c1"
        assert results[0].chunk.content == "Found content"
        assert results[0].score == 0.95
        assert results[0].rank == 1

    def test_search_empty_results(self, store, mock_driver):
        _, session = mock_driver
        session.run.return_value = []
        results = store.search([0.1], top_k=5)
        assert results == []

    def test_search_uses_default_top_k(self, store, mock_driver):
        _, session = mock_driver
        session.run.return_value = []
        store.search([0.1])
        call_kwargs = session.run.call_args[1]
        assert call_kwargs["top_k"] == 20  # default from settings

    def test_search_ranking(self, store, mock_driver):
        _, session = mock_driver
        records = [
            {"id": f"c{i}", "content": f"Content {i}", "context": "", "metadata": "{}", "score": 0.9 - i * 0.1}
            for i in range(3)
        ]
        session.run.return_value = records
        results = store.search([0.1], top_k=3)
        assert len(results) == 3
        assert results[0].rank == 1
        assert results[1].rank == 2
        assert results[2].rank == 3


class TestDeleteAndCount:
    def test_delete_all(self, store, mock_driver):
        _, session = mock_driver
        mock_record = MagicMock()
        mock_record.__getitem__ = lambda self, key: 10
        session.run.return_value.single.return_value = mock_record
        result = store.delete_all()
        assert result == 10

    def test_delete_all_empty(self, store, mock_driver):
        _, session = mock_driver
        session.run.return_value.single.return_value = None
        result = store.delete_all()
        assert result == 0

    def test_count(self, store, mock_driver):
        _, session = mock_driver
        mock_record = MagicMock()
        mock_record.__getitem__ = lambda self, key: 42
        session.run.return_value.single.return_value = mock_record
        result = store.count()
        assert result == 42

    def test_count_empty(self, store, mock_driver):
        _, session = mock_driver
        session.run.return_value.single.return_value = None
        result = store.count()
        assert result == 0

    def test_close(self, store, mock_driver):
        driver, _ = mock_driver
        store.close()
        driver.close.assert_called_once()


class TestChunkModel:
    def test_enriched_content_with_context(self):
        c = Chunk(content="main", context="ctx")
        assert c.enriched_content == "ctx\n\nmain"

    def test_enriched_content_without_context(self):
        c = Chunk(content="main")
        assert c.enriched_content == "main"

    def test_chunk_defaults(self):
        c = Chunk(content="test")
        assert c.id == ""
        assert c.context == ""
        assert c.embedding == []
        assert c.metadata == {}
