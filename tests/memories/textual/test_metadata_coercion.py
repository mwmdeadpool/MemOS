"""Regression tests for TextualMemoryMetadata JSON-string coercion.

Graph backends (Neo4j) serialize nested dict properties to JSON strings on
write. Without before-validators on `info` / `internal_info`, the round-trip
through `TextualMemoryItem.from_dict` fails with a Pydantic ValidationError
and silently drops search hits at the API layer.
"""

import unittest
import uuid

from unittest.mock import MagicMock

from memos.memories.textual.item import (
    TextualMemoryItem,
    TextualMemoryMetadata,
)


class MetadataJsonStringCoercionTest(unittest.TestCase):
    def _item_kwargs(self, **metadata):
        return {
            "id": str(uuid.uuid4()),
            "memory": "hello",
            "metadata": metadata,
        }

    def test_internal_info_accepts_json_string(self):
        payload = '{"ingest_batch_id": "ca4d1234-aaaa-bbbb-cccc-1c651d264833", "chunk_index": 0}'
        item = TextualMemoryItem.from_dict(
            self._item_kwargs(internal_info=payload)
        )
        self.assertIsInstance(item.metadata.internal_info, dict)
        self.assertEqual(
            item.metadata.internal_info["ingest_batch_id"],
            "ca4d1234-aaaa-bbbb-cccc-1c651d264833",
        )
        self.assertEqual(item.metadata.internal_info["chunk_index"], 0)

    def test_info_accepts_json_string(self):
        item = TextualMemoryItem.from_dict(
            self._item_kwargs(info='{"foo": "bar"}')
        )
        self.assertEqual(item.metadata.info, {"foo": "bar"})

    def test_dict_passthrough_unchanged(self):
        payload = {"ingest_batch_id": "abc", "extra": [1, 2]}
        meta = TextualMemoryMetadata(internal_info=payload)
        self.assertEqual(meta.internal_info, payload)

    def test_none_stays_none(self):
        meta = TextualMemoryMetadata(internal_info=None, info=None)
        self.assertIsNone(meta.internal_info)
        self.assertIsNone(meta.info)

    def test_unparseable_string_coerces_to_none(self):
        # Bad JSON should not raise; field is internal — degrade gracefully.
        meta = TextualMemoryMetadata(internal_info="{not valid json")
        self.assertIsNone(meta.internal_info)

    def test_non_object_json_coerces_to_none(self):
        # Lists / scalars aren't valid metadata shapes; coerce to None.
        meta = TextualMemoryMetadata(internal_info="[1, 2, 3]")
        self.assertIsNone(meta.internal_info)

    def test_empty_string_coerces_to_none(self):
        meta = TextualMemoryMetadata(internal_info="")
        self.assertIsNone(meta.internal_info)

    def test_bytes_payload_decoded(self):
        meta = TextualMemoryMetadata(
            internal_info=b'{"ingest_batch_id": "from-bytes"}'
        )
        self.assertEqual(meta.internal_info, {"ingest_batch_id": "from-bytes"})

    def test_e2e_search_keeps_text_mem_bucket_with_stringified_internal_info(self):
        from memos.api.product_models import APISearchRequest
        from memos.api.handlers.formatters_handler import format_memory_item, post_process_textual_mem
        from memos.search.search_service import search_text_memories
        from memos.types import SearchMode, UserContext

        mock_text_mem = MagicMock()
        mock_text_mem.search.return_value = [
            TextualMemoryItem.from_dict(
                self._item_kwargs(
                    internal_info='{"ingest_batch_id":"batch-1","chunk_index":1}',
                    user_id="u1",
                    session_id="s1",
                    memory_type="WorkingMemory",
                    relativity=0.9,
                )
            )
        ]
        user_context = UserContext(user_id="u1", mem_cube_id="cube_test", session_id="s1")

        req = APISearchRequest(
            query="batch",
            user_id="u1",
            mem_cube_id="cube_test",
            mode="fast",
            dedup="no",
        )
        # Regression boundary: `_search_text` swallows exceptions and returns [].
        # This emulates that control flow and verifies stringified internal_info
        # no longer triggers an exception that would zero out text_mem.
        try:
            search_results = search_text_memories(
                text_mem=mock_text_mem,
                search_req=req,
                user_context=user_context,
                mode=SearchMode.FAST,
                include_embedding=False,
            )
            all_formatted = [format_memory_item(item, include_embedding=False) for item in search_results]
        except Exception:
            all_formatted = []

        result = post_process_textual_mem(
            memories_result={
                "text_mem": [],
                "act_mem": [],
                "para_mem": [],
                "pref_mem": [],
                "pref_note": "",
                "tool_mem": [],
                "skill_mem": [],
            },
            text_formatted_mem=all_formatted,
            mem_cube_id="cube_test",
        )

        self.assertEqual(len(result["text_mem"]), 1)
        self.assertEqual(result["text_mem"][0]["total_nodes"], 1)
        self.assertEqual(len(result["text_mem"][0]["memories"]), 1)
        self.assertEqual(
            result["text_mem"][0]["memories"][0]["metadata"]["internal_info"]["ingest_batch_id"],
            "batch-1",
        )


if __name__ == "__main__":
    unittest.main()
