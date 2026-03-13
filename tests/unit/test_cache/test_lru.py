"""Tests for LRU prompt cache with prefix trie (§6.3, FR5, AC5).

Patterns: happy path, edge cases, negative path, boundary/limit,
stateful/lifecycle, corner cases, alternate flows.
"""

from __future__ import annotations

from mlxs.prompt_cache.lru import LRUPromptCache, _PrefixTrie

# =============================================================================
# _PrefixTrie
# =============================================================================


class TestPrefixTrieHappyPath:
    def test_insert_and_lookup(self) -> None:
        trie = _PrefixTrie()
        trie.insert("m1", (1, 2, 3))
        key, length = trie.longest_prefix("m1", (1, 2, 3, 4, 5))
        assert key == ("m1", (1, 2, 3))
        assert length == 3

    def test_multiple_prefixes(self) -> None:
        trie = _PrefixTrie()
        trie.insert("m1", (1, 2))
        trie.insert("m1", (1, 2, 3, 4))
        key, length = trie.longest_prefix("m1", (1, 2, 3, 4, 5))
        assert key == ("m1", (1, 2, 3, 4))
        assert length == 4

    def test_shorter_prefix_returned_when_longer_absent(self) -> None:
        trie = _PrefixTrie()
        trie.insert("m1", (1, 2))
        key, length = trie.longest_prefix("m1", (1, 2, 99))
        assert key == ("m1", (1, 2))
        assert length == 2


class TestPrefixTrieNegativePath:
    def test_no_match(self) -> None:
        trie = _PrefixTrie()
        trie.insert("m1", (1, 2, 3))
        key, length = trie.longest_prefix("m1", (9, 8, 7))
        assert key is None
        assert length == 0

    def test_wrong_model_id(self) -> None:
        trie = _PrefixTrie()
        trie.insert("m1", (1, 2, 3))
        key, length = trie.longest_prefix("m2", (1, 2, 3))
        assert key is None
        assert length == 0


class TestPrefixTrieEdgeCases:
    def test_empty_tokens_query(self) -> None:
        trie = _PrefixTrie()
        trie.insert("m1", (1, 2))
        key, length = trie.longest_prefix("m1", ())
        assert key is None
        assert length == 0

    def test_remove_entry(self) -> None:
        trie = _PrefixTrie()
        trie.insert("m1", (1, 2, 3))
        trie.remove("m1", (1, 2, 3))
        key, _ = trie.longest_prefix("m1", (1, 2, 3, 4))
        assert key is None

    def test_remove_nonexistent_no_crash(self) -> None:
        trie = _PrefixTrie()
        trie.remove("m1", (99, 99))  # should not raise

    def test_remove_one_model_preserves_other(self) -> None:
        trie = _PrefixTrie()
        trie.insert("m1", (1, 2))
        trie.insert("m2", (1, 2))
        trie.remove("m1", (1, 2))
        key_m1, _ = trie.longest_prefix("m1", (1, 2, 3))
        assert key_m1 is None
        key_m2, _ = trie.longest_prefix("m2", (1, 2, 3))
        assert key_m2 == ("m2", (1, 2))

    def test_single_token_prefix(self) -> None:
        trie = _PrefixTrie()
        trie.insert("m1", (42,))
        key, length = trie.longest_prefix("m1", (42, 100))
        assert key == ("m1", (42,))
        assert length == 1


# =============================================================================
# LRUPromptCache
# =============================================================================


class TestLRUHappyPath:
    def test_put_and_get(self) -> None:
        cache = LRUPromptCache(max_entries=10)
        cache.put("m1", (1, 2, 3), [{"k": "v"}])
        state, prefix_len = cache.get("m1", (1, 2, 3, 4))
        assert state is not None
        assert prefix_len == 3

    def test_get_returns_deep_copy(self) -> None:
        cache = LRUPromptCache(max_entries=10)
        original = [{"k": "v"}]
        cache.put("m1", (1, 2), original)
        state1, _ = cache.get("m1", (1, 2, 3))
        state2, _ = cache.get("m1", (1, 2, 3))
        assert state1 is not state2  # deep copies
        state1[0]["k"] = "mutated"
        state3, _ = cache.get("m1", (1, 2, 3))
        assert state3[0]["k"] == "v"  # original unmodified

    def test_longest_prefix_match(self) -> None:
        cache = LRUPromptCache(max_entries=10)
        cache.put("m1", (1, 2), ["short"])
        cache.put("m1", (1, 2, 3, 4), ["long"])
        state, prefix_len = cache.get("m1", (1, 2, 3, 4, 5))
        assert prefix_len == 4
        assert state == ["long"]

    def test_stats_hit_miss(self) -> None:
        cache = LRUPromptCache(max_entries=10)
        cache.put("m1", (1, 2), ["x"])
        cache.get("m1", (1, 2, 3))  # hit
        cache.get("m1", (99,))  # miss
        s = cache.stats()
        assert s.hit_count == 1
        assert s.miss_count == 1
        assert s.entry_count == 1


class TestLRUNegativePath:
    def test_get_empty_cache(self) -> None:
        cache = LRUPromptCache(max_entries=10)
        state, prefix_len = cache.get("m1", (1, 2, 3))
        assert state is None
        assert prefix_len == 0

    def test_get_wrong_model_id(self) -> None:
        cache = LRUPromptCache(max_entries=10)
        cache.put("m1", (1, 2), ["x"])
        state, _ = cache.get("m2", (1, 2, 3))
        assert state is None

    def test_get_no_prefix_match(self) -> None:
        cache = LRUPromptCache(max_entries=10)
        cache.put("m1", (1, 2), ["x"])
        state, _ = cache.get("m1", (99, 100))
        assert state is None


# -- Boundary / limit cases ---------------------------------------------------


class TestLRUBoundary:
    def test_max_entries_one(self) -> None:
        cache = LRUPromptCache(max_entries=1)
        cache.put("m1", (1,), ["a"])
        cache.put("m1", (2,), ["b"])  # evicts (1,)
        assert cache.stats().entry_count == 1
        state, _ = cache.get("m1", (1, 99))
        assert state is None
        state, _ = cache.get("m1", (2, 99))
        assert state == ["b"]

    def test_max_bytes_eviction(self) -> None:
        """Entries exceeding max_bytes are evicted."""

        class FakeCache:
            state_size_bytes = 100

        cache = LRUPromptCache(max_entries=100, max_bytes=150)
        cache.put("m1", (1,), [FakeCache()])
        cache.put("m1", (2,), [FakeCache()])  # total 200 > 150, evict (1,)
        assert cache.stats().entry_count == 1
        assert cache.stats().eviction_count == 1

    def test_trim_zero(self) -> None:
        cache = LRUPromptCache(max_entries=10)
        cache.put("m1", (1,), ["x"])
        removed = cache.trim(0)
        assert removed == 0
        assert cache.stats().entry_count == 1

    def test_trim_more_than_entries(self) -> None:
        cache = LRUPromptCache(max_entries=10)
        cache.put("m1", (1,), ["x"])
        cache.put("m1", (2,), ["y"])
        removed = cache.trim(100)
        assert removed == 2
        assert cache.stats().entry_count == 0

    def test_clear(self) -> None:
        cache = LRUPromptCache(max_entries=10)
        for i in range(5):
            cache.put("m1", (i,), [f"v{i}"])
        cache.clear()
        assert cache.stats().entry_count == 0
        assert cache.stats().total_bytes == 0


# -- Stateful / lifecycle -----------------------------------------------------


class TestLRUStateful:
    def test_lru_eviction_order(self) -> None:
        """Least recently used entry evicted first."""
        cache = LRUPromptCache(max_entries=3)
        cache.put("m1", (1,), ["a"])
        cache.put("m1", (2,), ["b"])
        cache.put("m1", (3,), ["c"])

        # Access (1,) to make it recently used
        cache.get("m1", (1, 99))

        # Adding (4,) should evict (2,) — the least recently used
        cache.put("m1", (4,), ["d"])

        state_2, _ = cache.get("m1", (2, 99))
        assert state_2 is None  # evicted
        state_1, _ = cache.get("m1", (1, 99))
        assert state_1 is not None  # still present

    def test_put_updates_existing_entry(self) -> None:
        """Re-putting with same key updates value without increasing count."""
        cache = LRUPromptCache(max_entries=10)
        cache.put("m1", (1, 2), ["old"])
        cache.put("m1", (1, 2), ["new"])
        assert cache.stats().entry_count == 1
        state, _ = cache.get("m1", (1, 2, 3))
        assert state == ["new"]

    def test_access_count_increments(self) -> None:
        cache = LRUPromptCache(max_entries=10)
        cache.put("m1", (1,), ["x"])
        cache.get("m1", (1, 2))
        cache.get("m1", (1, 3))
        s = cache.stats()
        assert s.hit_count == 2

    def test_eviction_count_tracks_all_evictions(self) -> None:
        cache = LRUPromptCache(max_entries=2)
        cache.put("m1", (1,), ["a"])
        cache.put("m1", (2,), ["b"])
        cache.put("m1", (3,), ["c"])  # evict 1
        cache.put("m1", (4,), ["d"])  # evict 2
        assert cache.stats().eviction_count == 2

    def test_trim_removes_lru_first(self) -> None:
        cache = LRUPromptCache(max_entries=10)
        cache.put("m1", (1,), ["a"])
        cache.put("m1", (2,), ["b"])
        cache.put("m1", (3,), ["c"])
        cache.get("m1", (3, 99))  # make (3,) recently used

        cache.trim(2)  # should remove (1,) and (2,)

        state_1, _ = cache.get("m1", (1, 99))
        assert state_1 is None
        state_2, _ = cache.get("m1", (2, 99))
        assert state_2 is None
        state_3, _ = cache.get("m1", (3, 99))
        assert state_3 is not None


# -- Corner cases --------------------------------------------------------------


class TestLRUCornerCases:
    def test_same_prefix_different_models(self) -> None:
        cache = LRUPromptCache(max_entries=10)
        cache.put("m1", (1, 2), ["model1"])
        cache.put("m2", (1, 2), ["model2"])
        assert cache.stats().entry_count == 2

        state1, _ = cache.get("m1", (1, 2, 3))
        assert state1 == ["model1"]
        state2, _ = cache.get("m2", (1, 2, 3))
        assert state2 == ["model2"]

    def test_put_deep_copies_input(self) -> None:
        """Mutating the input after put() doesn't corrupt the cache."""
        cache = LRUPromptCache(max_entries=10)
        data = [{"key": "original"}]
        cache.put("m1", (1,), data)
        data[0]["key"] = "mutated"
        state, _ = cache.get("m1", (1, 2))
        assert state[0]["key"] == "original"

    def test_max_entries_zero_allows_nothing(self) -> None:
        """max_entries=0 means nothing is cached."""
        cache = LRUPromptCache(max_entries=0)
        cache.put("m1", (1,), ["x"])
        assert cache.stats().entry_count == 0

    def test_max_bytes_none_means_unlimited(self) -> None:
        class FakeCache:
            state_size_bytes = 999999999

        cache = LRUPromptCache(max_entries=100, max_bytes=None)
        cache.put("m1", (1,), [FakeCache()])
        cache.put("m1", (2,), [FakeCache()])
        assert cache.stats().entry_count == 2  # no byte-based eviction
