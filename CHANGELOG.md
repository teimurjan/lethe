# Changelog

## [0.7.1](https://github.com/teimurjan/lethe/compare/lethe-v0.7.0...lethe-v0.7.1) (2026-04-28)


### Bug Fixes

* **release:** Bump inter-crate version pins to 0.7.0 ([c110aa1](https://github.com/teimurjan/lethe/commit/c110aa1249bb5e848a5febd8df5441d5caeaa8eb))
* **release:** Match component-prefixed tag in release.yml trigger ([35ec1b7](https://github.com/teimurjan/lethe/commit/35ec1b76c54d46df22c63f251d107c176923123b))

## [0.7.0](https://github.com/teimurjan/lethe/compare/lethe-v0.6.0...lethe-v0.7.0) (2026-04-28)


### ⚠ BREAKING CHANGES

* rust port — single `lethe` binary, bindings, polyglot release ([#19](https://github.com/teimurjan/lethe/issues/19))

### Features

* Rust port — single `lethe` binary, bindings, polyglot release ([#19](https://github.com/teimurjan/lethe/issues/19)) ([8f0d1e6](https://github.com/teimurjan/lethe/commit/8f0d1e6af8cd8690cbd0b888cba43a4f72e45c90))


### Bug Fixes

* **release:** Switch root to release-type simple with toml updater ([a3e607c](https://github.com/teimurjan/lethe/commit/a3e607c6d1f5e1f9deb47f9e54093c6dce0f605a))

## [0.6.0](https://github.com/teimurjan/lethe/compare/lethe-memory-v0.5.1...lethe-memory-v0.6.0) (2026-04-24)


### Features

* Regex BM25 tokenizer (+3.68pp NDCG, +6.79pp Recall) ([#17](https://github.com/teimurjan/lethe/issues/17)) ([c17828c](https://github.com/teimurjan/lethe/commit/c17828ce3eeb7178377934c7a9c33af454cfddc8))

## [0.5.1](https://github.com/teimurjan/lethe/compare/lethe-memory-v0.5.0...lethe-memory-v0.5.1) (2026-04-24)


### Bug Fixes

* Perf micro-wins in the index and save hot paths ([#15](https://github.com/teimurjan/lethe/issues/15)) ([a7fcf49](https://github.com/teimurjan/lethe/commit/a7fcf49c67e2684902f533179d97e225bd0dd9e4))

## [0.5.0](https://github.com/teimurjan/lethe/compare/lethe-memory-v0.4.2...lethe-memory-v0.5.0) (2026-04-24)


### Features

* Tui live-expand on highlight, fix duckdb conflict and clipped search input ([2159499](https://github.com/teimurjan/lethe/commit/21594994fa30e4d34eb1e159d2b204248839353d))

## [0.4.2](https://github.com/teimurjan/lethe/compare/lethe-memory-v0.4.1...lethe-memory-v0.4.2) (2026-04-24)


### Bug Fixes

* Lower adaptive deep-pass k_deep from 200 to 100 ([#12](https://github.com/teimurjan/lethe/issues/12)) ([5e8c1db](https://github.com/teimurjan/lethe/commit/5e8c1db87ca44a915493f27cfe31a7811791c311))

## [0.4.1](https://github.com/teimurjan/lethe/compare/lethe-memory-v0.4.0...lethe-memory-v0.4.1) (2026-04-23)


### Bug Fixes

* Tui scope transitions release duckdb handles; document tui install ([#10](https://github.com/teimurjan/lethe/issues/10)) ([609acc9](https://github.com/teimurjan/lethe/commit/609acc935e7b7621b93ccb6b9a4ecb456e3320d4))

## [0.4.0](https://github.com/teimurjan/lethe/compare/lethe-memory-v0.3.0...lethe-memory-v0.4.0) (2026-04-23)


### Features

* Lethe tui, split recall skill, lazy session headers ([#7](https://github.com/teimurjan/lethe/issues/7)) ([b5799b3](https://github.com/teimurjan/lethe/commit/b5799b3089c94f48828c98990f417db63d305db0))

## [0.3.0](https://github.com/teimurjan/lethe/compare/lethe-memory-v0.2.2...lethe-memory-v0.3.0) (2026-04-17)


### Features

* Add --all global search to memory-recall skill ([3ccdafe](https://github.com/teimurjan/lethe/commit/3ccdafe9a2260e997387c0471343a78447c695a2))

## [0.2.2](https://github.com/teimurjan/lethe/compare/lethe-memory-v0.2.1...lethe-memory-v0.2.2) (2026-04-17)


### Bug Fixes

* Remove chunk_map.json, expand reads from DuckDB directly ([5cb8c84](https://github.com/teimurjan/lethe/commit/5cb8c843213d91b5c561e97a5b058d14d52b0652))

## [0.2.1](https://github.com/teimurjan/lethe/compare/lethe-memory-v0.2.0...lethe-memory-v0.2.1) (2026-04-17)


### Bug Fixes

* Add README to PyPI, absolute logo URL, MIT license ([78188dd](https://github.com/teimurjan/lethe/commit/78188dd9642694f441f1a42a4ea99ae42a498dd7))

## [0.2.0](https://github.com/teimurjan/lethe/compare/lethe-memory-v0.1.0...lethe-memory-v0.2.0) (2026-04-17)


### Features

* Add analysis script with plots and summary table ([b1e140b](https://github.com/teimurjan/lethe/commit/b1e140b93a4df4ebd32e483c779b7a8ea093fde5))
* Add BM25 hybrid retrieval, +73% over static on LongMemEval ([7dd3320](https://github.com/teimurjan/lethe/commit/7dd33206804039d6076481e2b8aaff259a314caf))
* Add data preparation script for NFCorpus ([a47ab4b](https://github.com/teimurjan/lethe/commit/a47ab4b793d62b613ef702e1e59207d70998d711))
* Add health metrics (diversity, anchor drift, NDCG, recall) ([e0d13eb](https://github.com/teimurjan/lethe/commit/e0d13eb6a734a31b8760b937e4c621efc09076bd))
* Add MemoryEntry dataclass and Tier enum ([c021bfb](https://github.com/teimurjan/lethe/commit/c021bfb7c6f6c42473c1cb80c7a5f4cf957fc942))
* Add Static and RandomMutation baseline stores ([f20803f](https://github.com/teimurjan/lethe/commit/f20803f83598648db13e3799f0beeba34b3df421))
* Clustered RIF, +5.8% NDCG — cue-dependent suppression 5x better than global ([cbfe8c3](https://github.com/teimurjan/lethe/commit/cbfe8c3e32b099eda3589656c638f09e66f37db1))
* Enrichment layer validated on covered queries — +8.3pp NDCG over RIF ([eb4e551](https://github.com/teimurjan/lethe/commit/eb4e5513f6460aca8309b35ff94ff86ca908d86d))
* Exploration + rescue list with smart top-K injection ([a828402](https://github.com/teimurjan/lethe/commit/a828402744f63e0bad8a64f56f261586873bef6d))
* Implement experiment runner with circuit breakers ([4b6c4dd](https://github.com/teimurjan/lethe/commit/4b6c4dd289deb0832d8c803fde3c3df6b2e6f2cb))
* Implement GCMemoryStore with FAISS retrieval and GC loop ([9fc0544](https://github.com/teimurjan/lethe/commit/9fc05440033d1a8729d6704aa95fba0695d82442))
* Implement mutation and selection with unit tests ([a731564](https://github.com/teimurjan/lethe/commit/a731564f506adb761543fb0f922e65d1762f210f))
* Lifecycle benchmark confirms GC tiers don't help retrieval ([2bc0df3](https://github.com/teimurjan/lethe/commit/2bc0df3c5845ec2638daa3914f59055464ff9bab))
* LLM enrichment layer for checkpoint 13 (pipeline, not yet run) ([d9676ca](https://github.com/teimurjan/lethe/commit/d9676ca36193dc1fdfbd63e31ad1e1d4fe09d35a))
* MLP with per-step delta cap shows first positive result ([fbd70fb](https://github.com/teimurjan/lethe/commit/fbd70fb9089e91eddc682bcbf222afccaa3208ae))
* Production MemoryStore API with SQLite + FAISS + BM25 ([82f7076](https://github.com/teimurjan/lethe/commit/82f7076f23cc80e364f50def38340614a0ea90d3))
* Rank-gap competition formula, clustered+gap hits +6.5% NDCG / +9.5% recall ([784bd59](https://github.com/teimurjan/lethe/commit/784bd597c81cbb936ad2e9c0b46c751d55bacca3))
* Rescue index gives +12.9% hot NDCG on LongMemEval ([231e06c](https://github.com/teimurjan/lethe/commit/231e06c9eed988d6807452133450e79a82306941))
* Retrieval-induced forgetting, first positive learned mechanism (+2.0% NDCG) ([3bb484f](https://github.com/teimurjan/lethe/commit/3bb484f8240bd8f356e79009c4e7385eb20af776))
* Scaffold repo with pyproject.toml, config, and README ([f278435](https://github.com/teimurjan/lethe/commit/f27843593e8a6bcbd322da74a3ad1dd6fcf4aef2))
* Ship lethe as a Claude Code plugin ([87f1a40](https://github.com/teimurjan/lethe/commit/87f1a403e10d2549d1380e9cef83c138a9eeb4bf))
* Sparse Distributed Memory (SDM) research prototype ([7056b6c](https://github.com/teimurjan/lethe/commit/7056b6c2cb050fd590803e3f766b97bfc69c403a))
* Stabilize MLP with lower promotion thresholds ([2798a4a](https://github.com/teimurjan/lethe/commit/2798a4afa044e98e566eac1d2bab41af18fca3ca))
* Switch benchmark from NFCorpus/BEIR to LongMemEval ([b4a7194](https://github.com/teimurjan/lethe/commit/b4a7194e6cb924e6e6e82863097d7f5462fedd43))
* V3 cross-encoder guided adapter mutation ([019bb1f](https://github.com/teimurjan/lethe/commit/019bb1f55ea1c6c62c0c8692b014c591d015c187))
* V4 learned MLP adapter + segmentation mutation ([ef9a9c6](https://github.com/teimurjan/lethe/commit/ef9a9c649d8cf3a738ab06c1c093542adf34719f))
* V4 results on both datasets + MLP selection fix ([aa4f333](https://github.com/teimurjan/lethe/commit/aa4f33377c60b526ad90de74c915821ec80e99ac))
* Wire clustered RIF into production, add video demo ([d2b728b](https://github.com/teimurjan/lethe/commit/d2b728b6b90bbf27326e2e705cca0c14ce49d687))


### Bug Fixes

* Average NDCG@10 over all eval queries, not single sample ([413943e](https://github.com/teimurjan/lethe/commit/413943e7b9a567a765edc6729c8626d4ad35012d))
* Enrichment rate-limiting — default concurrency 20→5, max_retries 4→8 ([2c04558](https://github.com/teimurjan/lethe/commit/2c045582ffe17c97d567d74fc12008d9353bfc00))


### Documentation

* Checkpoint 14 negative result — retrieval-only ceiling reached ([ca6fb06](https://github.com/teimurjan/lethe/commit/ca6fb063d8b4fdcc0d6385c84bc478b9a77510a4))
* Checkpoint 15 — SDM prototype, negative result ([d657de4](https://github.com/teimurjan/lethe/commit/d657de4ee8a93e7c4be55698d2698171c207b7f8))
* Checkpoints 16-17 + benchmark-methodology disclaimer ([082b0c6](https://github.com/teimurjan/lethe/commit/082b0c6380f872ac0e2a60399f58b817a41f96db))
* Restructure — move result snapshots into benchmarks/results, rewrite BENCHMARKS and RESEARCH_JOURNEY ([1373264](https://github.com/teimurjan/lethe/commit/13732649ae27a8f42673d9c8573665f1002c3aa2))
* Rewrite README for checkpoints 13+17 and methodology note ([7637ce2](https://github.com/teimurjan/lethe/commit/7637ce2a00c59d6fd1cbd883c154976877ea46bc))
