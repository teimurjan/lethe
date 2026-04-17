# Changelog

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
