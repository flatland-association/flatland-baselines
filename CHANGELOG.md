# Changelog

## [4.3.0](https://github.com/flatland-association/flatland-baselines/compare/v4.2.5...v4.3.0) (2026-08-13)


### Features

* add debug output which agent blocks which other agent. ([e9fd34f](https://github.com/flatland-association/flatland-baselines/commit/e9fd34fc22df246aa781f720d5d5e6d301a02456))
* add dla-intermediate Docker image. ([f65add4](https://github.com/flatland-association/flatland-baselines/commit/f65add494086c173005ff087e6ab2579be0daf0a))
* add dla-intermediate Docker image. ([6cb34d5](https://github.com/flatland-association/flatland-baselines/commit/6cb34d5a2b2869d2e521c38be3f24cee5b0a6ec9))
* add do nothing heuristic. ([23e143d](https://github.com/flatland-association/flatland-baselines/commit/23e143d4e7a5aa3c56199ed4ca31e1fb40938c3b))
* add forever heuristic. ([b35a50d](https://github.com/flatland-association/flatland-baselines/commit/b35a50d5a607cc0592acac2b1db80d2cecccb549))
* add forward only heuristic. ([4cf055d](https://github.com/flatland-association/flatland-baselines/commit/4cf055d3970223274706537c7399cf742dc3afe3))
* add option use_alternative_at_first_intermediate_and_then_always_first_strategy. ([5ac308c](https://github.com/flatland-association/flatland-baselines/commit/5ac308c0915a4a13fcd377d4301dd28106d1493e))
* **baselines docker:** add ml dependencies docker image. ([#33](https://github.com/flatland-association/flatland-baselines/issues/33)) ([e863a35](https://github.com/flatland-association/flatland-baselines/commit/e863a351ddfd27c76c37e09676d43dc46a10ba88))
* **dla:** add audit logging. ([86dbf58](https://github.com/flatland-association/flatland-baselines/commit/86dbf581cf7c4e01f29d4ea7a88786a17363148e))
* Entrypoint refactoring ([7a0b407](https://github.com/flatland-association/flatland-baselines/commit/7a0b4077e43850ee52162991606a5ee75ff8677e))
* implement alternative-at-first-intermediate-and-then-always-first strategy. ([d08d67b](https://github.com/flatland-association/flatland-baselines/commit/d08d67b25680dd474a9fa97d9b2363e1e1130094))
* intermediate stops DLA. ([022bc34](https://github.com/flatland-association/flatland-baselines/commit/022bc345b78407a53681d49f11231ec9f58f3be8))
* introduce seeding of random generator to enable reproducibility. ([03e7145](https://github.com/flatland-association/flatland-baselines/commit/03e7145b6d837367467d55b2e7a59bf867993e5e))
* online and offline evaluation regression. ([#42](https://github.com/flatland-association/flatland-baselines/issues/42)) ([bc2e45f](https://github.com/flatland-association/flatland-baselines/commit/bc2e45fa2afa4621335912b218f5a33b994aa7ef))
* prevent agents from blocking each other upon entering at the same time. ([e235685](https://github.com/flatland-association/flatland-baselines/commit/e235685aafe4c82ab62d3498df23096c8a2be8e8))
* sanity check heuristics. ([#55](https://github.com/flatland-association/flatland-baselines/issues/55)) ([681d7bf](https://github.com/flatland-association/flatland-baselines/commit/681d7bf4d1e74f0ae5ae43ef3950c4cd8819b466))
* **trajectory API:** allow combination --seed with --env. ([cf19f3b](https://github.com/flatland-association/flatland-baselines/commit/cf19f3baf32bd0b03d362825b460e2c2c54b83e2))
* **trajectory API:** allow combination --seed with --env. ([8fbec2d](https://github.com/flatland-association/flatland-baselines/commit/8fbec2d41c5ab734fb33335afc13791ca965ce1c))
* try all options in use_alternative_at_first_intermediate_and_then_always_first_strategy. ([83c09e2](https://github.com/flatland-association/flatland-baselines/commit/83c09e262f0328506f23c9070b6e9be862327684))
* use intermediate targets, always taking first flexibility option. ([5082b41](https://github.com/flatland-association/flatland-baselines/commit/5082b412e11c986f37d2d4b034e9009602e13166))


### Bug Fixes

* add guard against invalid initial position. ([bc07805](https://github.com/flatland-association/flatland-baselines/commit/bc07805984bc64897a179b1bd2f3fb1e1b39887c))
* add guard against invalid initial position. ([3e598cd](https://github.com/flatland-association/flatland-baselines/commit/3e598cdfb6aa59d9356026d905be7a9314337376))
* comply with changes from upstream. ([7a5cc88](https://github.com/flatland-association/flatland-baselines/commit/7a5cc88a1a8ab50efc20f5bc140588658ddbcae0))
* **dla:** fix regression introduced in previous commit. ([34b1e98](https://github.com/flatland-association/flatland-baselines/commit/34b1e988f7a29d9a06b25623ea542b52d1a32342))
* **dla:** workaround for skip loopy lines. ([4fe7c28](https://github.com/flatland-association/flatland-baselines/commit/4fe7c2895b1b2af9ce86ae56e04ce0f62de4dff8))
* regresion upstream mean_normalized_reward. ([#43](https://github.com/flatland-association/flatland-baselines/issues/43)) ([9ad1f20](https://github.com/flatland-association/flatland-baselines/commit/9ad1f203542e541aae4385cb13b1e13ff0f5b06d))
* update full shortest distance agent map. ([b7682a4](https://github.com/flatland-association/flatland-baselines/commit/b7682a447808565b21d26dd20ca2d1cc8ccdccdb))


### Performance Improvements

* **dla:** avoid backwards from opposing_agent in trivial cases. ([19d16af](https://github.com/flatland-association/flatland-baselines/commit/19d16affcac3e0dbf51bf3bf16bd0aba9f670f6b))
* **dla:** avoid recomputation of bitmaps. ([7deb766](https://github.com/flatland-association/flatland-baselines/commit/7deb76614efee4827034efa5a22e7c23b1b6a041))
* **dla:** avoid recomputation when waiting. ([f993204](https://github.com/flatland-association/flatland-baselines/commit/f9932048c93e565c69902be3511dd0bc74557077))
* **dla:** avoid recomputation when waiting. ([2ac7406](https://github.com/flatland-association/flatland-baselines/commit/2ac740674354a3dd55a1884a87bfc47fb9b1feb1))
* **dla:** avoid summing full bitmap. ([811b1c3](https://github.com/flatland-association/flatland-baselines/commit/811b1c322e969512d52f98e90edbd98dbd6fe987))
* **dla:** avoid unneccessary re-allocation/gc. ([e07d9aa](https://github.com/flatland-association/flatland-baselines/commit/e07d9aaf2259d0dffda81d16c91a4b0e630c4c4a))
* **dla:** compute shortest paths only once by inheriting from shortest path policy. ([2201246](https://github.com/flatland-association/flatland-baselines/commit/220124644106622b89191ae246c500dd7a9035dc))
* **dla:** drop agent_positions_map in _check_agent_can_move as already covered by my_shortest_walking_path. ([b47832b](https://github.com/flatland-association/flatland-baselines/commit/b47832baac20c9fac815929524ea77ba45fa896c))
* **DLA:** performance improvements for deadlock avoidance heuristics. ([c54167b](https://github.com/flatland-association/flatland-baselines/commit/c54167bb3639a3cad186716da4a28376ed176829))
* **dla:** re-build shortest_distance_agent_map only if new positions. ([13e1ae4](https://github.com/flatland-association/flatland-baselines/commit/13e1ae4c6154b701e53e3146c5a27b34bd3409f8))
* **dla:** re-build shortest_distance_agent_map only if new positions. ([585e546](https://github.com/flatland-association/flatland-baselines/commit/585e546f06ed32624a8165433d655f8f0c98d8a7))
* **dla:** re-use agent_positions_map. ([6476855](https://github.com/flatland-association/flatland-baselines/commit/64768553a6b17c5daa8755be8148dcf9cf78fa57))
* **dla:** refactor into sub-methods to fine-grain profiling. ([27b25cd](https://github.com/flatland-association/flatland-baselines/commit/27b25cd1404f220c88ad08691ecbdb810821d24a))
* **dla:** refactor into sub-methods to fine-grain profiling. Do not re-compute shortest_distance_agent_map if no position-update. ([fe659e7](https://github.com/flatland-association/flatland-baselines/commit/fe659e79eefabd84606f4eeb2a83de0bcfc14a3e))
* **dla:** remove unused method. ([934d8ba](https://github.com/flatland-association/flatland-baselines/commit/934d8bac8caa2c812491050b5e6674603bb1912e))
* **dla:** remove unused same agent map. ([7006444](https://github.com/flatland-association/flatland-baselines/commit/70064449ee64645aec763f405506b8163e71abfe))
* **dla:** use position overlap to determine oncoming agents. ([6f73e8e](https://github.com/flatland-association/flatland-baselines/commit/6f73e8ec0832e4bc1375ae5a5aa105f43c128b7d))


### Miscellaneous Chores

* release 4.2.7 ([0fdcd54](https://github.com/flatland-association/flatland-baselines/commit/0fdcd54225e1ac1f8e923671be38d5123e93d68b))
* release 4.3.0 ([73231ab](https://github.com/flatland-association/flatland-baselines/commit/73231abd62c5d96ddd5be9a6fce21d110666a24b))

## [4.2.5] - Initial release tracked by release-please
