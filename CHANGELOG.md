# Changelog

## [4.4.0](https://github.com/flatland-association/flatland-baselines/compare/v4.3.0...v4.4.0) (2026-11-26)


### Features

* **sipp-locks:** add the ECML 2026 winner as baseline on flatland-rl 4.2.6. ([c0efe89](https://github.com/flatland-association/flatland-baselines/commit/c0efe899635756c1585029737dfc2ad3939626de))
* **sipp-locks:** import ECML 2026 winner v4-sipp-locks unchanged from darshanmakwana412/ecml2026@efcdd89. ([80124cb](https://github.com/flatland-association/flatland-baselines/commit/80124cb1c4c531de4915145cc8e628698fcdce36))
* **sipp-locks:** run v4-sipp-locks as baseline on flatland-rl 4.2.6. ([2172dfb](https://github.com/flatland-association/flatland-baselines/commit/2172dfbc62622a38af8fa1eaaf86b16a70e7dad4))


### Bug Fixes

* also detect a first-step earliest_departure=0 WAITING agent as entering ([c5af0fb](https://github.com/flatland-association/flatland-baselines/commit/c5af0fb395990f7d73a6e67271349d51aff54519))
* bump flatland-rl pin to fix/policy-runner-skip-state-machine-update, request wrapped env for DLA ([c1a55fe](https://github.com/flatland-association/flatland-baselines/commit/c1a55fe5ff4ff04f2dd90ed10712afcc9d6867dc))
* bump flatland-rl pin to main, resolving the reload-without-no_save gap ([f75fcad](https://github.com/flatland-association/flatland-baselines/commit/f75fcad1ba7f41f92bb3c3c3c4093760491950da))
* comply with upstream D1*, consider agents ending an off-map malfunction in entering prevention (flatland-rl[#513](https://github.com/flatland-association/flatland-baselines/issues/513)). ([efbad31](https://github.com/flatland-association/flatland-baselines/commit/efbad31c42a2dfa82c64ac58236ccb789f6e2c70))
* comply with upstream D2.7, send the planned direction instead of STOP_MOVING on an unavoidable crossing (flatland-rl[#509](https://github.com/flatland-association/flatland-baselines/issues/509)). ([b494818](https://github.com/flatland-association/flatland-baselines/commit/b4948186dae40c3d547d03ef9528b791b76df7a1))
* comply with upstream D5/6, see an agent with earliest departure 0 as entering in the first step (flatland-rl[#515](https://github.com/flatland-association/flatland-baselines/issues/515)). ([ce15b5f](https://github.com/flatland-association/flatland-baselines/commit/ce15b5f93247423ea04cedcfe4d2d106c7bf2246))
* comply with upstream optional state machine, wrap the environments the tests build (flatland-rl[#520](https://github.com/flatland-association/flatland-baselines/issues/520)). ([509a60d](https://github.com/flatland-association/flatland-baselines/commit/509a60df4bbbb428ced38bee9810d51eec85b7dc))
* consider malfunction-off-map agents in entering-prevention pairwise check ([9aef117](https://github.com/flatland-association/flatland-baselines/commit/9aef11789fd0e15205a726967863c9a909cc460d))
* **dla:** mirror the engine's departure rules in entering prevention. ([2b0a6c9](https://github.com/flatland-association/flatland-baselines/commit/2b0a6c9fcd8cd7caa09caca189805a7ed9aa9be2))
* **dla:** send a directional action for an unavoidable crossing only where STOP_MOVING would leave the planned path. ([04793df](https://github.com/flatland-association/flatland-baselines/commit/04793df989aa82d6cfecb8f1104c601b5d9fb36f))
* honor no_save in create_from_policy reload, bump flatland-rl pin to edge-19 ([d169d2f](https://github.com/flatland-association/flatland-baselines/commit/d169d2f82dd47a5a104a2cd0082b082e73d4494e))
* only treat a malfunction-off-map agent as entering once its malfunction actually clears ([16ae0cf](https://github.com/flatland-association/flatland-baselines/commit/16ae0cf4bf6e1e84e9fa84017a916b0893806db8))
* pin flatland-rl to [#541](https://github.com/flatland-association/flatland-baselines/issues/541)'s head with the pandas 3 fix for tuple rewards. ([5401473](https://github.com/flatland-association/flatland-baselines/commit/54014731286841329059ee63dd54c02abe0a8567))
* pin flatland-rl to the merge commit of flatland-rl[#541](https://github.com/flatland-association/flatland-baselines/issues/541) on main. ([911fbc9](https://github.com/flatland-association/flatland-baselines/commit/911fbc9fe67500f257745f9d9aae2c028863524e))
* pin flatland-rl to the rebased head of fix/policy-runner-skip-state-machine-update, re-calibrate the offline demo for flatland-rl [#545](https://github.com/flatland-association/flatland-baselines/issues/545). ([2377240](https://github.com/flatland-association/flatland-baselines/commit/23772406e62b446129511da9ad06ac8b9b868398))
* point edge-19 pin at flatland-rl's actual merged PR [#522](https://github.com/flatland-association/flatland-baselines/issues/522) commit ([d1bafae](https://github.com/flatland-association/flatland-baselines/commit/d1bafae7fe49fd58fa68538248805cff90b8d0d6))
* **policies:** let SetPathPolicy act for an agent parked at its target. ([4970b3d](https://github.com/flatland-association/flatland-baselines/commit/4970b3d292c31b999de5fb917b50990efe7927fb))
* **policies:** try SetPathPolicy's actions in a fixed order, so the chosen action does not depend on the hash seed. ([2c74391](https://github.com/flatland-association/flatland-baselines/commit/2c7439151155605d9fe0547fb3ac44df1e70b69c))
* **policies:** use flatland-rl's _check_action_on_agent in SetPathPolicy._act and test it. ([e61c670](https://github.com/flatland-association/flatland-baselines/commit/e61c670043318fb96d00db6c09fb0e3d082312fe))
* re-calibrate DLA success-rate/reward golden values for flatland-rl@edge-9 ([3563fe9](https://github.com/flatland-association/flatland-baselines/commit/3563fe9e6145d213189d34e6639228f00bd07770))
* re-calibrate offline_demo environments_v2 golden values for flatland-rl@edge-17 ([c9b7956](https://github.com/flatland-association/flatland-baselines/commit/c9b7956aa2d98690df8cf063441818349116eb19))
* re-calibrate offline_demo golden values for flatland-rl@edge-9 ([ad7d1ce](https://github.com/flatland-association/flatland-baselines/commit/ad7d1cee7eed673b1fceef5f79fd73be72844443))
* track agent.state for DLA in online/offline demos, re-calibrate envs_v2 for flatland-rl 6788c0a7 ([3280cf7](https://github.com/flatland-association/flatland-baselines/commit/3280cf7ffa5b4cc84bec1bbd2e2047c4d44ba875))
* track flatland-rl's Configuration/ObservationType renames and the cell-entry action timing change ([0bc55e1](https://github.com/flatland-association/flatland-baselines/commit/0bc55e1d6b4d95ddac3a20e33b76d561c2559aef))
* track flatland-rl's ObservationBuilder TypeVar rename (Observation -&gt; ObservationT) ([5a4e7fb](https://github.com/flatland-association/flatland-baselines/commit/5a4e7fbe49e125f7f89069788cd467d8afb44720))
* wrap test RailEnv construction with RailEnvStateMachineWrapper, bump flatland-rl pin to edge-17 ([3bf802d](https://github.com/flatland-association/flatland-baselines/commit/3bf802d96ff196925dbb6d5158d3f0455e2ff87d))

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
