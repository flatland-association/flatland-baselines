## TL;DR;

```shell
docker compose  -f flatland_baselines/deadlock_avoidance_heuristic/online-demo/docker-compose.yml up --force-recreate --build


evaluator-1   | ====================================================================================================
evaluator-1   | ## Server Performance Stats
submission-1  | \ end random_agent
evaluator-1   | ====================================================================================================
evaluator-1   |          - message_queue_latency         => min: 0.0001671314239501953 || mean: 0.0019416656371024736 || max: 0.18240714073181152
evaluator-1   |          - current_episode_controller_inference_time     => min: 2.86102294921875e-05 || mean: 0.0011958306734971245 || max: 0.4178955554962158
evaluator-1   |          - controller_inference_time     => min: 2.86102294921875e-05 || mean: 0.0011958306734971245 || max: 0.4178955554962158
evaluator-1   |          - internal_env_step_time        => min: 7.700920104980469e-05 || mean: 0.0006845473827464444 || max: 0.13511109352111816
evaluator-1   | ====================================================================================================
evaluator-1   | ####################################################################################################
evaluator-1   | EVALUATION COMPLETE !!
evaluator-1   | ####################################################################################################
evaluator-1   | # Mean Reward : -3541.52
evaluator-1   | # Sum Normalized Reward : 43.08898598301832 (primary score)
evaluator-1   | # Mean Percentage Complete : 0.678 (secondary score)
evaluator-1   | # Mean Normalized Reward : 0.86178
evaluator-1   | ####################################################################################################
evaluator-1   | ####################################################################################################
evaluator-1   | \ end grader
evaluator-1   | \ end evaluator/run.sh
evaluator-1   | + echo '\ end evaluator/run.sh'
evaluator-1 exited with code 0
submission-1  | \ end submission_template/run.sh
submission-1  | + echo '\ end submission_template/run.sh'
submission-1 exited with code 0
redis-1       | 1:M 20 Jan 2026 22:09:07.928 * User requested shutdown...
redis-1       | 1:M 20 Jan 2026 22:09:07.930 * Saving the final RDB snapshot before exiting.
redis-1       | 1:M 20 Jan 2026 22:09:07.930 * BGSAVE done, 0 keys saved, 0 keys skipped, 88 bytes written.
redis-1       | 1:M 20 Jan 2026 22:09:07.934 * DB saved on disk
redis-1       | 1:M 20 Jan 2026 22:09:07.934 # Redis is now ready to exit, bye bye...
redis-1 exited with code 0
shutdown-redis-1 exited with code 0

```