Evaluation calibration
======================

With `flatland-rl==4.1.2`:

```log

evaluator-1   | ====================================================================================================
evaluator-1   | ## Server Performance Stats
evaluator-1   | ====================================================================================================
evaluator-1   |          - message_queue_latency         => min: 0.00011229515075683594 || mean: 0.00779857359967132 || max: 1.302614450454712
evaluator-1   |          - current_episode_controller_inference_time     => min: 2.765655517578125e-05 || mean: 0.002113061910608245 || max: 0.15073108673095703
evaluator-1   |          - controller_inference_time     => min: 2.765655517578125e-05 || mean: 0.002113061910608245 || max: 0.15073108673095703
evaluator-1   |          - internal_env_step_time        => min: 4.410743713378906e-05 || mean: 0.0009631906605960431 || max: 0.5997908115386963
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
```

With `flatland-rl==4.2.1`:

one commit before new get_k_shortest_path (https://github.com/flatland-association/flatland-baselines/pull/41)
8f8bf7c6a50f4fc71634c5e1755ec26c6136c55c 