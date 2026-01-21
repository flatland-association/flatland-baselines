from flatland.evaluators.service import FlatlandRemoteEvaluationService

if __name__ == '__main__':
    print("/ start grader", flush=True)
    grader = FlatlandRemoteEvaluationService(
        test_env_folder="/tmp/debug-environments",
        # verbose=True,
        # temporarily use pickle because of fastenum failing with msgpack: https://github.com/flatland-association/flatland-rl/pull/214/files
        use_pickle=True,
        analysis_data_dir="/tmp/analysis_data_dir",
        action_dir="/tmp/actions",
        # does not work with subdir env paths:
        #episode_dir="/tmp/episodes",
        result_output_path="/tmp/results/results.csv",
        visualize=True,
        video_generation_envs=[
            # writes to the same folder, so take only one till fixed:
            # TODO fix in flatland-rl
            'Test_00/Level_0.pkl',
            # 'Test_00/Level_1.pkl', 'Test_00/Level_2.pkl', 'Test_00/Level_3.pkl', 'Test_00/Level_4.pkl',
            # 'Test_00/Level_5.pkl', 'Test_00/Level_6.pkl', 'Test_00/Level_7.pkl', 'Test_00/Level_8.pkl', 'Test_00/Level_9.pkl',
            # 'Test_01/Level_0.pkl', 'Test_01/Level_1.pkl', 'Test_01/Level_2.pkl', 'Test_01/Level_3.pkl', 'Test_01/Level_4.pkl',
            # 'Test_01/Level_5.pkl', 'Test_01/Level_6.pkl', 'Test_01/Level_7.pkl', 'Test_01/Level_8.pkl', 'Test_01/Level_9.pkl',
            # 'Test_02/Level_0.pkl', 'Test_02/Level_1.pkl', 'Test_02/Level_2.pkl', 'Test_02/Level_3.pkl', 'Test_02/Level_4.pkl',
            # 'Test_02/Level_5.pkl', 'Test_02/Level_6.pkl', 'Test_02/Level_7.pkl', 'Test_02/Level_8.pkl', 'Test_02/Level_9.pkl',
            # 'Test_03/Level_0.pkl', 'Test_03/Level_1.pkl', 'Test_03/Level_2.pkl', 'Test_03/Level_3.pkl', 'Test_03/Level_4.pkl',
            # 'Test_03/Level_5.pkl', 'Test_03/Level_6.pkl', 'Test_03/Level_7.pkl', 'Test_03/Level_8.pkl', 'Test_03/Level_9.pkl',
            # 'Test_04/Level_0.pkl', 'Test_04/Level_1.pkl', 'Test_04/Level_2.pkl', 'Test_04/Level_3.pkl', 'Test_04/Level_4.pkl',
            # 'Test_04/Level_5.pkl', 'Test_04/Level_6.pkl', 'Test_04/Level_7.pkl', 'Test_04/Level_8.pkl', 'Test_04/Level_9.pkl',
            # 'Test_05/Level_0.pkl', 'Test_05/Level_1.pkl', 'Test_05/Level_2.pkl', 'Test_05/Level_3.pkl', 'Test_05/Level_4.pkl',
            # 'Test_05/Level_5.pkl', 'Test_05/Level_6.pkl', 'Test_05/Level_7.pkl', 'Test_05/Level_8.pkl', 'Test_05/Level_9.pkl',
            # 'Test_06/Level_0.pkl', 'Test_06/Level_1.pkl', 'Test_06/Level_2.pkl', 'Test_06/Level_3.pkl', 'Test_06/Level_4.pkl',
            # 'Test_06/Level_5.pkl', 'Test_06/Level_6.pkl', 'Test_06/Level_7.pkl', 'Test_06/Level_8.pkl', 'Test_06/Level_9.pkl',
            # 'Test_07/Level_0.pkl', 'Test_07/Level_1.pkl', 'Test_07/Level_2.pkl', 'Test_07/Level_3.pkl', 'Test_07/Level_4.pkl',
            # 'Test_07/Level_5.pkl', 'Test_07/Level_6.pkl', 'Test_07/Level_7.pkl', 'Test_07/Level_8.pkl', 'Test_07/Level_9.pkl',
            # 'Test_08/Level_0.pkl', 'Test_08/Level_1.pkl', 'Test_08/Level_2.pkl', 'Test_08/Level_3.pkl', 'Test_08/Level_4.pkl',
            # 'Test_08/Level_5.pkl', 'Test_08/Level_6.pkl', 'Test_08/Level_7.pkl', 'Test_08/Level_8.pkl', 'Test_08/Level_9.pkl',
            # 'Test_09/Level_0.pkl', 'Test_09/Level_1.pkl', 'Test_09/Level_2.pkl', 'Test_09/Level_3.pkl', 'Test_09/Level_4.pkl',
            # 'Test_09/Level_5.pkl', 'Test_09/Level_6.pkl', 'Test_09/Level_7.pkl', 'Test_09/Level_8.pkl', 'Test_09/Level_9.pkl',
            # 'Test_10/Level_0.pkl', 'Test_10/Level_1.pkl', 'Test_10/Level_2.pkl', 'Test_10/Level_3.pkl', 'Test_10/Level_4.pkl',
            # 'Test_10/Level_5.pkl', 'Test_10/Level_6.pkl', 'Test_10/Level_7.pkl', 'Test_10/Level_8.pkl', 'Test_10/Level_9.pkl',
            # 'Test_11/Level_0.pkl', 'Test_11/Level_1.pkl', 'Test_11/Level_2.pkl', 'Test_11/Level_3.pkl', 'Test_11/Level_4.pkl',
            # 'Test_11/Level_5.pkl', 'Test_11/Level_6.pkl', 'Test_11/Level_7.pkl', 'Test_11/Level_8.pkl', 'Test_11/Level_9.pkl',
            # 'Test_12/Level_0.pkl', 'Test_12/Level_1.pkl', 'Test_12/Level_2.pkl', 'Test_12/Level_3.pkl', 'Test_12/Level_4.pkl',
            # 'Test_12/Level_5.pkl', 'Test_12/Level_6.pkl', 'Test_12/Level_7.pkl', 'Test_12/Level_8.pkl', 'Test_12/Level_9.pkl',
            # 'Test_13/Level_0.pkl', 'Test_13/Level_1.pkl', 'Test_13/Level_2.pkl', 'Test_13/Level_3.pkl', 'Test_13/Level_4.pkl',
            # 'Test_13/Level_5.pkl', 'Test_13/Level_6.pkl', 'Test_13/Level_7.pkl', 'Test_13/Level_8.pkl', 'Test_13/Level_9.pkl',
            # 'Test_14/Level_0.pkl', 'Test_14/Level_1.pkl', 'Test_14/Level_2.pkl', 'Test_14/Level_3.pkl', 'Test_14/Level_4.pkl',
            # 'Test_14/Level_5.pkl', 'Test_14/Level_6.pkl', 'Test_14/Level_7.pkl', 'Test_14/Level_8.pkl', 'Test_14/Level_9.pkl'
        ]
    )
    # TODO fix in flatland-rl
    grader.visualize = True
    grader.vizualization_folder_name = "/tmp/visualizations"
    grader.run()
    print("\\ end grader", flush=True)
