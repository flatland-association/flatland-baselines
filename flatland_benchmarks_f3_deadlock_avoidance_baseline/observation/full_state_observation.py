from flatland.core.env_observation_builder import ObservationBuilder, AgentHandle, ObservationType


class FullStateObservationBuilder(ObservationBuilder):
    def get(self, handle: AgentHandle = 0) -> ObservationType:
        """
        Called whenever an observation has to be computed for the `env` environment, possibly
        for each agent independently (agent id `handle`).

        Parameters
        ----------
        handle : int, optional
            Handle of the agent for which to compute the observation vector.

        Returns
        -------
        function
            An observation structure, specific to the corresponding environment.
        """
        return self.env

    def reset(self):
        pass


if __name__ == '__main__':
    for i in range(5):
        for j in range(10):
            print(f'("malfunction_deadlock_avoidance_heuristics/Test_0{i}/Level_{j}", "Test_0{i}_Level_{j}"),')
