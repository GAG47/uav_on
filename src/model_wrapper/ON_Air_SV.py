from model_wrapper.ON_Air_2 import ONAir


class ONAirSV(ONAir):
    """
    Semantic-Value Navigation wrapper.

    Step 1 only creates an independent method entry without changing the
    original UAV-ON baseline behavior.

    Later steps will replace the direct LLM-action pipeline with:
        Task1 semantic value reasoning
        semantic map update
        GDINO candidate detection
        Task2 target verification
        navigator decision
        StopGate
    """

    def __init__(self, fixed, batch_size):
        super().__init__(fixed=fixed, batch_size=batch_size)
        self.method_name = "SVNav"

    def prepare_inputs(self, episodes, fixed):
        return super().prepare_inputs(episodes, fixed)

    def run(self, inputs, fixed, prompt_info_list=None):
        return super().run(inputs, fixed, prompt_info_list)
