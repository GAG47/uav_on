import torch
import tqdm
import os
from pathlib import Path
import sys
import numpy as np
import random
import time

sys.path.append(str(Path(str(os.getcwd())).resolve()))
from common.param import args
from env_uav import AirVLNENV
from src.closeloop_util import BatchIterator, EvalBatchState, initialize_env_eval
from utils.logger import logger
from model_wrapper.base_model import BaseModelWrapper
from model_wrapper.ON_Air_2 import ONAir


class TeeStream(object):
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)


def verbose_eval_enabled():
    return os.environ.get("AIRHUNT_VERBOSE_EVAL", "0") == "1"


def setup_local_log_file(save_eval_path):
    if os.environ.get("AIRHUNT_DISABLE_LOG_FILE", "0") == "1":
        return None

    log_dir = os.environ.get(
        "AIRHUNT_LOG_DIR",
        os.path.join(save_eval_path, "airhunt_stop_logs")
    )
    os.makedirs(log_dir, exist_ok=True)

    log_name = time.strftime("stop_trace_%Y%m%d_%H%M%S.log")
    log_path = os.path.join(log_dir, log_name)
    log_file = open(log_path, "a", encoding="utf-8", buffering=1)

    old_stdout = sys.stdout
    old_stderr = sys.stderr

    sys.stdout = TeeStream(old_stdout, log_file)
    sys.stderr = TeeStream(old_stderr, log_file)

    print(f"[LogFile] saving console output to: {log_path}")

    return {
        "path": log_path,
        "file": log_file,
        "stdout": old_stdout,
        "stderr": old_stderr
    }


def close_local_log_file(log_info):
    if log_info is None:
        return

    sys.stdout = log_info["stdout"]
    sys.stderr = log_info["stderr"]

    log_file = log_info.get("file", None)
    if log_file is not None:
        log_file.close()


def print_decision_separator(step, completed, total):
    print("")
    print("=" * 100)
    print(f"[DecisionStep] step={step}, completed={completed}/{total}")
    print("=" * 100)


def print_action_summary(index, action, step_size, done):
    if action == "stop":
        print(
            "[StopAction] "
            f"Episode {index}: "
            f"action={action}, "
            f"step_size={step_size}, "
            f"done={done}"
        )
        return

    if done:
        print(
            "[EpisodeDone] "
            f"Episode {index}: "
            f"action={action}, "
            f"step_size={step_size}, "
            f"done={done}"
        )
        return

    if verbose_eval_enabled():
        print(
            "[EvalAction] "
            f"Episode {index}: "
            f"action={action}, "
            f"step_size={step_size}, "
            f"done={done}"
        )


def eval(modelWrapper: BaseModelWrapper, env: AirVLNENV, is_fixed, save_eval_path):
    with torch.no_grad():
        data = BatchIterator(env)
        data_len = len(data)
        pbar = tqdm.tqdm(total=data_len, desc="batch")
        cnt = 0

        while True:
            env_batch = env.next_minibatch(skip_scenes=[])
            if env_batch is None:
                break

            batch_state = EvalBatchState(
                batch_size=env.batch_size,
                env_batchs=env_batch,
                env=env,
                save_eval_path=save_eval_path
            )

            pbar.update(n=env.batch_size)
            cnt += env.batch_size

            for t in range(args.maxActions):
                completed = cnt - batch_state.skips.count(False)
                print_decision_separator(t, completed, data_len)

                logger.info('Step: {} \t Completed: {} / {}'.format(
                    t,
                    completed,
                    data_len
                ))

                inputs, user_prompts = modelWrapper.prepare_inputs(
                    batch_state.episodes,
                    is_fixed
                )

                if verbose_eval_enabled():
                    start1 = time.time()

                actions, steps_size, dones = modelWrapper.run(inputs, is_fixed)

                if verbose_eval_enabled():
                    print("get actions time:", time.time() - start1)

                for i in range(env.batch_size):
                    if dones[i]:
                        batch_state.dones[i] = True

                for i in range(len(actions)):
                    print_action_summary(
                        index=i,
                        action=actions[i],
                        step_size=steps_size[i],
                        done=dones[i]
                    )

                planned_paths = getattr(modelWrapper, "planned_paths", None)

                env.makeActions(
                    action_list=actions,
                    steps_size=steps_size,
                    is_fixed=is_fixed,
                    planned_paths=planned_paths
                )

                obs = env.get_obs()

                batch_state.update_from_env_output(
                    obs,
                    user_prompts,
                    actions,
                    steps_size,
                    is_fixed
                )

                batch_state.update_metric()
                is_terminate = batch_state.check_batch_termination(t)

                if is_terminate:
                    break

        try:
            pbar.close()
        except:
            pass


if __name__ == "__main__":
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    env = initialize_env_eval(
        dataset_path=args.dataset_path,
        save_path=args.eval_save_path
    )

    fixed = args.is_fixed
    save_eval_path = os.path.join(args.eval_save_path, args.name)

    if not os.path.exists(args.eval_save_path):
        os.makedirs(args.eval_save_path)

    log_info = setup_local_log_file(save_eval_path)

    try:
        modelWrapper = ONAir(fixed=fixed, batch_size=args.batchSize)

        eval(
            modelWrapper=modelWrapper,
            env=env,
            is_fixed=fixed,
            save_eval_path=save_eval_path
        )
    finally:
        env.delete_VectorEnvUtil()
        close_local_log_file(log_info)