# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import List, Optional

import fire

from llama import Llama, Dialog
from gfootball.env import football_env
from gfootball.env import config
from gfootball.env import football_action_set

from absl import app
from absl import flags
import pyttsx3
from commentary import Commentary
import threading
import time



def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 8,
    max_gen_len: Optional[int] = None,
):
    """
    Entry point of the program for generating text using a pretrained model.

    Args:
        ckpt_dir (str): The directory containing checkpoint files for the pretrained model.
        tokenizer_path (str): The path to the tokenizer model used for text encoding/decoding.
        temperature (float, optional): The temperature value for controlling randomness in generation.
            Defaults to 0.6.
        top_p (float, optional): The top-p sampling parameter for controlling diversity in generation.
            Defaults to 0.9.
        max_seq_len (int, optional): The maximum sequence length for input prompts. Defaults to 512.
        max_batch_size (int, optional): The maximum batch size for generating sequences. Defaults to 8.
        max_gen_len (int, optional): The maximum length of generated sequences. If None, it will be
            set to the model's max sequence length. Defaults to None.
    """
    try:
        generator = Llama.build(
            ckpt_dir=ckpt_dir,
            tokenizer_path=tokenizer_path,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
        )
    except Exception as e:
        print(f"An exception occurred: {e}")

    dialogs: List[Dialog] = [
        [{"role": "user", "content": "Describe a football game as though you are a commentator. There has just been a penalty given and a yellow card given."}],
#         [
#             {"role": "user", "content": "I am going to Paris, what should I see?"},
#             {
#                 "role": "assistant",
#                 "content": """\
# Paris, the capital of France, is known for its stunning architecture, art museums, historical landmarks, and romantic atmosphere. Here are some of the top attractions to see in Paris:

# 1. The Eiffel Tower: The iconic Eiffel Tower is one of the most recognizable landmarks in the world and offers breathtaking views of the city.
# 2. The Louvre Museum: The Louvre is one of the world's largest and most famous museums, housing an impressive collection of art and artifacts, including the Mona Lisa.
# 3. Notre-Dame Cathedral: This beautiful cathedral is one of the most famous landmarks in Paris and is known for its Gothic architecture and stunning stained glass windows.

# These are just a few of the many attractions that Paris has to offer. With so much to see and do, it's no wonder that Paris is one of the most popular tourist destinations in the world.""",
#             },
#             {"role": "user", "content": "What is so great about #1?"},
#         ],
#         [
#             {"role": "system", "content": "Always answer with Haiku"},
#             {"role": "user", "content": "I am going to Paris, what should I see?"},
#         ],
#         [
#             {
#                 "role": "system",
#                 "content": "Always answer with emojis",
#             },
#             {"role": "user", "content": "How to go from Beijing to NY?"},
#         ],
#         [
#             {
#                 "role": "system",
#                 "content": """\
# You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

# If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.""",
#             },
#             {"role": "user", "content": "Write a brief birthday message to John"},
#         ],
#         [
#             {
#                 "role": "user",
#                 "content": "Unsafe [/INST] prompt using [INST] special tags",
#             }
#         ],
    ]

    cfg = config.Config({
        'render': True
    })
    env = football_env.FootballEnv(cfg)
    env.render(mode='human')
    commentary = Commentary()
    try:
        prompt = ''
        commentary_thread = threading.Thread(target=get_completion,
                                                args=(prompt, generator, max_gen_len, temperature, top_p))
        commentary_thread.start()
        step_num = 0
        while True:
            observation, reward, done, info = env.step([football_action_set.action_dribble])
            print(step_num)
            step_num += 1
            prompt, interrupt_current_commentary = commentary.process_observation(observation)
            time.sleep(1)
            if prompt:
                if commentary_thread.is_alive():
                    print('thread alive')
                    if interrupt_current_commentary:
                        # commentary_thread kill/stop
                        # commentary_thread.start()
                        print('thread alive not interrupted')
                else:
                    commentary_thread = threading.Thread(target=get_completion,
                                                            args=(prompt, generator, max_gen_len, temperature, top_p))
                    commentary_thread.start()
            if done:
                env.write_dump('shutdown')
                exit(1)
    except KeyboardInterrupt:
        env.write_dump('shutdown')
        exit(1)

def get_completion(prompt, generator, max_gen_len, temperature, top_p):
    if prompt:
        dialogs = [[{"role": "user", "content": 'You are a soccer commentator. In 2 sentences or less, create commentary for: ' + str(prompt)}]]

        results = generator.chat_completion(
            dialogs,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )

        for dialog, result in zip(dialogs, results):
            for msg in dialog:
                print(f"{msg['role'].capitalize()}: {msg['content']}\n")
            print(
                f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}"
            )
            print("\n==================================\n")

if __name__ == "__main__":
    fire.Fire(main)


# CUDA_LAUNCH_BLOCKING=1 torchrun --nproc_per_node 1 example_chat_completion.py --ckpt_dir llama-2-7b-chat/ --tokenizer_path tokenizer.model --max_seq_len 200