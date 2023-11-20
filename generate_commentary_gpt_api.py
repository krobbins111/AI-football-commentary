import json
import os

import encoder
import model
import numpy as np
import sample
import tensorflow as tf
import threading

from gfootball.env import football_env
from gfootball.env import config
from gfootball.env import football_action_set

from absl import app
from absl import flags
import pyttsx3
from commentary import Commentary
from openai import OpenAI
import time

FLAGS = flags.FLAGS
flags.DEFINE_string(
    'home_players', 'keyboard',
    'Comma separated list of home players, single keyboard player by default')
flags.DEFINE_string('env_name', '11_vs_11_easy_stochastic', 'game difficulty')
flags.DEFINE_string('away_players', '', 'List of away players')
flags.DEFINE_string('level', '', 'Level to play')
flags.DEFINE_enum('action_set', 'full', ['default', 'full'], 'Action set')
flags.DEFINE_bool('real_time', True,
                  'If true, environment will slow down so humans can play.')

all_commentary = []

def threaded_inference(client, prompt):
    client = OpenAI(api_key='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')
    completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are an enthusiastic soccer commentator. All your remarks should be about 2 sentences"},
        {"role": "user", "content": f'{prompt}'}
    ],
    timeout=10)
    text = completion.choices[0].message.content
    print("=" * 80)
    print('prompt: ' + prompt)
    print('model o/p: ' + text)
    # print('model o/p: ' + str(text.strip().find('.')))
    print("=" * 80)
    all_commentary.append(text)
    # tts_engine = pyttsx3.init()
    # tts_engine.setProperty('voice', 'english')
    # tts_engine.setProperty('rate', 100)
    # tts_engine.say(text)
    # tts_engine.runAndWait()


def main(_):
    client = OpenAI(api_key='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')

    cfg = config.Config({
        'env_name': FLAGS.env_name,
        'action_set': FLAGS.action_set,
        'away_players':
            FLAGS.away_players.split(',') if FLAGS.away_players else '',
        'dump_full_episodes': True,
        'home_players':
            FLAGS.home_players.split(',') if FLAGS.home_players else '',
        'real_time': FLAGS.real_time,
        'render': True
    })
    if FLAGS.level:
        cfg['level'] = FLAGS.level

    env = football_env.FootballEnv(cfg)
    env.render(mode='human')
    commentary = Commentary()
    try:
        prompt = 'initializing'
        commentary_thread = threading.Thread(target=threaded_inference,
                                                args=(client, prompt))

        while True:
            observation, reward, done, info = env.step([football_action_set.action_long_pass])
            prompt, interrupt_current_commentary = commentary.process_observation(observation)
            if prompt:
                if commentary_thread.is_alive():
                    if interrupt_current_commentary:
                        # commentary_thread kill/stop
                        # commentary_thread.start()
                        print('thread alive')
                else:
                    commentary_thread = threading.Thread(target=threaded_inference,
                                                            args=(client, prompt))
                    commentary_thread.start()
            if done:
                print('Done with game')
                print(all_commentary)
                # make audio
                # put audio in video
                exit(1)
    except KeyboardInterrupt:
        env.write_dump('shutdown')
        exit(1)


if __name__ == '__main__':
    app.run(main)
