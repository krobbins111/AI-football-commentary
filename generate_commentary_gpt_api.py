import json
import os
from pathlib import Path   

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
from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_audioclips, AudioClip

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
full_steps = 3000 * 9
current_10_min = 9
api_response = ''

def get_chat_completion(prompt):
    client = OpenAI(api_key='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')
    print('start inner thread')
    completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are an enthusiastic soccer commentator. All your remarks should be about 2 sentences"},
        {"role": "user", "content": f'{prompt}'}
    ],
    timeout=10)
    print('end')
    global api_response
    api_response = completion.choices[0].message.content

def threaded_inference(prompt, steps_time): # RIGHT PLACE WRONG ORDER
    print(prompt, ' at ', steps_time)
    api_thread = threading.Thread(target=get_chat_completion, args=(prompt,))

    # Start the thread
    api_thread.start()

    # Do other work in the main thread if needed

    # Wait for the API thread to complete
    api_thread.join()
    print('end inner thread')
    global api_response
    print("=" * 80)
    print('prompt: ' + prompt)
    print('model o/p: ' + api_response)
    # print('model o/p: ' + str(text.strip().find('.')))
    print("=" * 80)
    global current_10_min
    all_commentary.append((api_response, steps_time))
    # all_commentary.append((text, (3000 - steps_left) // 100 * current_10_min))
    print(current_10_min)
    # if steps_left == 0:
    #     current_10_min -= 1
    

def get_most_recent_avi(directory_path):
    files = os.listdir(directory_path)
    avi_files = [file for file in files if file.lower().endswith('.avi')]

    if not avi_files:
        print("No AVI files found in the directory.")
        return None

    most_recent_avi = max(avi_files, key=lambda file: os.path.getmtime(os.path.join(directory_path, file)))
    return os.path.join(directory_path, most_recent_avi)

def get_audio_files(directory_path):
    audio_files = [file for file in os.listdir(directory_path) if file.lower().endswith('.mp3')]
    return audio_files

def add_audio_to_video(video_path, audio_files):
    video_clip = VideoFileClip(video_path)
    final_audio = None

    for audio_file in audio_files:
        # Extract the start time from the audio file name (assuming the file name is in seconds)
        start_time = int(os.path.splitext(audio_file)[0])

        # Load the audio clip
        audio_clip = AudioFileClip(audio_file)

        # Trim the audio clip to start at the specified time
        audio_clip = audio_clip.subclip(start_time)

        # If it's the first audio file, initialize the final_audio
        if final_audio is None:
            final_audio = audio_clip
        else:
            # Overlay subsequent audio clips
            final_audio = final_audio.overlay(audio_clip)

    # Set the final audio on the video
    video_clip = video_clip.set_audio(final_audio)

    # Write the final video with added audio
    output_path = os.path.splitext(video_path)[0] + "_with_audio.mp4"
    video_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")

def add_multiple_audio(video_path, audio_clips):
    # Load the video clip
    video_clip = VideoFileClip(video_path)

    # Create an empty audio array with the same duration as the video
    combined_audio = video_clip.audio.subclip(0, video_clip.duration)

    # Add each audio clip at its specified time
    for audio_path in audio_clips:
        audio_clip = AudioFileClip(audio_path)
        audio_clip = combined_audio.subclip(Path(audio_path).name.split('.')[0])
        combined_audio = combined_audio.set_duration(max(combined_audio.duration, audio_clip.duration))
        combined_audio = combined_audio.audio(audio_clip)

    # Set the audio of the video clip to the combined audio
    video_clip = video_clip.set_audio(combined_audio)

    # Write the result to a file
    output_path = os.path.splitext(video_path)[0] + "_with_audio.mp4"
    video_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")

make_frame = lambda t: np.array([
    0,
    0
]).T.copy(order="C")

def generate_silence(duration):
    return AudioClip(make_frame=make_frame, duration=duration - 0.001)

def concatenate_audio_from_paths(video_path, audio_paths, steps_time):
    video_clip = VideoFileClip(video_path)
    
    audio_clips_with_silence = [AudioFileClip(audio_paths[0])]
    audio_paths.pop(0)
    pre_silence_time = 0
    last_played_time = 0
    total_time = 0
    last_duration = AudioFileClip(audio_paths[0]).duration
    audio_paths_with_time = [(audio_path, os.path.splitext(audio_path)[0].split('.')[0]) for audio_path in audio_paths]
    audio_paths_with_time.sort(key=lambda a: int(a[1]))
    print(audio_paths_with_time)

    for audio_path, start_time in audio_paths_with_time:
        print(audio_path, start_time)
        # Load the original audio clip
        current_clip = AudioFileClip(audio_path)

        current_actual_time = ((video_clip.duration / steps_time) * (int(start_time) - int(pre_silence_time)))
        actual_pre_silence_time = (video_clip.duration / steps_time) * int(pre_silence_time)
        actual_start_time = (video_clip.duration / steps_time) * int(start_time)
        print('total time:', total_time, 'last played at:', last_played_time)
        if not last_played_time > actual_start_time:
            # Generate silence before the current clip
            last_played_time = actual_pre_silence_time + last_duration
            print(actual_start_time, last_played_time, actual_start_time - last_played_time)
            pre_silence = generate_silence(actual_start_time - last_played_time)
            # Concatenate the generated silence, original clip, and post silence
            audio_clip_with_silence = concatenate_audioclips([pre_silence, current_clip])
            last_duration = current_clip.duration
            # Append the result to the list
            audio_clips_with_silence.append(audio_clip_with_silence)
            total_time += audio_clip_with_silence.duration
            #INTERRUPTION TODO

        last_duration = current_clip.duration
        pre_silence_time = start_time

    # Concatenate all clips with silence
    concatenated_audio = concatenate_audioclips(audio_clips_with_silence)

    video_clip = video_clip.set_audio(concatenated_audio)
    print(concatenated_audio.duration)

    output_path = os.path.splitext(video_path)[0] + "_with_audio.mp4"
    video_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")

example_commentary = ["And we're off! The match has officially begun with a swift kick-off, setting the tone for an exciting game ahead.", "And it looks like he's lining up to take it, let's see if he can get it past the wall and into the back of the net!", 'The player has been fouled, and now they have a great opportunity to score from this free kick. Things are about to get interesting!', 'The referee is giving a free kick, an opportunity for the team to try and score from a set piece.', "What a missed opportunity for the attacking team! They couldn't convert and it's now a goal kick.", "And it's a corner kick! The attacking team will have a chance to create a scoring opportunity from this set-piece.", 'The referee has awarded a free kick! This is a great opportunity for the team to capitalize and create a scoring chance.', "Great pressure from the attacking team! They've earned themselves a corner kick to keep up the momentum.", "The attacking team couldn't convert their chances into a goal, and now it's a goal kick for the opposing team.", "The attacking team's efforts were in vain as the ball crossed the goal line, resulting in a goal kick for the defending team.", 'What a fantastic strike! The away team has taken the lead with a clinical finish.', "What a disappointment for the attacking team! After all that build-up play, it's just a goal kick for the opposition.", "The attacking team had a promising build-up but ultimately didn't find the mark. It's just a goal kick for the defending team.", "And the player lines up for the shot... it's a strong strike, but the goalkeeper dives and makes an incredible save!", "The goalkeeper wasn't even troubled by that shot, it was an easy save for him."]


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
        'render': False,
        'write_video': True,
    })
    if FLAGS.level:
        cfg['level'] = FLAGS.level

    env = football_env.FootballEnv(cfg)
    env.render(mode='human')
    commentary = Commentary()
    try:
        prompt = 'initializing'
        commentary_thread = threading.Thread(target=threaded_inference,
                                                args=(prompt, 0))

        while True:
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            print(env._env._steps_time, env._env._step_count)
            prompt, interrupt_current_commentary = commentary.process_observation(observation)
            # done = True
            # print(observation[0]['steps_left'])
            # if prompt:
            #     if commentary_thread.is_alive():
            #         if interrupt_current_commentary:
            #             # commentary_thread kill/stop
            #             # commentary_thread.start()
            #             print('thread alive')
            #     else:
            #         commentary_thread = threading.Thread(target=threaded_inference,
            #                                                 args=(prompt, env._env._steps_time))
            #         commentary_thread.start()
            for _ in range(3):
                    if prompt:
                        threading.Thread(target=threaded_inference, args=(prompt, env._env._steps_time)).start()
            if done:
                print('Done with game')
                print(all_commentary)
                time.sleep(10)
                tts_engine = pyttsx3.init()
                tts_engine.setProperty('voice', 'english')
                tts_engine.setProperty('rate', 200)
                for commentary, second in all_commentary:
                    tts_engine.save_to_file(commentary, f'{second}.mp3')
                    tts_engine.runAndWait()

                most_recent_avi = get_most_recent_avi('C:\\Users\\kevin\\AppData\\Local\\Temp\\dumps')

                if most_recent_avi:
                    print(f"The most recent AVI file is: {most_recent_avi}")

                    # Get audio files in the current directory
                    audio_files = get_audio_files('.')

                    if audio_files:
                        print(f"Found audio files: {audio_files}")

                        # Add audio to the video
                        concatenate_audio_from_paths(most_recent_avi, audio_files, env._env._steps_time)
                    else:
                        print("No audio files found.")
                else:
                    print("No AVI files found.")
                # make audio
                # put audio in video
                exit(1)
    except KeyboardInterrupt:
        env.write_dump('shutdown')
        exit(1)


if __name__ == '__main__':
    app.run(main)
