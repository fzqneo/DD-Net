import collections
import csv
import subprocess
import os


PATH_PREFIX = '/home/ubuntu/ActorConditionedAttentionMaps/data/AVA'
MP4_SUFFIX = 'movies/trainval/{}.mp4'
MKV_SUFFIX = 'movies/trainval/{}.mkv'
WEBM_SUFFIX = 'movies/trainval/{}.webm'
OUTPUT_PREFIX = '/home/ubuntu/avaclips'


def write_video(filename, output, start_timestamp, end_timestamp):
    video_path = os.path.join(PATH_PREFIX, MP4_SUFFIX.format(filename))
    if not os.path.exists(video_path):
        video_path = os.path.join(PATH_PREFIX, MKV_SUFFIX.format(filename))
        if not os.path.exists(video_path):
            video_path = os.path.join(PATH_PREFIX, WEBM_SUFFIX.format(filename))
            if not os.path.exists(video_path):
                raise Exception('Bad video', video_path)

    full_output = os.path.join(OUTPUT_PREFIX, output)
    subprocess.call(
        ['ffmpeg', '-i', video_path, '-ss', str(start_timestamp), '-to', str(end_timestamp),
         '-c:v', 'libx264', '-crf', '18', '-preset', 'veryfast', full_output])


def main():
    with open(os.path.join(PATH_PREFIX, 'annotations/ava_train_v2.2.csv')) as annotations:
        clap_times = collections.defaultdict(list)
        wave_times = collections.defaultdict(list)

        clap_count = 0
        wave_count = 0
        clap_wave_count = 0
        
        annreader = csv.reader(annotations)
        for row in annreader:
            filename = row[0]
            mid_timestamp = int(row[1])
            label = int(row[6])

            if label == 67:
                clap_times[filename].append(mid_timestamp)
            elif label == 69:
                wave_times[filename].append(mid_timestamp)

        for filename in clap_times:
            print(filename)
            clap_set = set(clap_times[filename])
            wave_set = set(wave_times[filename])

            clap_times[filename] = clap_set.copy()
            wave_times[filename] = wave_set.copy()
            
            while len(clap_times[filename]) > 0:
                min_timestamp = min(clap_times[filename])

                clap_times[filename].remove(min_timestamp)
                if (min_timestamp + 1) in clap_set:
                    clap_times[filename].remove(min_timestamp + 1)

                start_timestamp = min_timestamp - 2
                end_timestamp = min_timestamp + 2

                while end_timestamp in clap_set:
                    clap_times[filename].remove(end_timestamp)
                    
                    end_timestamp += 1

                contains_both = False
                for i in range(start_timestamp, end_timestamp + 1):
                    if filename != '_ithRWANKB0' and i != 1062 and i in wave_set:
                        wave_times[filename].remove(i)
                        contains_both = True

                if contains_both:
                    write_video(filename, 'clapwave{}.mp4'.format(clap_wave_count),
                                start_timestamp, end_timestamp)
                    clap_wave_count += 1
                else:
                    write_video(filename, 'clap{}.mp4'.format(clap_count),
                                start_timestamp, end_timestamp)
                    clap_count += 1

            while len(wave_times[filename]) > 0:
                min_timestamp = min(wave_times[filename])
                
                wave_times[filename].remove(min_timestamp)
                if (min_timestamp + 1) in wave_set:
                    wave_times[filename].remove(min_timestamp + 1)

                start_timestamp = min_timestamp - 2
                end_timestamp = min_timestamp + 2

                while end_timestamp in wave_set:
                    wave_times[filename].remove(end_timestamp)

                    end_timestamp += 1

                write_video(filename, 'wave{}.mp4'.format(wave_count),
                            start_timestamp, end_timestamp)
                wave_count += 1

    print('clap', clap_count)
    print('wave', wave_count)
    print('clap and wave', clap_wave_count)
        
if __name__ == '__main__':
    main()
