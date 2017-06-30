import os, pandas as pd, numpy as np, yaml, shutil, argparse, textwrap, logging

classnames = ['babycry','glassbreak','gunshot']

def bg_dcase_annotations_preprocess(bg_screening_filepath, classnames):
    screening_df = pd.read_csv(bg_screening_filepath, sep=',')
    screening_skips = [scr[0] for scr in screening_df.values if not np.all(pd.isnull(scr[1:4]))]
    classwise_screening_skips = {}
    for id, classname in enumerate(classnames):
        classwise_screening_skips[classname] = [scr[0] for scr in screening_df.values if not (pd.isnull(scr[id+1]))]
    return classwise_screening_skips

def get_list_of_files(path, ext):
    files = []
    for dirpath, d, f in os.walk(path):
        for file in f:
            if os.path.splitext(file)[-1].lower()[1:] == ext.lower():
                files.append(os.path.join(dirpath, file))
    return files

def read_meta_yaml(filename):
    with open(filename, 'r') as infile:
        data = yaml.load(infile)
    return data

def text_file_to_list(path):
    with open(path) as f:
        content = f.readlines()
    return content

def list_to_text_file(content, path):
    with open(path, 'w') as f:
        for line in content:
            f.write(line)

def remove_lines_containing_given_substrings_from_file(filepath, substrings):
    counter = 0
    content = text_file_to_list(filepath)
    filtered_contend = [line for line in content
                        if not any(affected_mixture_file in line for affected_mixture_file in substrings)]
    if not filtered_contend == content:
        shutil.move(filepath, filepath + '_old_dontuse')
        counter += 1
    list_to_text_file(filtered_contend, filepath)  # let's save just in case anyway: what if equality breaks for some encoding issue or smth
    return counter


def main(path_to_dataset, log='file'):

    if log=='stdout':
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    else:
        logging.basicConfig(filename='log_of_patching.txt', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

    bg_screening_filepath = os.path.join(path_to_dataset, 'data', 'source_data', 'bgs', 'bg_screening.csv')
    classwise_screening_skips = bg_dcase_annotations_preprocess(bg_screening_filepath, classnames=classnames)


    if os.path.exists(os.path.join(path_to_dataset, 'generated_data')):
        mode = 'baseline'
    elif os.path.exists(os.path.join(path_to_dataset, 'data','mixture_data','devtrain')):
        mode = 'standalone'
    else:
        raise IOError('Unexpected folder structure in given {} dataset path. It should be either '
                      '..applications/data/TUT-rare-sound-events-2017-development for baseline mode or root folder '
                      'TUT-rare-sound-events-2017-development in standalone mode. ')

    if mode=='baseline':
        generated_data_folder = os.path.join(path_to_dataset, 'generated_data')
        meta_files = get_list_of_files(generated_data_folder,'txt')
    else:
        generated_data_folder = os.path.join(path_to_dataset, 'data','mixture_data')
        meta_files = []

    yaml_files = get_list_of_files(generated_data_folder,'yaml')
    recipe_files = [filepath for filepath in yaml_files if filepath.find('mixture_recipe')>=0]

    affected_mixture_files = []
    counter=0
    for recipe_file in recipe_files:
        logging.debug('Filename: {}'.format(recipe_file))
        classname = [classname for classname in classnames if classname in recipe_file][0]
        recipe_data = read_meta_yaml(recipe_file)
        for recipe in recipe_data:
            bg_file = recipe['bg_path'].replace('audio/','') # in case it's there, but support also cases if not
            mixture_audio_filename = recipe['mixture_audio_filename']
            if bg_file in classwise_screening_skips[classname]:
                counter+=1
                logging.debug('{}. Mixture {} was found to be using the bg {}, which is known to naturally contain {} sounds. '
                      'Adding to the list.'.format(counter, mixture_audio_filename, bg_file, classname))
                affected_mixture_files.append(mixture_audio_filename)


    # remove affected files from the meta files, event list files (sort of idential things, but one was generated as extra by the baseline,
    # and we cannot guarantee which of them users might be using), as well as make audio files unusable
    if mode=='baseline':
        meta_file_change_counter = 0
        for meta_file in meta_files:
            meta_file_change_counter += remove_lines_containing_given_substrings_from_file(meta_file, affected_mixture_files)

    csv_files = get_list_of_files(generated_data_folder,'csv')
    eventlist_files = [filepath for filepath in csv_files if filepath.find('event_list')>=0]

    eventlist_file_change_counter = 0
    for eventlist_file in eventlist_files:
        eventlist_file_change_counter += remove_lines_containing_given_substrings_from_file(eventlist_file, affected_mixture_files)

    # make the audio files unusable too
    wav_files = get_list_of_files(generated_data_folder,'wav')
    wav_file_change_counter = 0
    for wav_file in wav_files:
        if any(affected_mixture_file in wav_file for affected_mixture_file in affected_mixture_files):
            logging.debug('Making affected file {} unusable'.format(wav_file))
            shutil.move(wav_file,wav_file+'_old_dontuse')
            wav_file_change_counter += 1

    logging.debug('='*10)
    if eventlist_file_change_counter + wav_file_change_counter > 0:
        logging.debug('Patching done. Following files were affected and thus patched successfully: ')
        if mode=='baseline':
            logging.debug('{} meta files with all-class event rolls (baseline style),'.format(meta_file_change_counter))
        logging.debug('{} per-class event list files,'.format(eventlist_file_change_counter))
        logging.debug('{} mixture wav files,'.format(wav_file_change_counter))
    else:
        logging.debug('For some reason there was nothing to patch. Was the dataset already patched? Contact us if you suspect a bug!')
        logging.debug('email: Aleksandr Diment ( firstname.lastname at tut.fi )')
    return (eventlist_file_change_counter, wav_file_change_counter)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prefix_chars='-+',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent('''\
            DCASE 2017
            Rare event detection
            Dataset patcher
            ---------------------------------------------
                Tampere University of Technology / Audio Research Group
                Author:  Aleksandr Diment ( firstname.lastname at tut.fi )

            This is the patched to the TUT-rare-sound-events-2017-development dataset from March 2017 to June 2017 version.
            It fixes the issue of naturally occurring baby cry sounds appearing in the background recordings.
            By running this patch on the March 2017 version of the dataset, the June 2017 dataset used in the challenge is obtained.
            Patching can be performed both on stand-alone dataset and the dataset as structured in the baseline system.
            For more details, see http://www.cs.tut.fi/sgn/arg/dcase2017/

        '''))

    parser.add_argument("-data_path", help="Path to dataset folder TUT-rare-sound-events-2017-development (inside it should be folder "
                                           "data -> mixture_data in standalone mode and data -> generated_data in baseline mode)."
                                           "By default, we assume the script to be placed directly into folder  "
                                           "TUT-rare-sound-events-2017-development and launched from there.",
                        default='.', dest='path_to_dataset')
    args = parser.parse_args()
    path_to_dataset = args.path_to_dataset
    main(path_to_dataset, log='stdout')
