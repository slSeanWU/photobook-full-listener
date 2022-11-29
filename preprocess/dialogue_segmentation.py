# Modified from https://github.com/dmg-photobook/photobook_dataset/blob/master/segmentation/utils/dialogue_segmentation.py

from processor import Log
import os
import json
import pickle
import argparse
import random as rd
from collections import defaultdict

import sys
from os.path import dirname, abspath
sys.path.insert(0, dirname(dirname(abspath(__file__))))


round_counter = 0


# Log Loader # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def load_logs(log_repository, data_path):

    filepath = os.path.join(data_path, log_repository)
    print("Loading logs from {}".format(filepath))

    missing_counter = 0
    file_count = 0
    for _, _, files in os.walk(filepath):
        file_count += len(files)
    logs = []
    for root, dirs, files in os.walk(filepath):
        for file in files:
            if file.endswith(".json"):
                with open(os.path.join(root, file), 'r') as logfile:
                    log = Log(json.load(logfile))
                    if log.complete:
                        logs.append(log)
                    else:
                        missing_counter += 1

    print("Complete. Loaded {} completed game logs.".format(len(logs)))
    return logs


# Dataset Splitter # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def generate_game_sets(sample_size, domain_dict, remaining_total):
    """
    Generates a data set of the given sample size by allocating games relative to their domain's frequencies.
    :param sample_size: int. Number of games to be allocated to the set
    :param domain_dict: dict. Dictionary linking domain IDs and the IDs of all games in that domain
    :param remaining_total: Total number of games remaining in the domain_dict.
    :return: [list, dict, int]. List of game_ids, updated domain_dict and total number of games remaining in the domain_dict
    """
    game_set = []
    sampled_games = 0

    for domain_id, games in domain_dict.items():
        domain_sample_size = int(
            len(games) / remaining_total * sample_size + 0.5)
        sampled_games += domain_sample_size
        game_set.extend([(domain_id, game_id)
                        for game_id in rd.sample(games, domain_sample_size)])

    for domain_id, game_id in game_set:
        games = domain_dict[domain_id]
        games.remove(game_id)
        domain_dict[domain_id] = games

    game_list = [tup[1] for tup in game_set]
    remaining_total = remaining_total - sampled_games

    return game_list, domain_dict, remaining_total


# Dialogue Segmentation Heuristics # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def clean_clicks(round_data):
    """
    Removes duplicate clicks from the dialogues recorded in the passed round data object
    :param round_data: Round object.
    :return: A Round object that is cleared of multiple clicks
    """
    filtered_data = []
    speaker_selections = dict({"A": [], "B": []})
    messages = round_data.messages.copy()
    messages.reverse()

    cleaning_counter = 0

    for message in messages:
        if message.type == 'selection':
            previous_selections = speaker_selections[message.speaker]
            target = message.text.split()[2]
            if target in previous_selections:
                cleaning_counter += 1
                continue
            else:
                speaker_selections[message.speaker].append(target)
                filtered_data.append(message)
        else:
            assert "<selection>" not in message.text
            filtered_data.append(message)

    filtered_data.reverse()
    return filtered_data, cleaning_counter


def is_selection(message):
    """
    Returns True if the passed message is a labeling action
    :param message: Message object.
    :return: True if the passed message is a labeling action
    """
    if len(message.text.split()) == 3 and message.text.split()[0] == '<selection>':
        return True
    return False


def is_common_label(message):
    """
    Returns True if the passed message is a selection and the target image was marked common
    Returns False if the message is no selection or the target was marked different
    :param message: Message object.
    :return: True if the passed message is a selection and the target image was marked common
    """
    if not message.type == 'selection':
        print("Received message not a selection!")
        return False

    if message.text.split()[1] == '<com>':
        return True
    else:
        return False


def get_target(message):
    """
    Returns the target identifier from a message or None if the message was no selection
    :param message: Message object.
    :return: The target identifier of the message or None if the message was no selection
    """
    if not len(message.text.split()) == 3:
        print("Received message not a selection!")
        return None
    return tuple([message.agent_id] + message.text.split()[1:])


def dialogue_segmentation(logs, selection, seg_verbose=False):
    """
    sections the dialogues in the game rounds based on a pre-defined heuristics
    :param logs: list. List containing the Log objects created from the log files
    :param selection: list. List containing the set of game indexes to be included in the current split
    :param seg_verbose: bool. Set to True to print the decision structure
    :return: A list of lists containing tuples of dialogue segments and their corresponding targets for the games in the given set
    """
    cleaning_total = 0
    section_counter = 0

    dialogue_sections = []
    for game in logs:
        game_id = game.game_id
        if selection and game_id not in selection:
            continue
        game_sections, cleaning_total, section_counter = game_segmentation(
            game, seg_verbose, cleaning_total, section_counter)

        dialogue_sections.append((game_id, game_sections))

    if seg_verbose:
        print("Total of {} duplicate labeling action(s) removed.".format(cleaning_total))
    if seg_verbose:
        print("Processed {} dialogue(s).".format(len(dialogue_sections)))
    if seg_verbose:
        print("Generated a total of {} dialogue section(s).".format(section_counter))

    return dialogue_sections

# NOTE: add this function


def parse(section):
    # a list of messages
    ret = []
    for msg in section:
        ret.append((msg.agent_id, msg.text))
    return ret


def game_segmentation(game, seg_verbose, cleaning_total, section_counter):
    game_sections = []
    agent_ids = game.agent_ids

    for round_data in game.rounds:

        # NOTE (Shih-Lun): filter out rounds with mistakes
        if round_data.total_score < 6:
            continue

        selections = []
        messages = round_data.messages

        if seg_verbose:
            print("\n")
        for message in round_data.messages:
            if seg_verbose:
                print("{}: {}".format(message.speaker, message.text))
            if message.type == 'selection':
                selections.append(
                    (message.message_id, message.speaker, message.text))
        if seg_verbose:
            print("\n")

        if len(selections) > 6:
            messages, cleaning_counter = clean_clicks(round_data)
            cleaning_total += cleaning_counter

        selections = []
        for message in messages:
            if message.type == 'selection':
                selections.append(
                    (message.message_id, message.speaker, message.text))

        # NOTE (Shih-Lun): filter out rounds with >6 answers (rare exceptions)
        if len(selections) != 6:
            continue

        global round_counter
        round_counter += 1

        # NOTE: a dictionary per round
        sections = {'segments': [], 'image_set': dict(), 'targets': [],
                    'roundnr': round_data.round_nr, 'gameid': game.game_id,
                    'rounddata': round_data}

        i = 0
        while i < len(messages):
            message = messages[i]
            current_section = []
            current_targets = []
            if message.type == 'text':
                current_section.append(message)
                i += 1
                while messages[i].type == 'selection':
                    current_targets.append(get_target(messages[i]))
                    i += 1
                i -= 1
                sections['segments'] += parse(current_section)
                sections['targets'].append(
                    (current_section[0], set(current_targets)))
            i += 1

        # NOTE: separate image set of 2 players
        sections['image_set'] = {
            agent_ids[0]: round_data.images["A"],
            agent_ids[1]: round_data.images["B"]
        }
        game_sections.append(sections)
        if seg_verbose:
            print("{} dialogue sections encountered in round".format(
                len(sections['targets'])))

        assert sum([len(sections['targets'][i][1]) for i in range(len(sections['targets']))]) == 6, \
            sum([len(sections['targets'][i][1])
                for i in range(len(sections['targets']))])

        section_counter += len(sections['targets'])

    return game_sections, cleaning_total, section_counter


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-data_path", type=str, default="../data")
    parser.add_argument("-new_split", type=bool, default=False)
    parser.add_argument("-split", type=list, default=[15, 15])

    args = parser.parse_args()
    data_path = args.data_path
    new_split = args.new_split
    split = list(args.split)
    if len(split) != 2:
        print("Alert: -split argument takes a list of length 2 with validation and test size in %. Using default 15/15/70 split.")
        split = [15, 15]

    logs_dir = "logs/"
    logs = load_logs(logs_dir, data_path)

    # Create a new split
    if new_split:
        val_size = int(split[0]/100 * len(logs))
        test_size = int(split[1]/100 * len(logs))

        domain_dict = defaultdict(lambda: [])
        for game in logs:
            domain_dict[game.domain_id].append(game.game_id)

        data_split = dict()
        remaining_total = len(logs)
        data_split["dev"], domain_dict, remaining_total = generate_game_sets(
            60, domain_dict, remaining_total)
        data_split["val"], domain_dict, remaining_total = generate_game_sets(
            val_size, domain_dict, remaining_total)
        data_split["test"], domain_dict, remaining_total = generate_game_sets(
            val_size, domain_dict, remaining_total)

        train_set = []
        for domain_id, games in domain_dict.items():
            train_set.extend(games)

        data_split["train"] = train_set

        with open(os.path.join(data_path, "new_data_splits.json"), 'w') as f:
            json.dump(data_split, f)

    # Load a pre-defined split
    else:
        with open(os.path.join(data_path, "data_splits.json"), 'r') as f:
            data_split = json.load(f)

    print("Development set contains {} games".format(len(data_split["dev"])))
    print("Validation set contains {} games".format(len(data_split["val"])))
    print("Test set contains {} games".format(len(data_split["test"])))
    print("Train set contains {} games".format(len(data_split["train"])))

    for set_name in ['dev', 'val', 'test', 'train']:
        set_ids = data_split[set_name]
        dialogue_sections = dialogue_segmentation(
            logs, set_ids, seg_verbose=False)
        with open(os.path.join(data_path, "{}_sections.pickle".format(set_name)), 'wb') as f:
            pickle.dump(dialogue_sections, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(
        f"Got {round_counter} out of {5 * len(logs)} rounds without mistake (total_score == 6)")

    print("Done.")
