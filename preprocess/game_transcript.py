from load_logs import load_logs
from processor import Log


def print_transcript(log):
    print("Game ID: {}".format(log.game_id))
    print("Domain ID: {}".format(log.domain_id))
    print("Image set main objects: '{}' and '{}'".format(
        log.domains[0], log.domains[1]))
    print("Participant IDs: {} and {}".format(
        log.agent_ids[0], log.agent_ids[1]))
    print("Start Time: {}".format(log.start_time))
    print("Duration: {}".format(log.duration))
    print("Total Score: {}".format(log.total_score))
    print(
        "Player scores: A - {}, B - {}".format(log.scores["A"], log.scores["B"]))
    print("Transcript:\n")

    for round_data in log.rounds:
        print("Round {}".format(round_data.round_nr))
        for message in round_data.messages:
            if message.type == "text":
                print("[{}] {}: {}".format(Log.format_time(
                    message.timestamp), message.speaker, message.text))

            if message.type == "selection":
                label = "common" if message.text.split()[
                    1] == "" else "different"
                print("[{}] {} marks image {} as {}".format(Log.format_time(
                    message.timestamp), message.speaker, Log.strip_image_id(message.text.split()[2]), label))

        print("\nDuration: {}".format(round_data.duration))
        print("Total Score: {}".format(round_data.total_score))
        print(
            "Player scores: A - {}, B - {}".format(round_data.scores["A"], round_data.scores["B"]))
        print("Number of messages: {}\n".format(round_data.num_messages))


if __name__ == '__main__':

    logs = load_logs("logs", '../data')

    thisGame = logs[0]  # Log, containing 5 rounds
    print_transcript(thisGame)
