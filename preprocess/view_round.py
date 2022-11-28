from load_logs import load_logs


def print_round(r):

    def print_images(r, player):
        print(f'\t{player}:')
        for i, image in enumerate(r.images[player]):
            common = ' (Common)' if image in r.common else '         '
            marked = ' (Marked)' if r.highlighted[player][i] else ''
            print('\t\t' + image + common + marked)

    print("total score: {}".format(r.total_score))
    print("player scores: A - {}, B - {}".format(r.scores["A"], r.scores["B"]))

    print('images:')
    print_images(r, 'A')
    print_images(r, 'B')

    print('\nmessages:')
    msg_list = []
    for m in r.messages:
        if m.type == "text":
            msg_list.append({'speaker': m.speaker, 'text': m.text})
        if m.type == "selection":
            speaker = m.speaker
            img = m.text.split()[2]
            label = "common" if m.text.split()[1] == "" else "different"
            correct = 'correctly' if img in r.common else 'incorrectly'
            if msg_list != []:
                msg_list[-1]['text'] += f' ({speaker} {correct} marked {img} as {label})'
    for d in msg_list:
        speaker = d['speaker']
        text = d['text']
        print(f'\t{speaker}: {text}')


if __name__ == '__main__':

    logs = load_logs("logs", '../data')

    thisGame = logs[0]  # Log, containing 5 rounds
    thisRound = logs[0].rounds[0]
    print_round(thisRound)
