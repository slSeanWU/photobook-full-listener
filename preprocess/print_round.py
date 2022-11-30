def print_round(r):

    def print_images(r, player, img_status):
        print(f'\t{player}:')
        for i, image in enumerate(r.images[player]):
            common = ' (Common)' if image in r.common else '         '
            marked = f' (Marked as {img_status[player][i]})' if r.highlighted[player][i] else ''
            print('\t\t' + image + common + marked)

    print("total score: {}".format(r.total_score))
    print("player scores: A - {}, B - {}".format(r.scores["A"], r.scores["B"]))

    print('\nmessages:')
    msg_list = []
    img_status = {'A': [0, 0, 0, 0, 0, 0], 'B': [0, 0, 0, 0, 0, 0]}
    for i, m in enumerate(r.messages):
        if m.type == "text":
            msg_list.append({'speaker': m.speaker, 'text': m.text})
        if m.type == "selection":
            speaker = m.speaker
            img = m.text.split()[2]
            label = "common" if m.text.split()[1] == "<com>" else "different"
            correct = 'correctly' if (img in r.common and label == 'common') or (
                img not in r.common and label == 'different') else 'incorrectly'
            img_status[speaker][r.images[speaker].index(img)] = label
            if msg_list != []:
                msg_list[-1]['text'] += f' ({speaker} {correct} marked {img} as {label})'

    for d in msg_list:
        speaker = d['speaker']
        text = d['text']
        print(f'\t{speaker}: {text}')

    print('images:')
    print_images(r, 'A', img_status)
    print_images(r, 'B', img_status)
