from transformers import DebertaTokenizer
from load_logs import load_logs


def round2dict(r, tokenizer, player):
    input_ids = []
    token_type_ids = []
    labels = []

    images = r.images[player]
    # 0 = undecided, 1 = common, 2 = different
    image_status = [[0] for _ in images]

    for m in r.messages:
        if m.type == "text":
            tokenized_msg = tokenizer(m.text, padding=False, truncation=True)[
                'input_ids']
            input_ids.append(tokenized_msg)
            token_type_id = 0 if m.speaker == player else 1
            token_type_ids.append([token_type_id for _ in tokenized_msg])
            labels.append([x * len(tokenized_msg) for x in image_status])

        if m.type == "selection" and m.speaker == player:
            img = m.text.split()[2]
            img_index = images.index(img)
            label = 1 if m.text.split()[1] == "" else 2
            image_status[img_index] = [label]

    return {'input_ids': input_ids, 'token_type_ids': token_type_ids, 'labels': labels}


if __name__ == '__main__':
    logs = load_logs("logs", '../data')
    thisGame = logs[0]  # Log, containing 5 rounds
    thisRound = logs[0].rounds[0]
    tokenizer = DebertaTokenizer.from_pretrained("microsoft/deberta-base")
    result = round2dict(thisRound, tokenizer, 'A')
    print(result)
